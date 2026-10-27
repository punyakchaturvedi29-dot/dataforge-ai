"""
Forecasting engine with time-aware evaluation.

Workflow
--------
1. Build a regular series from a date column and a numeric measure (or record
   counts) at a chosen frequency; report duplicate timestamps (aggregated) and
   missing periods (filled with an explicit, reported policy).
2. Fit candidate methods: naive, seasonal naive, drift, simple exponential
   smoothing, damped Holt, Holt-Winters, ARIMA (small AIC search) and a
   trend + seasonal-dummies regression.
3. Evaluate every method with **rolling-origin backtesting**: each fold trains
   only on data before the forecast origin, so no future information is used.
   Metrics: MAE, RMSE, MASE (scaled by the naive in-sample error) and sMAPE.
4. Select the method with the lowest mean MASE, refit on the full series and
   forecast the horizon with approximate prediction intervals. The empirical
   coverage of the 80 % intervals during backtesting is reported so users can
   judge whether the intervals are trustworthy.
"""
from __future__ import annotations

import math
import time
import warnings
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy import stats as sps

from core.errors import ValidationFailed
from core.jsonsafe import to_jsonable

from . import stats as st
from .schema import analysis_columns, infer_schema

try:
    from statsmodels.tsa.holtwinters import ExponentialSmoothing, SimpleExpSmoothing
    from statsmodels.tsa.statespace.sarimax import SARIMAX
    from statsmodels.tsa.stattools import adfuller
    HAS_SM = True
except Exception:  # pragma: no cover
    HAS_SM = False

SEASON = {"h": 24, "D": 7, "W": 52, "MS": 12, "QS": 4, "YS": 1}
MIN_POINTS = 8
MAX_POINTS = 1500
Z80, Z95 = sps.norm.ppf(0.9), sps.norm.ppf(0.975)
DESCRIPTIONS = {
    "naive": "Naive: repeats the last observed value (baseline).",
    "seasonal_naive": "Seasonal naive: repeats the value from the same season one cycle earlier.",
    "drift": "Drift: extends the line between the first and last observation.",
    "ses": "Simple exponential smoothing: weighted average favouring recent values; flat forecast.",
    "holt": "Holt's method with damped trend: level + trend that flattens over time.",
    "holt_winters": "Holt-Winters: level, damped trend and additive seasonality.",
    "arima": "ARIMA: autoregressive model with the order chosen by AIC.",
    "trend_regression": "Linear trend + seasonal dummies fitted by least squares.",
}
LABELS = {"naive": "Naive", "seasonal_naive": "Seasonal naive", "drift": "Drift", "ses": "Exponential smoothing",
          "holt": "Holt (damped trend)", "holt_winters": "Holt-Winters", "arima": "ARIMA",
          "trend_regression": "Trend + seasonality regression"}


# ── Candidate discovery ───────────────────────────────────────────────────────

def forecast_options(df: pd.DataFrame, schema: Optional[List[Dict]] = None) -> Dict[str, Any]:
    schema = schema or infer_schema(df)
    cols = analysis_columns(df, schema)
    smap = {c["name"]: c for c in schema}
    dates = []
    for d in cols["datetime"]:
        freq, info = st.infer_frequency(df[d])
        dates.append({"column": d, "frequency": info.get("native_frequency"), "chart_frequency": freq,
                      "span_days": info.get("span_days"), "n_unique_timestamps": info.get("n_unique_timestamps"),
                      "missing": int(df[d].isna().sum())})
    measures = [{"column": c, "default_agg": st.default_agg(c, smap[c]["semantic_type"]),
                 "semantic_type": smap[c]["semantic_type"]} for c in cols["numeric"]]
    reason = None
    if not dates:
        text_dates = [c["name"] for c in schema if "datetime_string" in c.get("hints", [])]
        reason = ("No date/time column was found." + (f" '{text_dates[0]}' looks like dates stored as text — convert it "
                  "in the Clean step." if text_dates else " Forecasting needs a column with timestamps."))
    elif not measures:
        reason = "No numeric measure is available; you can still forecast record counts."
    return {"available": bool(dates), "date_columns": dates, "measures": measures, "reason": reason,
            "frequencies": [{"id": k, "label": v} for k, v in st.FREQ_LABELS.items()],
            "aggregations": ["sum", "mean", "median", "count", "last", "min", "max"]}


# ── Series construction ────────────────────────────────────────────────────────

def build_series(df: pd.DataFrame, date_col: str, target: Optional[str], freq: str, agg: str,
                 fill: str = "auto") -> Tuple[pd.Series, Dict[str, Any]]:
    if date_col not in df.columns:
        raise ValidationFailed(f"Unknown date column '{date_col}'")
    if not pd.api.types.is_datetime64_any_dtype(df[date_col]):
        raise ValidationFailed(f"'{date_col}' is not a datetime column. Convert it in the Clean step first.")
    if target is not None:
        if target not in df.columns:
            raise ValidationFailed(f"Unknown column '{target}'")
        if not pd.api.types.is_numeric_dtype(df[target]) or pd.api.types.is_bool_dtype(df[target]):
            raise ValidationFailed(f"'{target}' must be numeric (or forecast record counts instead).")
    if agg not in ("sum", "mean", "median", "count", "last", "min", "max"):
        raise ValidationFailed("Unknown aggregation")
    d = df[[date_col] + ([target] if target else [])].dropna(subset=[date_col])
    diag: Dict[str, Any] = {"rows_used": int(len(d)), "rows_missing_date": int(df[date_col].isna().sum())}
    if target:
        diag["rows_missing_value"] = int(d[target].isna().sum())
    ts_counts = d[date_col].value_counts()
    diag["duplicate_timestamps"] = int((ts_counts > 1).sum())
    diag["records_per_timestamp_max"] = int(ts_counts.max()) if len(ts_counts) else 0
    counts = st.aggregate_series(d, date_col, None, freq, "count")
    s = st.aggregate_series(d, date_col, target, freq, "count" if target is None else agg)
    s = s.astype(float)
    empty = counts == 0
    diag["periods"] = int(len(s))
    diag["empty_periods"] = int(empty.sum())
    policy = fill
    if fill == "auto":
        policy = "zero" if agg in ("sum", "count") else "interpolate"
    if empty.any():
        if policy == "zero":
            s[empty] = 0.0
        elif policy == "interpolate":
            s[empty] = np.nan
            s = s.interpolate(limit_direction="both")
        elif policy == "ffill":
            s[empty] = np.nan
            s = s.ffill().bfill()
        else:
            raise ValidationFailed("fill must be auto, zero, interpolate or ffill")
    diag["fill_policy"] = policy if empty.any() else "none needed"
    diag["filled_flags"] = empty.tolist()
    # Partial last period (fewer than 50% of median records) is reported, not removed.
    if len(counts) >= 4:
        med = counts[counts > 0].median()
        diag["last_period_partial"] = bool(counts.iloc[-1] < 0.5 * med)
        diag["first_period_partial"] = bool(counts.iloc[0] < 0.5 * med)
    else:
        diag["last_period_partial"] = diag["first_period_partial"] = False
    return s, diag


# ── Forecast methods: f(train, h, m) -> (point, sigma1, fitted_residuals, interval_fn) ─────

def _resid_sigma(resid: np.ndarray) -> float:
    r = resid[np.isfinite(resid)]
    return float(np.sqrt(np.mean(r ** 2))) if len(r) else 0.0


def m_naive(y, h, m):
    res = np.diff(y)
    s = _resid_sigma(res)
    return np.repeat(y[-1], h), lambda j: s * math.sqrt(j)


def m_seasonal_naive(y, h, m):
    if m <= 1 or len(y) <= m:
        raise ValueError("needs at least one full season")
    fc = np.array([y[len(y) - m + (j % m)] for j in range(h)])
    res = y[m:] - y[:-m]
    s = _resid_sigma(res)
    return fc, lambda j: s * math.sqrt((j - 1) // m + 1)


def m_drift(y, h, m):
    n = len(y)
    slope = (y[-1] - y[0]) / max(n - 1, 1)
    fc = y[-1] + slope * np.arange(1, h + 1)
    res = np.diff(y) - slope
    s = _resid_sigma(res)
    return fc, lambda j: s * math.sqrt(j * (1 + j / max(n, 1)))


def _ses_own(y, h):
    best = (None, float("inf"))
    for a in np.linspace(0.05, 0.95, 19):
        lvl, err = y[0], []
        for v in y[1:]:
            err.append(v - lvl)
            lvl = a * v + (1 - a) * lvl
        sse = float(np.sum(np.square(err)))
        if sse < best[1]:
            best = ((a, lvl, np.array(err)), sse)
    a, lvl, err = best[0]
    return np.repeat(lvl, h), err, a


def m_ses(y, h, m):
    if HAS_SM and len(y) >= 6:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            fit = SimpleExpSmoothing(y, initialization_method="estimated").fit()
        fc = np.asarray(fit.forecast(h))
        res = y - np.asarray(fit.fittedvalues)
        a = float(fit.params.get("smoothing_level", 0.5))
    else:
        fc, res, a = _ses_own(y, h)
    s = _resid_sigma(res[1:] if len(res) > 1 else res)
    return fc, lambda j: s * math.sqrt(1 + (j - 1) * a ** 2)


def m_holt(y, h, m):
    if not HAS_SM or len(y) < 10:
        raise ValueError("needs statsmodels and at least 10 points")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        fit = ExponentialSmoothing(y, trend="add", damped_trend=True, initialization_method="estimated").fit()
    fc = np.asarray(fit.forecast(h))
    s = _resid_sigma(y - np.asarray(fit.fittedvalues))
    return fc, lambda j: s * math.sqrt(j)


def m_holt_winters(y, h, m):
    if not HAS_SM or m <= 1 or len(y) < 2 * m + 2:
        raise ValueError("needs at least two full seasons")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        fit = ExponentialSmoothing(y, trend="add", damped_trend=True, seasonal="add", seasonal_periods=m,
                                   initialization_method="estimated").fit()
    fc = np.asarray(fit.forecast(h))
    s = _resid_sigma(y - np.asarray(fit.fittedvalues))
    return fc, lambda j: s * math.sqrt(1 + (j - 1) * 0.3)


def m_arima(y, h, m):
    if not HAS_SM or len(y) < 15 or len(y) > 600:
        raise ValueError("ARIMA runs for series of 15–600 points")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        try:
            d = 1 if adfuller(y, autolag="AIC")[1] > 0.05 else 0
        except Exception:
            d = 1
        best = None
        for order in [(1, d, 0), (0, d, 1), (1, d, 1), (2, d, 0), (0, d, 2), (2, d, 1)]:
            try:
                fit = SARIMAX(y, order=order, trend="c" if d == 0 else "n", enforce_stationarity=False,
                              enforce_invertibility=False).fit(disp=False, maxiter=50)
            except Exception:
                continue
            if best is None or fit.aic < best[1].aic:
                best = (order, fit)
    if best is None:
        raise ValueError("no ARIMA order converged")
    order, fit = best
    f = fit.get_forecast(h)
    fc = np.asarray(f.predicted_mean)
    se = np.asarray(f.se_mean)
    m_arima.last_order = order
    return fc, lambda j: float(se[min(j, len(se)) - 1])


def m_trend_regression(y, h, m):
    n = len(y)
    if n < 8:
        raise ValueError("needs at least 8 points")
    t = np.arange(n)
    use_season = m > 1 and n >= 2 * m
    def design(idx):
        cols = [np.ones(len(idx)), idx]
        if use_season:
            for k in range(1, m):
                cols.append(((idx % m) == k).astype(float))
        return np.column_stack(cols)
    X = design(t)
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    resid = y - X @ beta
    p = X.shape[1]
    s2 = float(np.sum(resid ** 2) / max(n - p, 1))
    Xf = design(np.arange(n, n + h))
    fc = Xf @ beta
    try:
        XtXi = np.linalg.pinv(X.T @ X)
        se = np.sqrt(s2 * (1 + np.einsum("ij,jk,ik->i", Xf, XtXi, Xf)))
    except Exception:
        se = np.repeat(math.sqrt(s2), h)
    return fc, lambda j: float(se[j - 1])


METHODS: Dict[str, Callable] = {
    "naive": m_naive, "seasonal_naive": m_seasonal_naive, "drift": m_drift, "ses": m_ses, "holt": m_holt,
    "holt_winters": m_holt_winters, "arima": m_arima, "trend_regression": m_trend_regression,
}


def _mase_scale(train: np.ndarray, m: int) -> float:
    if m > 1 and len(train) > m:
        d = np.abs(train[m:] - train[:-m])
    else:
        d = np.abs(np.diff(train))
    scale = float(np.mean(d)) if len(d) else 0.0
    return scale if scale > 1e-12 else float(np.mean(np.abs(train - train.mean())) or 1.0)


def backtest(y: np.ndarray, h: int, m: int, methods: List[str], k: int = 3) -> Tuple[Dict[str, Any], List[int]]:
    n = len(y)
    min_train = max(6, 2 * m + 2 if m > 1 and n >= 3 * m else 6)
    while k > 1 and n - k * h < min_train:
        k -= 1
    if n - k * h < min_train:
        raise ValidationFailed(f"The series is too short to backtest a {h}-period forecast "
                               f"({n} periods; need at least {min_train + h}). Reduce the horizon or use a finer frequency.")
    origins = [n - (k - i) * h for i in range(k)]
    out: Dict[str, Any] = {}
    for name in methods:
        fn = METHODS[name]
        folds, errs, covered, total = [], [], 0, 0
        ok = True
        for o in origins:
            tr, te = y[:o], y[o:o + h]
            try:
                fc, sig = fn(tr, len(te), m)
            except Exception:
                ok = False
                break
            e = te - fc[: len(te)]
            scale = _mase_scale(tr, m)
            denom = np.abs(te) + np.abs(fc[: len(te)])
            smape = float(np.mean(np.where(denom > 0, 2 * np.abs(e) / np.where(denom > 0, denom, 1), 0)) * 100)
            lo = fc[: len(te)] - Z80 * np.array([sig(j + 1) for j in range(len(te))])
            hi = fc[: len(te)] + Z80 * np.array([sig(j + 1) for j in range(len(te))])
            covered += int(np.sum((te >= lo) & (te <= hi)))
            total += len(te)
            folds.append({"origin": int(o), "mae": float(np.mean(np.abs(e))), "rmse": float(np.sqrt(np.mean(e ** 2))),
                          "mase": float(np.mean(np.abs(e)) / scale), "smape": smape,
                          "actual": te.tolist(), "forecast": fc[: len(te)].tolist()})
            errs.append(e)
        if not ok or not folds:
            out[name] = {"status": "unavailable"}
            continue
        out[name] = {"status": "ok", "folds": folds,
                     "mae": float(np.mean([f["mae"] for f in folds])), "rmse": float(np.mean([f["rmse"] for f in folds])),
                     "mase": float(np.mean([f["mase"] for f in folds])), "smape": float(np.mean([f["smape"] for f in folds])),
                     "coverage80": covered / total if total else None}
    return out, origins


def run_forecast(df: pd.DataFrame, date_col: str, target: Optional[str], horizon: int = 6, agg: Optional[str] = None,
                 freq: Optional[str] = None, fill: str = "auto", methods: Optional[List[str]] = None,
                 n_backtest: int = 3, schema: Optional[List[Dict]] = None) -> Dict[str, Any]:
    t0 = time.time()
    schema = schema or infer_schema(df)
    smap = {c["name"]: c for c in schema}
    if not isinstance(horizon, int) or horizon < 1 or horizon > 365:
        raise ValidationFailed("horizon must be an integer between 1 and 365")
    if date_col not in df.columns:
        raise ValidationFailed(f"Unknown date column '{date_col}'")
    auto_freq, finfo = st.infer_frequency(df[date_col]) if pd.api.types.is_datetime64_any_dtype(df[date_col]) else ("D", {})
    auto = freq in (None, "", "auto")
    if auto:
        freq = finfo.get("native_frequency", auto_freq)
        if freq == "h" and finfo.get("span_days", 0) > 120:
            freq = "D"
        if freq == "D" and finfo.get("span_days", 0) > 3 * 365:
            freq = "W"
    if freq not in SEASON:
        raise ValidationFailed("freq must be one of h, D, W, MS, QS, YS or auto")
    if agg is None:
        agg = st.default_agg(target, smap.get(target, {}).get("semantic_type")) if target else "count"
    s, diag = build_series(df, date_col, target, freq, agg, fill)
    warnings_list: List[str] = []
    if auto:
        ladder = ["h", "D", "W", "MS", "QS"]
        while freq in ladder[:-1] and diag["empty_periods"] / max(len(s), 1) > 0.3:
            nxt = ladder[ladder.index(freq) + 1]
            s2, d2 = build_series(df, date_col, target, nxt, agg, fill)
            if len(s2) < 12:
                break
            adj = {"D": "daily", "W": "weekly", "MS": "monthly", "QS": "quarterly"}[nxt]
            warnings_list.append(f"Aggregated to {adj} periods because "
                                 f"{diag['empty_periods']} of {len(s)} {st.FREQ_LABELS[freq]}s had no records.")
            freq, s, diag = nxt, s2, d2
    if len(s) > MAX_POINTS:
        s = s.iloc[-MAX_POINTS:]
        diag["filled_flags"] = diag["filled_flags"][-MAX_POINTS:]
        warnings_list.append(f"Only the most recent {MAX_POINTS} periods were used.")
    n = len(s)
    if n < MIN_POINTS:
        return {"available": False, "reason": f"Only {n} {st.FREQ_LABELS[freq]}(s) of data after aggregation; at least "
                                              f"{MIN_POINTS} are needed. Try a finer frequency.", "diagnostics": to_jsonable(diag)}
    if diag["empty_periods"] / max(n, 1) > 0.3:
        warnings_list.append(f"{diag['empty_periods']} of {n} periods had no records and were filled "
                             f"({diag['fill_policy']}). Forecasts may be unreliable; consider a coarser frequency.")
    if diag["duplicate_timestamps"]:
        warnings_list.append(f"{diag['duplicate_timestamps']:,} timestamps have several records; they were aggregated "
                             f"with '{agg}'.")
    if diag.get("last_period_partial"):
        warnings_list.append("The last period has far fewer records than usual and is probably incomplete; "
                             "it may pull the forecast down.")
    max_h = max(1, n // 3)
    if horizon > max_h:
        warnings_list.append(f"The horizon was reduced from {horizon} to {max_h} periods (one third of the history).")
        horizon = max_h
    m = SEASON[freq]
    if m > 1 and n < 2 * m:
        warnings_list.append(f"Less than two full seasonal cycles ({2 * m} {st.FREQ_LABELS[freq]}s) of history: "
                             "seasonal methods are not evaluated.")
    y = s.to_numpy(dtype=float)
    m_eff = m if n >= 2 * m else 1

    def label(k: str) -> str:
        return "Linear trend regression" if k == "trend_regression" and m_eff == 1 else LABELS[k]

    chosen = methods or list(METHODS)
    bad = [x for x in chosen if x not in METHODS]
    if bad:
        raise ValidationFailed(f"Unknown method(s): {', '.join(bad)}")
    if "naive" not in chosen:
        chosen = ["naive"] + chosen
    if m <= 1 or n < 2 * m:
        chosen = [x for x in chosen if x not in ("seasonal_naive", "holt_winters")]
    bt, origins = backtest(y, horizon, m if n >= 2 * m else 1, chosen, n_backtest)
    ok = {k: v for k, v in bt.items() if v["status"] == "ok"}
    naive_mase = ok.get("naive", {}).get("mase")
    ranking = sorted(ok, key=lambda k: (ok[k]["mase"], list(METHODS).index(k)))
    best = ranking[0]
    nonneg = bool(np.all(y >= 0))
    fut_index = pd.date_range(s.index[-1], periods=horizon + 1, freq=freq)[1:]
    forecasts: Dict[str, Any] = {}
    for name in ranking:
        try:
            fc, sig = METHODS[name](y, horizon, m if n >= 2 * m else 1)
        except Exception:
            continue
        sd = np.array([sig(j) for j in range(1, horizon + 1)])
        lo80, hi80 = fc - Z80 * sd, fc + Z80 * sd
        lo95, hi95 = fc - Z95 * sd, fc + Z95 * sd
        if nonneg:
            lo80, lo95, fc = np.maximum(lo80, 0), np.maximum(lo95, 0), np.maximum(fc, 0)
        forecasts[name] = {"forecast": fc.tolist(), "lower80": lo80.tolist(), "upper80": hi80.tolist(),
                           "lower95": lo95.tolist(), "upper95": hi95.tolist()}
    comparison = []
    for name in METHODS:
        if name not in bt:
            continue
        r = bt[name]
        row = {"method": name, "label": label(name), "description": DESCRIPTIONS[name], "status": r["status"]}
        if r["status"] == "ok":
            row.update({k: r[k] for k in ("mae", "rmse", "mase", "smape", "coverage80")})
            row["skill_vs_naive_pct"] = (1 - r["mase"] / naive_mase) * 100 if naive_mase else None
            row["selected"] = name == best
        if name == "arima" and getattr(m_arima, "last_order", None):
            row["order"] = list(m_arima.last_order)
        comparison.append(row)
    sel = ok[best]
    skill = (1 - sel["mase"] / naive_mase) * 100 if naive_mase else None
    explanation = [
        f"{label(best)} had the lowest average scaled error (MASE {sel['mase']:.2f}) over {len(origins)} backtest "
        f"fold(s) of {horizon} {st.FREQ_LABELS[freq]}(s) each.",
        (f"It is {skill:.0f}% more accurate than the naive baseline." if skill is not None and skill > 1 else
         "It is not meaningfully better than the naive baseline, so the series is hard to forecast beyond its last value."),
        "MASE < 1 means the method beats a naive forecast made one step at a time on the training data.",
    ]
    cov = sel.get("coverage80")
    if cov is not None:
        explanation.append(f"In backtesting, {cov:.0%} of actual values fell inside the 80% intervals "
                           + ("— the intervals look reasonably calibrated." if 0.65 <= cov <= 0.95 else
                              "— the intervals are " + ("too narrow" if cov < 0.65 else "wider than needed") +
                              "; treat them with caution."))
    limitations = ["Forecasts extrapolate past patterns; they cannot anticipate events or changes not reflected in the history.",
                   "Prediction intervals are approximate and assume the future behaves like the past.",
                   "Uncertainty grows with the horizon; later periods are less reliable."]
    if n < 24:
        limitations.append(f"The history is short ({n} periods), so model selection is uncertain.")
    history = [{"t": t.isoformat(), "value": to_jsonable(v), "filled": bool(f)}
               for t, v, f in zip(s.index, s.to_numpy(), diag["filled_flags"])]
    diag.pop("filled_flags", None)
    return to_jsonable({
        "available": True, "date_column": date_col, "target": target, "aggregation": agg, "frequency": freq,
        "frequency_label": st.FREQ_LABELS[freq], "seasonal_period": m if n >= 2 * m else None, "horizon": horizon,
        "history": history, "future_index": [t.isoformat() for t in fut_index], "forecasts": forecasts,
        "selected_method": best, "selected_label": label(best), "comparison": comparison,
        "backtest": {"origins": [s.index[o].isoformat() for o in origins], "folds_selected": sel["folds"],
                     "n_folds": len(origins)},
        "diagnostics": diag, "warnings": warnings_list, "explanation": explanation, "limitations": limitations,
        "non_negative_clipping": nonneg, "duration_s": round(time.time() - t0, 2),
    })


# ── v4 compatibility ────────────────────────────────────────────────────────────

def run_forecasting(df: pd.DataFrame, eda: Dict, horizon: int = 6) -> Dict[str, Any]:
    schema = infer_schema(df)
    opts = forecast_options(df, schema)
    if not opts["available"]:
        return {"available": False, "reason": opts["reason"]}
    d = opts["date_columns"][0]["column"]
    from .schema import rank_measures
    smap = {c["name"]: c for c in schema}
    measures = rank_measures([m["column"] for m in opts["measures"]], smap)[:3]
    cols = {}
    for mcol in measures:
        try:
            r = run_forecast(df, d, mcol, horizon=horizon, schema=schema, methods=["naive", "drift", "ses", "holt", "trend_regression"])
        except ValidationFailed:
            continue
        if not r.get("available"):
            continue
        models = {k: {"model": LABELS[k], "forecast": v["forecast"], "lower": v["lower95"], "upper": v["upper95"],
                      "rmse": next((c.get("rmse") for c in r["comparison"] if c["method"] == k), None)}
                  for k, v in r["forecasts"].items()}
        cols[mcol] = {"historical_dates": [h["t"][:10] for h in r["history"]],
                      "historical_values": [h["value"] for h in r["history"]],
                      "future_dates": [t[:10] for t in r["future_index"]], "models": models,
                      "best_model": r["selected_method"], "horizon": r["horizon"],
                      "summary": " ".join(r["explanation"][:2])}
    if not cols:
        return {"available": False, "reason": "Not enough history to forecast."}
    return {"available": True, "columns": cols, "horizon": horizon}
