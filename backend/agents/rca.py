"""
Diagnostics ("root-cause analysis") with explicit evidence and caveats.

Observational data can rarely prove a cause. Every result here is therefore
labelled as one of:

* ``observation``          – a computed fact (e.g. the metric changed by 12 %),
* ``potential driver``     – a statistically supported association that remains
                             after the checks we can run (multiple-testing
                             correction, confounder stratification),
* ``diagnostic hypothesis``– a plausible explanation that needs validation.

Analyses
--------
1. Segment differences (Kruskal–Wallis, η², best-vs-worst effect sizes) with a
   stratification check against the strongest potential confounder.
2. Numeric relationships (Spearman) with partial correlations.
3. Period-over-period change with an exact segment contribution breakdown and a
   mix-vs-rate decomposition.
4. A cross-validated driver model with permutation importance (predictive
   association only).
5. High- vs low-performer comparison (standardised mean differences, lift).
6. Outcome anomalies (robust z-scores) per record and per period.
"""
from __future__ import annotations

import math
import warnings
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from core.errors import ValidationFailed
from core.jsonsafe import to_jsonable

from . import stats as st
from .schema import analysis_columns, infer_schema, rank_measures

CAUSAL_NOTE = ("These results describe associations in observational data. They can point to potential drivers "
               "but cannot prove causation; validate with experiments, domain knowledge or controlled comparisons.")


def _fmt(v: Optional[float]) -> str:
    if v is None:
        return "—"
    return f"{v:,.2f}" if abs(v) < 1e4 else f"{v:,.0f}"


def _prepare_outcome(df: pd.DataFrame, outcome: str, positive: Optional[str], smap) -> Tuple[pd.Series, str, str]:
    if outcome not in df.columns:
        raise ValidationFailed(f"Unknown outcome column '{outcome}'")
    s = df[outcome]
    if pd.api.types.is_numeric_dtype(s) and not pd.api.types.is_bool_dtype(s) and s.nunique() > 2:
        return s.astype(float), "mean", "numeric"
    vals = s.dropna().astype(str)
    levels = vals.value_counts()
    if len(levels) < 2:
        raise ValidationFailed("The outcome has fewer than two distinct values.")
    if len(levels) > 2 and positive is None:
        raise ValidationFailed("For a categorical outcome with more than two values, choose which value counts as "
                               "the 'positive' outcome (e.g. churned = Yes).", code="needs_positive_class",
                               details={"levels": levels.index[:20].tolist()})
    pos = positive if positive is not None else (levels.index[1] if len(levels) == 2 else levels.index[0])
    if pos not in set(levels.index):
        raise ValidationFailed(f"'{pos}' is not a value of '{outcome}'")
    y = s.astype("string").map(lambda v: np.nan if pd.isna(v) else float(v == pos)).astype(float)
    return y, "rate", f"share of rows where {outcome} = {pos}"


def run_diagnostics(df: pd.DataFrame, outcome: str, schema: Optional[List[Dict]] = None, positive: Optional[str] = None,
                    dimensions: Optional[List[str]] = None, date_col: Optional[str] = None,
                    period_a: Optional[List[str]] = None, period_b: Optional[List[str]] = None,
                    random_state: int = 42) -> Dict[str, Any]:
    schema = schema or infer_schema(df)
    smap = {c["name"]: c for c in schema}
    cols = analysis_columns(df, schema)
    y, kind, desc = _prepare_outcome(df, outcome, positive, smap)
    additive = kind == "mean" and st.default_agg(outcome, smap[outcome]["semantic_type"]) == "sum"
    data = df.assign(__y=y).dropna(subset=["__y"])
    if len(data) < 20:
        raise ValidationFailed("At least 20 rows with an outcome value are needed for diagnostics.")
    dims = dimensions or [c for c in cols["categorical"] if c != outcome and 2 <= smap[c]["n_unique"] <= 50]
    bad = [d for d in dims if d not in df.columns]
    if bad:
        raise ValidationFailed(f"Unknown dimension(s): {', '.join(bad)}")
    dims = dims[:12]
    nums = [c for c in cols["numeric"] if c != outcome]
    if date_col is None and cols["datetime"]:
        date_col = cols["datetime"][0]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        summary = _summary(data["__y"], kind)
        segments = _segments(data, dims, nums, kind, additive)
        numeric = _numeric(data, nums)
        period = _period(data, date_col, dims, kind, additive, period_a, period_b) if date_col else None
        model = _driver_model(data, dims, nums, kind, random_state)
        contrast = _contrast(data, dims, nums)
        anomalies = _anomalies(data, date_col, kind)
    findings = _findings(outcome, desc, kind, segments, numeric, period, model, contrast, anomalies)
    return to_jsonable({
        "available": True, "outcome": outcome, "outcome_kind": kind, "outcome_description": desc,
        "positive_class": positive, "additive": additive, "date_column": date_col, "dimensions": dims,
        "summary": summary, "segments": segments, "numeric_relationships": numeric, "period_change": period,
        "driver_model": model, "high_vs_low": contrast, "anomalies": anomalies, "findings": findings,
        "causal_note": CAUSAL_NOTE,
    })


def _summary(y: pd.Series, kind: str) -> Dict[str, Any]:
    return {"n": int(len(y)), "mean": float(y.mean()), "median": float(y.median()), "std": float(y.std()),
            "ci95": st.mean_ci(y), "min": float(y.min()), "max": float(y.max()), "total": float(y.sum()),
            "kind": kind}


def _segments(data, dims, nums, kind, additive) -> List[Dict[str, Any]]:
    out = []
    tests = []
    for d in dims:
        g = st.group_test(data["__y"], data[d], min_group=5)
        tests.append(g["kruskal_p"])
        grp = data.groupby(data[d].astype("string").fillna("(missing)"), observed=True)["__y"]
        table = []
        total = float(data["__y"].sum())
        for lvl, sub in grp:
            table.append({"level": str(lvl), "n": int(len(sub)), "mean": float(sub.mean()),
                          "median": float(sub.median()), "ci95": st.mean_ci(sub),
                          "share_of_total_pct": float(sub.sum() / total * 100) if additive and total else None,
                          "share_of_rows_pct": float(len(sub) / len(data) * 100)})
        table.sort(key=lambda r: -r["mean"])
        valid = [r for r in table if r["n"] >= 5]
        best_worst = None
        if len(valid) >= 2:
            hi, lo = valid[0], valid[-1]
            tg = st.two_group(data.loc[data[d].astype("string") == hi["level"], "__y"],
                              data.loc[data[d].astype("string") == lo["level"], "__y"])
            best_worst = {"best": hi["level"], "worst": lo["level"], "gap": hi["mean"] - lo["mean"],
                          "gap_pct": (hi["mean"] - lo["mean"]) / abs(lo["mean"]) * 100 if lo["mean"] else None, **tg,
                          "effect": st.effect_label_d(tg["cohens_d"])}
        out.append({"dimension": d, "levels": len(table), "table": table[:30], "kruskal_p": g["kruskal_p"],
                    "eta_squared": g["eta_squared"], "effect": st.effect_label_eta(g["eta_squared"]),
                    "best_vs_worst": best_worst})
    adj = st.benjamini_hochberg(tests)
    for s, q in zip(out, adj):
        s["p_adjusted"] = q
        s["significant"] = bool(q is not None and q < 0.05)
    out.sort(key=lambda s: -(s["eta_squared"] or 0))
    # Confounder / stratification check for the strongest dimensions
    for s in out[:3]:
        s["confounding_check"] = _stratify(data, s, dims, nums)
    return out


def _stratify(data, seg, dims, nums) -> Optional[Dict[str, Any]]:
    bw = seg.get("best_vs_worst")
    if not bw:
        return None
    d = seg["dimension"]
    candidates = []
    for other in dims:
        if other == d:
            continue
        cv = st.cramers_v(data[d], data[other])
        e = st.eta_squared(data["__y"], data[other])
        if cv["v"] and cv["v"] >= 0.2 and e and e >= 0.01:
            candidates.append((other, cv["v"], e, "categorical"))
    for nc in nums[:15]:
        e1 = st.eta_squared(data[nc], data[d])
        r = st.corr_test(data[nc], data["__y"], "spearman")["r"]
        if e1 and e1 >= 0.06 and r is not None and abs(r) >= 0.2:
            candidates.append((nc, math.sqrt(e1), abs(r), "numeric"))
    if not candidates:
        return {"confounders": [], "verdict": "No strong confounder found among the available columns.",
                "gap_raw": bw["gap"]}
    candidates.sort(key=lambda c: -(c[1] * c[2]))
    conf, assoc, _, ckind = candidates[0]
    strata = data[conf].astype("string") if ckind == "categorical" else pd.qcut(data[conf].rank(method="first"), 4,
                                                                                  labels=False).astype("string")
    sub = data[data[d].astype("string").isin([bw["best"], bw["worst"]])]
    gaps, weights = [], []
    for sv, part in sub.groupby(strata.loc[sub.index], observed=True):
        a = part.loc[part[d].astype("string") == bw["best"], "__y"]
        b = part.loc[part[d].astype("string") == bw["worst"], "__y"]
        if len(a) >= 3 and len(b) >= 3:
            gaps.append(a.mean() - b.mean())
            weights.append(len(a) + len(b))
    if not gaps:
        return {"confounders": [c[0] for c in candidates[:3]], "controlled_for": conf,
                "verdict": f"Could not compare '{bw['best']}' and '{bw['worst']}' within levels of '{conf}' "
                           "(too few rows per stratum).", "gap_raw": bw["gap"]}
    adj_gap = float(np.average(gaps, weights=weights))
    raw = bw["gap"]
    ratio = adj_gap / raw if raw else None
    if ratio is None:
        verdict = "n/a"
    elif ratio < 0:
        verdict = (f"The gap reverses within levels of '{conf}' (Simpson's paradox): the raw difference is driven by "
                   f"how '{conf}' is distributed across segments.")
    elif ratio < 0.5:
        verdict = f"The gap shrinks by {100 - ratio * 100:.0f}% after accounting for '{conf}', which explains much of it."
    elif ratio <= 1.2:
        verdict = f"The gap largely persists ({ratio * 100:.0f}% of the raw gap) within levels of '{conf}'."
    else:
        verdict = f"The gap becomes larger ({ratio * 100:.0f}% of the raw gap) within levels of '{conf}'."
    return {"confounders": [c[0] for c in candidates[:3]], "controlled_for": conf, "gap_raw": raw,
            "gap_adjusted": adj_gap, "adjusted_to_raw_ratio": ratio, "verdict": verdict,
            "method": f"Weighted average of within-stratum gaps across levels of '{conf}'"
                      + (" (quartiles)" if ckind == "numeric" else "") + "."}


def _rank_resid(a: pd.Series, controls: pd.DataFrame) -> np.ndarray:
    ra = a.rank().to_numpy(dtype=float)
    C = np.column_stack([np.ones(len(a))] + [controls[c].rank().to_numpy(dtype=float) for c in controls])
    beta, *_ = np.linalg.lstsq(C, ra, rcond=None)
    return ra - C @ beta


def _numeric(data, nums) -> List[Dict[str, Any]]:
    res = []
    for c in nums[:25]:
        t = st.corr_test(data[c], data["__y"], "spearman")
        if t["r"] is None:
            continue
        res.append({"feature": c, "spearman": t["r"], "p_value": t["p_value"], "ci95": t["ci95"], "n": t["n"],
                    "strength": st.strength_label(t["r"])})
    adj = st.benjamini_hochberg([r["p_value"] for r in res])
    for r, q in zip(res, adj):
        r["p_adjusted"] = q
    res.sort(key=lambda r: -abs(r["spearman"]))
    # Partial correlation for top relationships controlling for the other strongest numeric drivers
    top = [r["feature"] for r in res[:4]]
    for r in res[:4]:
        controls = [c for c in top if c != r["feature"]][:2]
        if not controls:
            continue
        sub = data[[r["feature"], "__y"] + controls].dropna()
        if len(sub) < 20:
            continue
        ea = _rank_resid(sub[r["feature"]], sub[controls])
        ey = _rank_resid(sub["__y"], sub[controls])
        pr = float(np.corrcoef(ea, ey)[0, 1]) if np.std(ea) > 0 and np.std(ey) > 0 else None
        r["partial_spearman"] = pr
        r["controls"] = controls
        if pr is not None:
            r["partial_note"] = ("persists" if abs(pr) >= 0.7 * abs(r["spearman"]) else
                                 "weakens" if abs(pr) >= 0.2 * abs(r["spearman"]) else "largely disappears") + \
                                f" after controlling for {', '.join(controls)}"
    return res[:12]


def _auto_periods(data, date_col) -> Optional[Tuple[Tuple[pd.Timestamp, pd.Timestamp], Tuple[pd.Timestamp, pd.Timestamp], str]]:
    t = pd.to_datetime(data[date_col])
    span = (t.max() - t.min()).days
    if span < 14:
        return None
    if span >= 90:
        last_month_start = t.max().to_period("M").start_time
        # Use the last complete month unless data ends on the final day of the month
        month_end = (last_month_start + pd.offsets.MonthEnd(0)).normalize()
        if t.max().normalize() < month_end - pd.Timedelta(days=1):
            b_start = (last_month_start - pd.offsets.MonthBegin(1))
        else:
            b_start = last_month_start
        b_end = b_start + pd.offsets.MonthEnd(0) + pd.Timedelta(hours=23, minutes=59, seconds=59)
        a_start = b_start - pd.offsets.MonthBegin(1)
        a_end = b_start - pd.Timedelta(seconds=1)
        return (a_start, a_end), (b_start, b_end), "last complete month vs the month before"
    mid = t.min() + (t.max() - t.min()) / 2
    return (t.min(), mid), (mid + pd.Timedelta(seconds=1), t.max()), "second half vs first half of the date range"


def _period(data, date_col, dims, kind, additive, period_a, period_b) -> Optional[Dict[str, Any]]:
    if date_col not in data.columns or not pd.api.types.is_datetime64_any_dtype(data[date_col]):
        return None
    if period_a and period_b:
        try:
            A = (pd.Timestamp(period_a[0]), pd.Timestamp(period_a[1]) + pd.Timedelta(hours=23, minutes=59, seconds=59))
            B = (pd.Timestamp(period_b[0]), pd.Timestamp(period_b[1]) + pd.Timedelta(hours=23, minutes=59, seconds=59))
        except Exception:
            raise ValidationFailed("Periods must be [start, end] ISO dates")
        how = "user-defined periods"
    else:
        auto = _auto_periods(data, date_col)
        if not auto:
            return None
        A, B, how = auto
    t = pd.to_datetime(data[date_col])
    a = data[(t >= A[0]) & (t <= A[1])]
    b = data[(t >= B[0]) & (t <= B[1])]
    if len(a) < 5 or len(b) < 5:
        return {"available": False, "reason": "Too few rows in one of the periods (need ≥ 5 each).",
                "period_a": [A[0].isoformat(), A[1].isoformat()], "period_b": [B[0].isoformat(), B[1].isoformat()]}
    stat = "total" if additive else ("rate" if kind == "rate" else "mean")
    va = float(a["__y"].sum() if additive else a["__y"].mean())
    vb = float(b["__y"].sum() if additive else b["__y"].mean())
    tg = st.two_group(b["__y"], a["__y"])
    res: Dict[str, Any] = {
        "available": True, "how": how, "statistic": stat,
        "period_a": {"start": A[0].isoformat(), "end": A[1].isoformat(), "rows": int(len(a)), "value": va,
                     "mean": float(a["__y"].mean())},
        "period_b": {"start": B[0].isoformat(), "end": B[1].isoformat(), "rows": int(len(b)), "value": vb,
                     "mean": float(b["__y"].mean())},
        "change": vb - va, "change_pct": (vb - va) / abs(va) * 100 if va else None,
        "row_count_change_pct": (len(b) - len(a)) / len(a) * 100,
        "test": {"mann_whitney_p": tg["mann_whitney_p"], "mean_diff_ci95": tg["diff_ci95"], "cohens_d": tg["cohens_d"]},
        "contributions": [],
    }
    for d in dims[:8]:
        ga = a.groupby(a[d].astype("string").fillna("(missing)"), observed=True)["__y"]
        gb = b.groupby(b[d].astype("string").fillna("(missing)"), observed=True)["__y"]
        levels = sorted(set(ga.groups) | set(gb.groups), key=str)
        rows, mix_total, rate_total = [], 0.0, 0.0
        na, nb = len(a), len(b)
        for lv in levels:
            sa = ga.get_group(lv) if lv in ga.groups else pd.Series(dtype=float)
            sb = gb.get_group(lv) if lv in gb.groups else pd.Series(dtype=float)
            if additive:
                contrib = float(sb.sum() - sa.sum())
                mix = rate = None
            else:
                wa, wb = len(sa) / na, len(sb) / nb
                ma = float(sa.mean()) if len(sa) else None
                mb = float(sb.mean()) if len(sb) else None
                contrib = (wb * (mb or 0)) - (wa * (ma or 0))
                if ma is not None and mb is not None:
                    mix = (wb - wa) * ma
                    rate = wb * (mb - ma)
                    mix_total += mix
                    rate_total += rate
                else:
                    mix = rate = None
            rows.append({"level": str(lv), "rows_a": int(len(sa)), "rows_b": int(len(sb)),
                         "value_a": float(sa.sum() if additive else (sa.mean() if len(sa) else np.nan)),
                         "value_b": float(sb.sum() if additive else (sb.mean() if len(sb) else np.nan)),
                         "contribution": contrib, "mix_effect": mix, "rate_effect": rate})
        rows.sort(key=lambda r: -abs(r["contribution"]))
        total_change = vb - va
        for r in rows:
            r["share_of_change_pct"] = r["contribution"] / total_change * 100 if total_change else None
        concentration = abs(rows[0]["contribution"]) / sum(abs(r["contribution"]) for r in rows) if rows and any(r["contribution"] for r in rows) else 0
        res["contributions"].append({"dimension": d, "rows": rows[:12], "concentration": concentration,
                                     "mix_total": None if additive else mix_total,
                                     "rate_total": None if additive else rate_total})
    res["contributions"].sort(key=lambda c: -c["concentration"])
    res["method"] = ("Change in the total is split exactly into each segment's change in total."
                     if additive else
                     "Change in the mean is split exactly into segment contributions; for segments present in both "
                     "periods, each contribution is decomposed into a mix effect (change in the segment's share of rows) "
                     "and a rate effect (change in the segment's own mean).")
    return res


def _driver_model(data, dims, nums, kind, rs) -> Dict[str, Any]:
    from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor
    from sklearn.inspection import permutation_importance
    from sklearn.model_selection import KFold, StratifiedKFold, cross_val_score, train_test_split
    feats = dims[:12] + nums[:20]
    if not feats:
        return {"available": False, "reason": "No candidate driver columns."}
    sub = data[feats + ["__y"]]
    if len(sub) > 50_000:
        sub = sub.sample(50_000, random_state=rs)
    X = sub[feats].copy()
    for d in dims[:12]:
        X[d] = X[d].astype("category")
    y = sub["__y"].to_numpy()
    is_cls = kind == "rate"
    if is_cls and (np.bincount(y.astype(int)).min() < 5):
        return {"available": False, "reason": "Too few positive or negative outcomes for a driver model."}
    est = (HistGradientBoostingClassifier(categorical_features="from_dtype", max_iter=150, random_state=rs) if is_cls
           else HistGradientBoostingRegressor(categorical_features="from_dtype", max_iter=150, random_state=rs))
    cv = (StratifiedKFold(3, shuffle=True, random_state=rs) if is_cls else KFold(3, shuffle=True, random_state=rs))
    scoring = "roc_auc" if is_cls else "r2"
    try:
        scores = cross_val_score(est, X, y, cv=cv, scoring=scoring)
    except Exception as e:
        return {"available": False, "reason": f"Driver model failed ({type(e).__name__})."}
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.3, random_state=rs, stratify=y if is_cls else None)
    est.fit(Xtr, ytr)
    pi = permutation_importance(est, Xte, yte, scoring=scoring, n_repeats=5, random_state=rs)
    items = sorted([{"feature": f, "importance": float(pi.importances_mean[i]), "std": float(pi.importances_std[i])}
                    for i, f in enumerate(feats)], key=lambda d: -d["importance"])
    score = float(np.mean(scores))
    explained = ("The available columns explain little of the outcome; important drivers are probably missing."
                 if (not is_cls and score < 0.1) or (is_cls and score < 0.6) else
                 "The available columns explain a moderate share of the outcome." if (not is_cls and score < 0.4) or (is_cls and score < 0.75)
                 else "The available columns explain a substantial share of the outcome.")
    return {"available": True, "metric": "ROC-AUC" if is_cls else "R²", "cv_score": score, "cv_std": float(np.std(scores)),
            "interpretation": explained, "importance": items[:12],
            "notes": ["Gradient-boosting model, 3-fold cross-validation; importance by permutation on a 30% hold-out.",
                      "High importance means the model relies on the column to predict the outcome — not that the column causes it."]}


def _contrast(data, dims, nums) -> Dict[str, Any]:
    q1, q3 = data["__y"].quantile([0.25, 0.75])
    if q1 == q3:
        return {"available": False, "reason": "The outcome does not vary enough to compare top and bottom quartiles."}
    hi, lo = data[data["__y"] >= q3], data[data["__y"] <= q1]
    numeric = []
    for c in nums[:20]:
        a, b = hi[c].dropna().astype(float), lo[c].dropna().astype(float)
        if len(a) < 5 or len(b) < 5:
            continue
        pooled = math.sqrt((a.var() + b.var()) / 2) or None
        smd = (a.mean() - b.mean()) / pooled if pooled else None
        numeric.append({"feature": c, "mean_high": float(a.mean()), "mean_low": float(b.mean()), "smd": smd,
                        "effect": st.effect_label_d(smd)})
    numeric.sort(key=lambda r: -abs(r["smd"] or 0))
    cats = []
    for d in dims[:10]:
        sh = hi[d].astype("string").value_counts(normalize=True)
        sl = lo[d].astype("string").value_counts(normalize=True)
        for lv in set(sh.index) | set(sl.index):
            a, b = float(sh.get(lv, 0)), float(sl.get(lv, 0))
            if max(a, b) >= 0.05:
                cats.append({"dimension": d, "level": str(lv), "share_high": a, "share_low": b,
                             "lift": (a / b) if b > 0 else None, "difference_pp": (a - b) * 100})
    cats.sort(key=lambda r: -abs(r["difference_pp"]))
    return {"available": True, "threshold_high": float(q3), "threshold_low": float(q1), "rows_high": int(len(hi)),
            "rows_low": int(len(lo)), "numeric": numeric[:10], "categorical": cats[:12],
            "notes": ["Compares rows in the top quartile of the outcome with rows in the bottom quartile.",
                      "Standardised mean difference (SMD): |SMD| ≥ 0.5 is a medium, ≥ 0.8 a large difference."]}


def _anomalies(data, date_col, kind) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    y = data["__y"]
    if kind == "mean":
        med = y.median()
        mad = (y - med).abs().median() * 1.4826
        if mad > 0:
            z = (y - med) / mad
            flagged = z[z.abs() > 3.5].sort_values(key=lambda s: -s.abs())
            out["records"] = {"count": int(len(flagged)), "pct": float(len(flagged) / len(y) * 100),
                              "top": [{"row": to_jsonable(i), "value": float(y[i]), "robust_z": float(z[i])}
                                      for i in flagged.index[:10]],
                              "method": "robust z-score (median/MAD) above 3.5"}
    if date_col and date_col in data.columns and pd.api.types.is_datetime64_any_dtype(data[date_col]):
        freq, info = st.infer_frequency(data[date_col])
        agg = "mean" if kind == "rate" else "mean"
        s = st.aggregate_series(data, date_col, "__y", freq, agg).dropna()
        if len(s) >= 8:
            med = s.median()
            mad = (s - med).abs().median() * 1.4826
            if mad > 0:
                z = (s - med) / mad
                pts = z[z.abs() > 3]
                out["periods"] = {"frequency": freq, "count": int(len(pts)),
                                  "items": [{"period": t.isoformat(), "value": float(s[t]), "robust_z": float(z[t])}
                                            for t in pts.index[:10]],
                                  "method": f"robust z-score of the {agg} per {st.FREQ_LABELS.get(freq, freq)} above 3"}
    return out


def _findings(outcome, desc, kind, segments, numeric, period, model, contrast, anomalies) -> List[Dict[str, Any]]:
    F: List[Dict[str, Any]] = []
    metric = f"mean {outcome}" if kind == "mean" else desc
    if period and period.get("available") and period.get("change_pct") is not None:
        p = period
        top = p["contributions"][0] if p["contributions"] else None
        lead = top["rows"][0] if top and top["rows"] else None
        sig = p["test"]["mann_whitney_p"]
        real = sig is not None and sig < 0.05
        F.append({
            "label": "observation",
            "title": f"{'Increase' if p['change'] > 0 else 'Decrease'} of {abs(p['change_pct']):.1f}% ({p['how']})"
                     + ("" if real else " — not distinguishable from random variation"),
            "outcome": outcome, "metric": f"{p['statistic']} of {outcome}",
            "comparison": f"{p['period_b']['start'][:10]}–{p['period_b']['end'][:10]} vs {p['period_a']['start'][:10]}–{p['period_a']['end'][:10]}",
            "evidence": f"{_fmt(p['period_a']['value'])} → {_fmt(p['period_b']['value'])}; rows {p['period_a']['rows']} → {p['period_b']['rows']}"
                        + (f"; Mann–Whitney p = {sig:.3g}" if sig is not None else ""),
            "effect_size": p["test"].get("cohens_d"), "uncertainty": p["test"].get("mean_diff_ci95"),
            "confounders": [], "alternatives": ["A change in the number of records (volume) rather than per-record values."
                                                if abs(p["row_count_change_pct"]) > 10 else "Random period-to-period variation.",
                                                "Incomplete data for one of the periods."],
            "validation": ["Check that both periods are complete.", "Compare with the same period last year for seasonality."],
            "strength": 0.9 if sig is not None and sig < 0.05 else 0.5,
        })
        if lead and top["concentration"] >= 0.4:
            F.append({
                "label": "potential driver" if real else "observation",
                "title": f"'{top['dimension']} = {lead['level']}' accounts for most of the change"
                         + ("" if real else " (the change itself is not statistically significant)"),
                "outcome": outcome, "metric": f"contribution to the change in {p['statistic']}",
                "comparison": "segment contribution between the two periods",
                "evidence": f"Contribution {_fmt(lead['contribution'])} "
                            + (f"({lead['share_of_change_pct']:.0f}% of the total change)" if lead.get("share_of_change_pct") is not None else "")
                            + (f"; mix effect {_fmt(lead['mix_effect'])}, rate effect {_fmt(lead['rate_effect'])}" if lead.get("mix_effect") is not None else ""),
                "effect_size": None, "uncertainty": None, "confounders": [],
                "alternatives": ["The segment's share of rows changed (mix) rather than its performance (rate)."
                                 if lead.get("mix_effect") is not None and abs(lead["mix_effect"]) > abs(lead.get("rate_effect") or 0) else
                                 "Other dimensions overlapping with this segment may be the real driver."],
                "validation": [f"Drill into '{lead['level']}' rows for both periods.", "Repeat the decomposition with other dimensions."],
                "strength": (0.8 * top["concentration"] + 0.2) * (1 if real else 0.4),
            })
    for s in segments[:3]:
        if not s.get("significant") or (s["eta_squared"] or 0) < 0.01 or not s.get("best_vs_worst"):
            continue
        bw = s["best_vs_worst"]
        cc = s.get("confounding_check") or {}
        ratio = cc.get("adjusted_to_raw_ratio")
        label = "potential driver" if ratio is None or ratio >= 0.5 else "diagnostic hypothesis"
        F.append({
            "label": label, "title": f"{metric.capitalize()} differs by '{s['dimension']}'",
            "outcome": outcome, "metric": metric,
            "comparison": f"'{bw['best']}' vs '{bw['worst']}' (highest vs lowest of {s['levels']} groups)",
            "evidence": f"η² = {s['eta_squared']:.3f} ({s['effect']}), Kruskal–Wallis adjusted p = {s['p_adjusted']:.3g}; "
                        f"gap {_fmt(bw['gap'])}" + (f" ({bw['gap_pct']:+.0f}%)" if bw.get('gap_pct') is not None else ""),
            "effect_size": bw.get("cohens_d"), "uncertainty": bw.get("diff_ci95"),
            "confounders": cc.get("confounders", []),
            "alternatives": [cc["verdict"]] if cc.get("verdict") else ["Differences in segment composition."],
            "validation": [f"Compare '{bw['best']}' and '{bw['worst']}' within comparable sub-groups.",
                           "Check whether the gap is stable over time."],
            "strength": min(1.0, 0.3 + (s["eta_squared"] or 0) * 3) * (1 if label == "potential driver" else 0.6),
        })
    for r in numeric[:2]:
        if r.get("p_adjusted") is None or r["p_adjusted"] >= 0.05 or abs(r["spearman"]) < 0.2:
            continue
        persists = r.get("partial_note", "")
        label = "potential driver" if "disappears" not in persists else "diagnostic hypothesis"
        F.append({
            "label": label, "title": f"'{r['feature']}' moves with {outcome}",
            "outcome": outcome, "metric": "Spearman rank correlation", "comparison": f"{r['feature']} vs {outcome}",
            "evidence": f"ρ = {r['spearman']:.2f}" + (f" (95% CI {r['ci95'][0]:.2f} to {r['ci95'][1]:.2f})" if r.get("ci95") else "")
                        + f", n = {r['n']}, adjusted p = {r['p_adjusted']:.3g}" + (f"; {persists}" if persists else ""),
            "effect_size": r["spearman"], "uncertainty": r.get("ci95"), "confounders": r.get("controls", []),
            "alternatives": ["Both may be driven by a third factor.", "The relationship may be definitional."],
            "validation": ["Inspect the scatter plot for non-linearity and outliers."],
            "strength": min(1.0, abs(r["spearman"])),
        })
    if model.get("available") and model["importance"]:
        top = [i for i in model["importance"] if i["importance"] > 0][:3]
        if top:
            F.append({
                "label": "diagnostic hypothesis", "title": "Most predictive columns in a multivariate model",
                "outcome": outcome, "metric": f"permutation importance (model {model['metric']} {model['cv_score']:.2f})",
                "comparison": "all candidate columns jointly",
                "evidence": ", ".join(f"{i['feature']} ({i['importance']:.3f})" for i in top) + f". {model['interpretation']}",
                "effect_size": None, "uncertainty": model.get("cv_std"), "confounders": [],
                "alternatives": ["Predictive importance can reflect proxies for unobserved causes."],
                "validation": ["Test the top columns individually and with domain experts."],
                "strength": 0.5,
            })
    rec = anomalies.get("records")
    if rec and rec["count"]:
        F.append({
            "label": "observation", "title": f"{rec['count']} unusual record(s) in {outcome}",
            "outcome": outcome, "metric": rec["method"], "comparison": "each record vs the median",
            "evidence": f"{rec['pct']:.1f}% of records", "effect_size": None, "uncertainty": None, "confounders": [],
            "alternatives": ["Data-entry errors.", "Genuine extreme events."],
            "validation": ["Inspect the listed rows."], "strength": 0.35,
        })
    F.sort(key=lambda f: -f["strength"])
    return F


# ── v4 compatibility ──────────────────────────────────────────────────────────

def run_rca(df: pd.DataFrame, eda: Optional[Dict] = None, target_col: Optional[str] = None) -> Dict[str, Any]:
    schema = infer_schema(df)
    smap = {c["name"]: c for c in schema}
    cols = analysis_columns(df, schema)
    if not cols["numeric"]:
        return {"available": False, "reason": "No numeric columns found for analysis."}
    if target_col is None or target_col not in df.columns:
        target_col = rank_measures(cols["numeric"], smap)[0]
    r = run_diagnostics(df, target_col, schema)
    s = r["summary"]
    return to_jsonable({
        "target": target_col, "available": True,
        "summary_stats": {"mean": s["mean"], "median": s["median"], "std": s["std"], "min": s["min"], "max": s["max"],
                          "total": s["total"]},
        "categorical_drivers": [{"column": g["dimension"], "eta_squared": g["eta_squared"],
                                 "rows": [{"category": t["level"], "mean": t["mean"], "count": t["n"],
                                           "share_pct": t["share_of_total_pct"]} for t in g["table"][:10]],
                                 "insight": (g.get("confounding_check") or {}).get("verdict", "")} for g in r["segments"]],
        "correlation_drivers": [{"driver": n["feature"], "correlation": n["spearman"], "strength": n["strength"],
                                 "direction": "positive" if n["spearman"] > 0 else "negative"} for n in r["numeric_relationships"]],
        "segment_gaps": [g["best_vs_worst"] for g in r["segments"] if g.get("best_vs_worst")],
        "time_comparison": r["period_change"], "outlier_analysis": r["anomalies"].get("records"),
        "findings": [{"rank": i + 1, "type": f["label"], "title": f["title"], "text": f["evidence"]}
                     for i, f in enumerate(r["findings"])],
        "causal_note": CAUSAL_NOTE,
    })
