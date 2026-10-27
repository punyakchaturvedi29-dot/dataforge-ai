"""
Statistical building blocks shared by the analysis engines.

Everything returns plain floats/dicts and degrades gracefully (``None``) when a
test is not applicable (too few observations, zero variance, ...).
"""
from __future__ import annotations

import math
import warnings
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from scipy import stats as sps


def _f(v) -> Optional[float]:
    try:
        v = float(v)
    except (TypeError, ValueError):
        return None
    return None if math.isnan(v) or math.isinf(v) else v


# ── Correlation ──────────────────────────────────────────────────────────────

def corr_test(x: pd.Series, y: pd.Series, method: str = "pearson") -> Dict:
    d = pd.concat([x, y], axis=1).dropna()
    n = len(d)
    if n < 5 or d.iloc[:, 0].nunique() < 2 or d.iloc[:, 1].nunique() < 2:
        return {"r": None, "p_value": None, "n": n, "ci95": None}
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        if method == "spearman":
            r, p = sps.spearmanr(d.iloc[:, 0], d.iloc[:, 1])
        else:
            r, p = sps.pearsonr(d.iloc[:, 0].astype(float), d.iloc[:, 1].astype(float))
    return {"r": _f(r), "p_value": _f(p), "n": n, "ci95": fisher_ci(_f(r), n)}


def fisher_ci(r: Optional[float], n: int, conf: float = 0.95) -> Optional[List[float]]:
    if r is None or n < 4 or abs(r) >= 1:
        return None
    z = math.atanh(r)
    se = 1 / math.sqrt(n - 3)
    zc = sps.norm.ppf(1 - (1 - conf) / 2)
    return [math.tanh(z - zc * se), math.tanh(z + zc * se)]


def strength_label(r: Optional[float]) -> str:
    if r is None:
        return "n/a"
    a = abs(r)
    return ("very strong" if a >= 0.8 else "strong" if a >= 0.6 else "moderate" if a >= 0.4
            else "weak" if a >= 0.2 else "negligible")


def cramers_v(a: pd.Series, b: pd.Series) -> Dict:
    d = pd.concat([a, b], axis=1).dropna()
    if len(d) < 10:
        return {"v": None, "p_value": None, "n": len(d)}
    tab = pd.crosstab(d.iloc[:, 0], d.iloc[:, 1])
    if tab.shape[0] < 2 or tab.shape[1] < 2:
        return {"v": None, "p_value": None, "n": len(d)}
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        chi2, p, dof, exp = sps.chi2_contingency(tab, correction=False)
    n = tab.to_numpy().sum()
    r, k = tab.shape
    # Bias-corrected Cramér's V (Bergsma 2013)
    phi2 = chi2 / n
    phi2c = max(0.0, phi2 - (k - 1) * (r - 1) / (n - 1))
    rc = r - (r - 1) ** 2 / (n - 1)
    kc = k - (k - 1) ** 2 / (n - 1)
    denom = min(kc - 1, rc - 1)
    v = math.sqrt(phi2c / denom) if denom > 0 else None
    low_expected = float((exp < 5).mean())
    return {"v": _f(v), "p_value": _f(p), "n": int(n), "chi2": _f(chi2), "dof": int(dof),
            "low_expected_share": low_expected}


# ── Group comparisons ─────────────────────────────────────────────────────────

def eta_squared(values: pd.Series, groups: pd.Series) -> Optional[float]:
    d = pd.concat([values.astype(float), groups], axis=1).dropna()
    if len(d) < 3:
        return None
    y = d.iloc[:, 0]
    g = d.iloc[:, 1]
    grand = y.mean()
    ss_t = ((y - grand) ** 2).sum()
    if ss_t <= 0:
        return None
    ss_b = sum(len(s) * (s.mean() - grand) ** 2 for _, s in y.groupby(g, observed=True))
    return _f(ss_b / ss_t)


def group_test(values: pd.Series, groups: pd.Series, min_group: int = 3) -> Dict:
    """Kruskal–Wallis (robust default) + one-way ANOVA + eta² across groups."""
    d = pd.concat([values.astype(float), groups.astype(str)], axis=1).dropna()
    samples = [s.to_numpy() for _, s in d.iloc[:, 0].groupby(d.iloc[:, 1]) if len(s) >= min_group]
    res = {"n": len(d), "groups_tested": len(samples), "kruskal_p": None, "anova_p": None,
           "eta_squared": eta_squared(d.iloc[:, 0], d.iloc[:, 1]), "epsilon_squared": None}
    if len(samples) < 2:
        return res
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        try:
            h, p = sps.kruskal(*samples)
            res["kruskal_p"] = _f(p)
            n = sum(len(s) for s in samples)
            res["epsilon_squared"] = _f(h / ((n ** 2 - 1) / (n + 1))) if n > 1 else None
        except ValueError:
            pass
        try:
            res["anova_p"] = _f(sps.f_oneway(*samples).pvalue)
        except Exception:
            pass
    return res


def two_group(a: Sequence[float], b: Sequence[float]) -> Dict:
    a = np.asarray(pd.Series(a).dropna(), dtype=float)
    b = np.asarray(pd.Series(b).dropna(), dtype=float)
    out = {"n_a": int(len(a)), "n_b": int(len(b)), "mean_a": _f(a.mean()) if len(a) else None,
           "mean_b": _f(b.mean()) if len(b) else None, "mann_whitney_p": None, "welch_p": None,
           "cohens_d": None, "cliffs_delta": None, "diff_ci95": None}
    if len(a) < 3 or len(b) < 3:
        return out
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        try:
            out["mann_whitney_p"] = _f(sps.mannwhitneyu(a, b, alternative="two-sided").pvalue)
        except ValueError:
            pass
        t = sps.ttest_ind(a, b, equal_var=False)
        out["welch_p"] = _f(t.pvalue)
    sa, sb = a.var(ddof=1), b.var(ddof=1)
    pooled = math.sqrt(((len(a) - 1) * sa + (len(b) - 1) * sb) / (len(a) + len(b) - 2)) if len(a) + len(b) > 2 else 0
    out["cohens_d"] = _f((a.mean() - b.mean()) / pooled) if pooled > 0 else None
    out["cliffs_delta"] = cliffs_delta(a, b)
    se = math.sqrt(sa / len(a) + sb / len(b))
    if se > 0:
        dfw = (sa / len(a) + sb / len(b)) ** 2 / ((sa / len(a)) ** 2 / (len(a) - 1) + (sb / len(b)) ** 2 / (len(b) - 1))
        tc = sps.t.ppf(0.975, dfw)
        diff = a.mean() - b.mean()
        out["diff_ci95"] = [_f(diff - tc * se), _f(diff + tc * se)]
    return out


def cliffs_delta(a: np.ndarray, b: np.ndarray, max_n: int = 3000) -> Optional[float]:
    if len(a) == 0 or len(b) == 0:
        return None
    rng = np.random.default_rng(0)
    if len(a) > max_n:
        a = rng.choice(a, max_n, replace=False)
    if len(b) > max_n:
        b = rng.choice(b, max_n, replace=False)
    # Rank-based O((n+m) log(n+m)) computation via Mann–Whitney U
    u = sps.mannwhitneyu(a, b, alternative="two-sided").statistic
    return _f(2 * u / (len(a) * len(b)) - 1)


def effect_label_d(d: Optional[float]) -> str:
    if d is None:
        return "n/a"
    a = abs(d)
    return "large" if a >= 0.8 else "medium" if a >= 0.5 else "small" if a >= 0.2 else "negligible"


def effect_label_eta(e: Optional[float]) -> str:
    if e is None:
        return "n/a"
    return "large" if e >= 0.14 else "medium" if e >= 0.06 else "small" if e >= 0.01 else "negligible"


def mean_ci(x: Sequence[float], conf: float = 0.95) -> Optional[List[float]]:
    x = np.asarray(pd.Series(x).dropna(), dtype=float)
    if len(x) < 2:
        return None
    m, se = x.mean(), x.std(ddof=1) / math.sqrt(len(x))
    tc = sps.t.ppf(1 - (1 - conf) / 2, len(x) - 1)
    return [_f(m - tc * se), _f(m + tc * se)]


def benjamini_hochberg(pvals: List[Optional[float]]) -> List[Optional[float]]:
    idx = [i for i, p in enumerate(pvals) if p is not None]
    if not idx:
        return [None] * len(pvals)
    ps = np.array([pvals[i] for i in idx])
    m = len(ps)
    order = np.argsort(ps)
    ranked = ps[order] * m / (np.arange(m) + 1)
    adj = np.minimum.accumulate(ranked[::-1])[::-1]
    adj = np.clip(adj, 0, 1)
    out: List[Optional[float]] = [None] * len(pvals)
    for k, o in enumerate(order):
        out[idx[o]] = float(adj[k])
    return out


# ── Distribution shape ───────────────────────────────────────────────────────

def normality(x: pd.Series) -> Dict:
    v = x.dropna().astype(float)
    if len(v) > 5000:
        v = v.sample(5000, random_state=0)
    if len(v) < 20 or v.nunique() < 3:
        return {"test": None, "p_value": None}
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        try:
            p = sps.normaltest(v).pvalue
        except Exception:
            return {"test": None, "p_value": None}
    return {"test": "D'Agostino-Pearson K²", "p_value": _f(p)}


def shape_label(skew: Optional[float], kurt: Optional[float]) -> str:
    if skew is None:
        return "unknown"
    parts = []
    if abs(skew) < 0.5:
        parts.append("approximately symmetric")
    elif abs(skew) < 1:
        parts.append("moderately " + ("right" if skew > 0 else "left") + "-skewed")
    else:
        parts.append("strongly " + ("right" if skew > 0 else "left") + "-skewed")
    if kurt is not None and kurt > 3:
        parts.append("heavy-tailed")
    elif kurt is not None and kurt < -1:
        parts.append("light-tailed / flat")
    return ", ".join(parts)


def histogram(x: pd.Series, max_bins: int = 40, min_bins: int = 8) -> Dict:
    v = x.dropna().astype(float).to_numpy()
    if len(v) == 0:
        return {"edges": [], "counts": []}
    if np.unique(v).size <= max_bins and np.allclose(v, np.round(v)):
        u, c = np.unique(v, return_counts=True)
        if len(u) <= 30:
            return {"discrete": True, "values": u.tolist(), "counts": c.tolist()}
    q75, q25 = np.percentile(v, [75, 25])
    iqr = q75 - q25
    if iqr > 0:
        width = 2 * iqr / (len(v) ** (1 / 3))
        bins = int(np.ceil((v.max() - v.min()) / width)) if width > 0 else min_bins
    else:
        bins = min_bins
    bins = int(min(max(bins, min_bins), max_bins))
    counts, edges = np.histogram(v, bins=bins)
    return {"discrete": False, "edges": edges.tolist(), "counts": counts.tolist()}


def box_stats(x: pd.Series, max_outliers: int = 200) -> Dict:
    v = x.dropna().astype(float)
    if len(v) == 0:
        return {}
    q1, med, q3 = v.quantile([0.25, 0.5, 0.75]).tolist()
    iqr = q3 - q1
    lo_f, hi_f = q1 - 1.5 * iqr, q3 + 1.5 * iqr
    inside = v[(v >= lo_f) & (v <= hi_f)]
    out = v[(v < lo_f) | (v > hi_f)]
    sample = out.sort_values()
    if len(sample) > max_outliers:
        sample = sample.iloc[np.linspace(0, len(sample) - 1, max_outliers).astype(int)]
    return {"q1": _f(q1), "median": _f(med), "q3": _f(q3),
            "whisker_low": _f(inside.min() if len(inside) else q1),
            "whisker_high": _f(inside.max() if len(inside) else q3),
            "n_outliers": int(len(out)), "outliers": [_f(o) for o in sample.tolist()], "n": int(len(v)),
            "mean": _f(v.mean())}


def outlier_summary(x: pd.Series) -> Dict:
    v = x.dropna().astype(float)
    if len(v) < 5:
        return {"iqr_mild": 0, "iqr_extreme": 0, "mad": 0}
    q1, q3 = v.quantile(0.25), v.quantile(0.75)
    iqr = q3 - q1
    med = v.median()
    mad = (v - med).abs().median() * 1.4826
    robust_z = (v - med).abs() / mad if mad > 0 else pd.Series(0, index=v.index)
    return {
        "iqr_mild": int(((v < q1 - 1.5 * iqr) | (v > q3 + 1.5 * iqr)).sum()) if iqr > 0 else 0,
        "iqr_extreme": int(((v < q1 - 3 * iqr) | (v > q3 + 3 * iqr)).sum()) if iqr > 0 else 0,
        "mad": int((robust_z > 3.5).sum()),
        "fences_mild": [_f(q1 - 1.5 * iqr), _f(q3 + 1.5 * iqr)],
        "fences_extreme": [_f(q1 - 3 * iqr), _f(q3 + 3 * iqr)],
        "method_note": "IQR fences (Tukey) and robust z-score > 3.5 using the median absolute deviation.",
    }


# ── Time handling ─────────────────────────────────────────────────────────────

FREQ_LABELS = {"D": "day", "W": "week", "MS": "month", "QS": "quarter", "YS": "year", "h": "hour"}


def infer_frequency(dates: pd.Series) -> Tuple[str, Dict]:
    """Choose an analysis frequency from the spacing of timestamps."""
    d = pd.Series(pd.to_datetime(dates.dropna().unique())).sort_values()
    info: Dict = {"n_unique_timestamps": int(len(d))}
    if len(d) < 2:
        return "D", info
    diffs = d.diff().dropna().dt.total_seconds() / 86400
    med = float(diffs.median())
    span = float((d.iloc[-1] - d.iloc[0]).total_seconds() / 86400)
    info.update({"median_gap_days": med, "span_days": span})
    if med < 0.9:
        native = "h"
    elif med < 3:
        native = "D"
    elif med < 10:
        native = "W"
    elif med < 45:
        native = "MS"
    elif med < 140:
        native = "QS"
    else:
        native = "YS"
    info["native_frequency"] = native
    # For charts we aggregate irregular/daily data so there are at most ~150 points.
    chart = native
    if native in ("h", "D") and span > 150:
        chart = "W" if span <= 3 * 365 else "MS"
    if native == "W" and span > 3 * 365:
        chart = "MS"
    info["chart_frequency"] = chart
    return chart, info


def aggregate_series(df: pd.DataFrame, date_col: str, value_col: Optional[str], freq: str, agg: str = "sum") -> pd.Series:
    d = df[[date_col] + ([value_col] if value_col else [])].dropna(subset=[date_col]).copy()
    d[date_col] = pd.to_datetime(d[date_col])
    d = d.set_index(date_col).sort_index()
    if value_col is None or agg == "count":
        s = d.iloc[:, 0] if value_col else pd.Series(1, index=d.index)
        return s.resample(freq).count()
    s = d[value_col].astype(float)
    r = s.resample(freq)
    return {"sum": r.sum, "mean": r.mean, "median": r.median, "min": r.min, "max": r.max, "last": r.last}[agg]()


def default_agg(col: str, semantic: Optional[str]) -> str:
    """Sum for additive measures (amounts, quantities); mean for intensive ones (prices, rates, ages, scores)."""
    import re as _re
    toks = set(t for t in _re.split(r"[^a-z0-9]+", str(col).lower()) if t)
    if toks & {"price", "rate", "avg", "average", "mean", "pct", "percent", "ratio", "score", "age", "margin",
               "per", "unit", "temperature", "temp", "rating", "days", "duration"}:
        return "mean"
    if semantic in ("revenue", "quantity"):
        return "sum"
    return "mean"
