"""
Data profiling engine.

Produces verifiable descriptive statistics for every column plus dataset-level
quality metrics. All numbers are computed from the data; the quality score is a
documented heuristic whose components are returned alongside it.
"""
from __future__ import annotations

import math
import warnings
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from core.jsonsafe import to_jsonable

from . import stats as st
from .schema import analysis_columns, infer_schema, is_text_dtype

VALIDITY_ISSUES = {"numeric_as_text", "dates_as_text", "null_tokens", "inconsistent_labels", "implausible_dates",
                   "negative_values", "number_with_unit"}


def _f(v):
    return to_jsonable(v)


def profile_numeric(s: pd.Series) -> Dict[str, Any]:
    v = s.dropna().astype(float)
    if len(v) == 0:
        return {}
    q = v.quantile([0.01, 0.05, 0.10, 0.25, 0.5, 0.75, 0.90, 0.95, 0.99])
    mode = v.mode()
    skew = _f(v.skew()) if len(v) > 2 else None
    kurt = _f(v.kurtosis()) if len(v) > 3 else None
    mean = float(v.mean())
    std = float(v.std(ddof=1)) if len(v) > 1 else 0.0
    return {
        "mean": _f(mean), "median": _f(q[0.5]),
        "mode": _f(mode.iloc[0]) if len(mode) and len(mode) < len(v) else None,
        "mode_count": int((v == mode.iloc[0]).sum()) if len(mode) else 0,
        "min": _f(v.min()), "max": _f(v.max()), "range": _f(v.max() - v.min()),
        "variance": _f(v.var(ddof=1)) if len(v) > 1 else None, "std": _f(std),
        "sem": _f(std / math.sqrt(len(v))) if len(v) > 1 else None,
        "cv_pct": _f(std / abs(mean) * 100) if mean != 0 else None,
        "q1": _f(q[0.25]), "q3": _f(q[0.75]), "iqr": _f(q[0.75] - q[0.25]),
        "percentiles": {"p1": _f(q[0.01]), "p5": _f(q[0.05]), "p10": _f(q[0.10]), "p25": _f(q[0.25]),
                        "p50": _f(q[0.5]), "p75": _f(q[0.75]), "p90": _f(q[0.90]), "p95": _f(q[0.95]),
                        "p99": _f(q[0.99])},
        "skewness": skew, "kurtosis": kurt, "shape": st.shape_label(skew, kurt),
        "normality": st.normality(v),
        "sum": _f(v.sum()),
        "zeros": int((v == 0).sum()), "negatives": int((v < 0).sum()),
        "is_integer": bool(np.allclose(v.to_numpy(), np.round(v.to_numpy()))) if len(v) else False,
        "outliers": st.outlier_summary(v),
        "histogram": st.histogram(v),
        "box": st.box_stats(v),
    }


def profile_categorical(s: pd.Series, top: int = 20) -> Dict[str, Any]:
    nn = s.dropna()
    if len(nn) == 0:
        return {"top_values": [], "other_count": 0}
    vc = nn.astype(str).value_counts()
    total = int(vc.sum())
    tops = [{"value": str(k), "count": int(c), "pct": c / total * 100} for k, c in vc.head(top).items()]
    p = vc / total
    entropy = float(-(p * np.log2(p)).sum())
    max_ent = math.log2(len(vc)) if len(vc) > 1 else 1
    out = {
        "top_values": to_jsonable(tops), "other_count": int(vc.iloc[top:].sum()) if len(vc) > top else 0,
        "other_levels": int(max(0, len(vc) - top)),
        "mode": str(vc.index[0]), "mode_pct": float(vc.iloc[0] / total * 100),
        "entropy_normalized": _f(entropy / max_ent) if max_ent > 0 else None,
        "imbalance_ratio": _f(vc.iloc[0] / vc.iloc[-1]) if len(vc) > 1 else None,
    }
    if is_text_dtype(s):
        lens = nn.astype(str).str.len()
        out.update({"avg_length": _f(lens.mean()), "min_length": int(lens.min()), "max_length": int(lens.max())})
    return out


def profile_datetime(s: pd.Series) -> Dict[str, Any]:
    v = pd.to_datetime(s.dropna())
    if len(v) == 0:
        return {}
    freq, info = st.infer_frequency(v)
    counts = v.dt.to_period("M" if info.get("span_days", 0) > 60 else "D").value_counts().sort_index()
    if len(counts) > 200:
        counts = v.dt.to_period("Y").value_counts().sort_index()
    dup_ts = int(v.duplicated().sum())
    return {
        "min": v.min().isoformat(), "max": v.max().isoformat(),
        "span_days": _f((v.max() - v.min()).total_seconds() / 86400),
        "frequency": info, "duplicate_timestamps": dup_ts,
        "counts_by_period": {"labels": [str(p) for p in counts.index], "counts": counts.astype(int).tolist()},
        "weekday_counts": v.dt.day_name().value_counts().reindex(
            ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"], fill_value=0
        ).astype(int).to_dict(),
    }


def profile_column(df: pd.DataFrame, info: Dict[str, Any]) -> Dict[str, Any]:
    c = info["name"]
    s = df[c]
    n = len(s)
    out = dict(info)
    out["count"] = int(s.notna().sum())
    out["unique_pct"] = round(info["n_unique"] / max(out["count"], 1) * 100, 2)
    lt = info["logical_type"]
    if lt in ("numeric",) or (lt in ("identifier", "boolean") and pd.api.types.is_numeric_dtype(s)
                                and not pd.api.types.is_bool_dtype(s)):
        out["numeric"] = profile_numeric(s)
    if lt == "datetime":
        out["datetime"] = profile_datetime(s)
    if lt in ("categorical", "boolean", "text", "identifier", "constant") or is_text_dtype(s) or pd.api.types.is_bool_dtype(s):
        out["categorical"] = profile_categorical(s)
    return out


def quality_score(df: pd.DataFrame, schema: List[Dict], issues: Optional[List[Dict]] = None) -> Dict[str, Any]:
    n, m = df.shape
    cells = n * m
    missing = int(df.isna().sum().sum())
    dups = int(df.duplicated().sum()) if n else 0
    completeness = 1 - missing / cells if cells else 1.0
    uniqueness = 1 - dups / n if n else 1.0
    flagged = set()
    for i in issues or []:
        if i["issue"] in VALIDITY_ISSUES:
            flagged.update(i.get("columns", []))
    validity = 1 - len(flagged) / m if m else 1.0
    useless = sum(1 for c in schema if c["logical_type"] in ("empty", "constant"))
    usefulness = 1 - useless / m if m else 1.0
    weights = {"completeness": 0.4, "uniqueness": 0.2, "validity": 0.25, "usefulness": 0.15}
    comps = {"completeness": completeness, "uniqueness": uniqueness, "validity": validity, "usefulness": usefulness}
    score = sum(weights[k] * comps[k] for k in weights) * 100
    label = "Excellent" if score >= 90 else "Good" if score >= 75 else "Fair" if score >= 60 else "Poor"
    return {
        "score": round(score, 1), "label": label,
        "components": {k: {"value": round(v * 100, 2), "weight": weights[k]} for k, v in comps.items()},
        "details": {"missing_cells": missing, "total_cells": cells, "duplicate_rows": dups,
                    "columns_with_validity_issues": sorted(flagged), "empty_or_constant_columns": useless},
        "formula": "100 × (0.40·completeness + 0.20·row uniqueness + 0.25·validity + 0.15·usefulness). "
                   "Completeness = non-missing cells / cells; uniqueness = 1 − duplicate rows / rows; "
                   "validity = share of columns without type or format problems; usefulness = share of columns "
                   "that are not empty or constant. This is a heuristic summary, not a statistical measure.",
    }


def profile_dataset(df: pd.DataFrame, schema: Optional[List[Dict]] = None, issues: Optional[List[Dict]] = None) -> Dict[str, Any]:
    schema = schema or infer_schema(df)
    if issues is None:
        from .cleaner import detect_issues
        issues = detect_issues(df, schema)
    n, m = df.shape
    cols = analysis_columns(df, schema)
    type_counts: Dict[str, int] = {}
    for c in schema:
        type_counts[c["logical_type"]] = type_counts.get(c["logical_type"], 0) + 1
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        columns = [profile_column(df, c) for c in schema]
    missing_by_col = sorted(
        [{"column": c["name"], "missing": c["n_missing"], "pct": c["missing_pct"]} for c in schema if c["n_missing"]],
        key=lambda x: -x["missing"])
    row_missing = df.isna().sum(axis=1)
    dup_rows = int(df.duplicated().sum())
    return to_jsonable({
        "shape": {"rows": n, "columns": m},
        "memory_mb": round(float(df.memory_usage(deep=True).sum()) / 1e6, 3),
        "type_counts": type_counts,
        "analysis_columns": cols,
        "duplicates": {"rows": dup_rows, "pct": dup_rows / n * 100 if n else 0,
                       "duplicate_columns": _duplicate_columns(df)},
        "missing": {
            "cells": int(df.isna().sum().sum()), "pct": float(df.isna().sum().sum() / (n * m) * 100) if n * m else 0,
            "by_column": missing_by_col,
            "rows_with_missing": int((row_missing > 0).sum()),
            "complete_rows": int((row_missing == 0).sum()),
            "co_missing": _co_missing(df, [x["column"] for x in missing_by_col[:15]]),
        },
        "quality": quality_score(df, schema, issues),
        "columns": columns,
        "issues_count": {s: sum(1 for i in issues if i["severity"] == s) for s in ("high", "medium", "low", "info")},
    })


def _duplicate_columns(df: pd.DataFrame, limit: int = 60) -> List[List[str]]:
    """Columns with identical content (checked on hashes for speed)."""
    if df.shape[1] < 2:
        return []
    cols = list(df.columns[:limit])
    hashes: Dict[int, List[str]] = {}
    for c in cols:
        try:
            h = int(pd.util.hash_pandas_object(df[c], index=False).sum())
        except TypeError:
            continue
        hashes.setdefault(h, []).append(str(c))
    groups = []
    for g in hashes.values():
        if len(g) > 1 and all(df[g[0]].equals(df[x]) for x in g[1:]):
            groups.append(g)
    return groups


def _co_missing(df: pd.DataFrame, cols: List[str]) -> List[Dict[str, Any]]:
    out = []
    m = df[cols].isna() if cols else None
    for i, a in enumerate(cols):
        for b in cols[i + 1:]:
            both = int((m[a] & m[b]).sum())
            either = int((m[a] | m[b]).sum())
            if either and both / either >= 0.5 and both >= 3:
                out.append({"a": a, "b": b, "both_missing": both, "jaccard": both / either})
    return sorted(out, key=lambda x: -x["jaccard"])[:10]


# ── Correlations / associations ───────────────────────────────────────────────

def correlations(df: pd.DataFrame, schema: Optional[List[Dict]] = None, max_cols: int = 30) -> Dict[str, Any]:
    schema = schema or infer_schema(df)
    cols = analysis_columns(df, schema)
    num = [c for c in cols["numeric"] if df[c].nunique() > 1]
    truncated = False
    if len(num) > max_cols:
        var = df[num].std() / (df[num].mean().abs() + 1e-9)
        num = list(var.sort_values(ascending=False).index[:max_cols])
        truncated = True
    result: Dict[str, Any] = {"columns": num, "truncated": truncated, "pairs": [], "pearson": [], "spearman": []}
    if len(num) >= 2:
        sub = df[num].astype(float)
        pear = sub.corr(method="pearson", min_periods=5)
        spear = sub.corr(method="spearman", min_periods=5)
        result["pearson"] = [[_f(pear.loc[a, b]) for b in num] for a in num]
        result["spearman"] = [[_f(spear.loc[a, b]) for b in num] for a in num]
        pairs = []
        for i, a in enumerate(num):
            for b in num[i + 1:]:
                r = _f(pear.loc[a, b])
                rs = _f(spear.loc[a, b])
                if r is None and rs is None:
                    continue
                pairs.append((a, b, r, rs))
        pairs.sort(key=lambda x: -max(abs(x[2] or 0), abs(x[3] or 0)))
        tested = []
        for a, b, r, rs in pairs[:40]:
            t = st.corr_test(df[a], df[b], "pearson")
            ts = st.corr_test(df[a], df[b], "spearman")
            tested.append({"a": a, "b": b, "pearson": r, "spearman": rs, "p_value": t["p_value"],
                           "spearman_p": ts["p_value"], "n": t["n"], "ci95": t["ci95"],
                           "strength": st.strength_label(max(abs(r or 0), abs(rs or 0))),
                           "nonlinear_hint": (rs is not None and r is not None and abs(rs) - abs(r) > 0.15)})
        adj = st.benjamini_hochberg([p["p_value"] for p in tested])
        for p, q in zip(tested, adj):
            p["p_adjusted"] = q
        result["pairs"] = tested
    cats = [c for c in cols["categorical"] if 2 <= df[c].nunique() <= 30][:12]
    assoc = []
    for i, a in enumerate(cats):
        for b in cats[i + 1:]:
            cv = st.cramers_v(df[a], df[b])
            if cv["v"] is not None:
                assoc.append({"a": a, "b": b, **cv, "strength": st.strength_label(cv["v"])})
    assoc.sort(key=lambda x: -(x["v"] or 0))
    result["categorical_associations"] = assoc[:20]
    mixed = []
    for c in cats[:10]:
        for nc in num[:15]:
            e = st.eta_squared(df[nc], df[c])
            if e is not None:
                mixed.append({"categorical": c, "numeric": nc, "eta_squared": e, "effect": st.effect_label_eta(e)})
    mixed.sort(key=lambda x: -(x["eta_squared"] or 0))
    result["numeric_by_category"] = mixed[:20]
    result["notes"] = [
        "Pearson measures linear association; Spearman measures monotonic association and is robust to outliers.",
        "p-values are adjusted for multiple comparisons (Benjamini–Hochberg). Correlation does not imply causation.",
    ]
    return to_jsonable(result)


# ── Target analysis ─────────────────────────────────────────────────────────────

def target_analysis(df: pd.DataFrame, target: str, schema: Optional[List[Dict]] = None) -> Dict[str, Any]:
    from core.errors import ValidationFailed
    if target not in df.columns:
        raise ValidationFailed(f"Unknown target column '{target}'")
    schema = schema or infer_schema(df)
    sch = {c["name"]: c for c in schema}
    cols = analysis_columns(df, schema)
    y = df[target]
    lt = sch[target]["logical_type"]
    is_cls = lt in ("categorical", "boolean") or (lt == "numeric" and y.nunique() <= 10 and sch[target].get("n_unique", 99) <= 10)
    out: Dict[str, Any] = {"target": target, "logical_type": lt, "task_hint": "classification" if is_cls else "regression",
                           "missing": int(y.isna().sum()), "features": []}
    if is_cls:
        vc = y.astype(str).value_counts()
        out["class_balance"] = [{"class": k, "count": int(v), "pct": v / vc.sum() * 100} for k, v in vc.items()][:30]
        out["imbalance_ratio"] = _f(vc.iloc[0] / vc.iloc[-1]) if len(vc) > 1 else None
        out["minority_pct"] = _f(vc.iloc[-1] / vc.sum() * 100) if len(vc) else None
        for c in cols["numeric"]:
            if c == target:
                continue
            g = st.group_test(df[c], y)
            out["features"].append({"feature": c, "kind": "numeric", "measure": "eta²",
                                    "value": g["eta_squared"], "p_value": g["kruskal_p"],
                                    "effect": st.effect_label_eta(g["eta_squared"])})
        for c in cols["categorical"]:
            if c == target:
                continue
            cv = st.cramers_v(df[c], y)
            out["features"].append({"feature": c, "kind": "categorical", "measure": "Cramér's V",
                                    "value": cv["v"], "p_value": cv["p_value"], "effect": st.strength_label(cv["v"])})
    else:
        out["distribution"] = profile_numeric(y) if pd.api.types.is_numeric_dtype(y) else None
        for c in cols["numeric"]:
            if c == target:
                continue
            t = st.corr_test(df[c], y, "spearman")
            out["features"].append({"feature": c, "kind": "numeric", "measure": "Spearman ρ", "value": t["r"],
                                    "p_value": t["p_value"], "effect": st.strength_label(t["r"])})
        for c in cols["categorical"]:
            if c == target:
                continue
            g = st.group_test(y, df[c])
            out["features"].append({"feature": c, "kind": "categorical", "measure": "eta²", "value": g["eta_squared"],
                                    "p_value": g["kruskal_p"], "effect": st.effect_label_eta(g["eta_squared"])})
    adj = st.benjamini_hochberg([f["p_value"] for f in out["features"]])
    for f, q in zip(out["features"], adj):
        f["p_adjusted"] = q
    out["features"].sort(key=lambda f: -abs(f["value"] or 0))
    out["note"] = ("Associations are bivariate and do not account for other variables. They describe the data; "
                   "they do not establish that a feature causes the target.")
    return to_jsonable(out)
