"""
Evidence-based insight generation.

Each finding states:
  1. what was observed,
  2. the evidence (metric, test, sample size),
  3. why it may matter,
  4. possible interpretations,
  5. recommended follow-up analysis,
and its ``claim_type``: ``observation`` (a computed fact), ``association``
(a statistical relationship, never causal) or ``hypothesis``.

Findings are ranked by a transparent priority score combining evidence
strength, practical impact and relevance to the selected target.
"""
from __future__ import annotations

import math
import warnings
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from core.jsonsafe import to_jsonable

from . import stats as st
from .schema import analysis_columns, infer_schema, rank_measures

ICONS = {"data_quality": "alert-triangle", "distribution": "activity", "relationship": "trending-up",
         "segment": "bar-chart", "trend": "trending-up", "anomaly": "alert-circle", "summary": "database",
         "concentration": "pie-chart"}


def _fmt(v: Optional[float], digits: int = 2) -> str:
    if v is None:
        return "—"
    if abs(v) >= 1e6:
        return f"{v / 1e6:,.{digits}f}M"
    if abs(v) >= 1e4:
        return f"{v:,.0f}"
    return f"{v:,.{digits}f}"


def _p(p: Optional[float]) -> str:
    if p is None:
        return "n/a"
    return "< 0.001" if p < 0.001 else f"{p:.3f}"


def _confidence(p: Optional[float], n: int, effect_ok: bool = True) -> str:
    if p is None:
        return "medium" if n >= 30 else "low"
    if p < 0.01 and n >= 30 and effect_ok:
        return "high"
    if p < 0.05:
        return "medium"
    return "low"


class _Builder:
    def __init__(self, target: Optional[str]):
        self.items: List[Dict[str, Any]] = []
        self.target = target

    def add(self, *, category: str, title: str, observation: str, evidence: Dict[str, Any], evidence_text: str,
            why: str, interpretations: List[str], follow_up: List[str], claim_type: str, confidence: str,
            strength: float, impact: float, columns: List[str], chart: Optional[Dict] = None):
        relevance = 1.3 if self.target and self.target in columns else 1.0
        score = max(0.0, min(100.0, 100 * strength * impact * relevance))
        self.items.append(to_jsonable({
            "id": f"{category}:{'|'.join(columns)}:{len(self.items)}",
            "category": category, "type": category, "icon": ICONS.get(category, "zap"),
            "title": title, "observation": observation, "evidence": evidence, "evidence_text": evidence_text,
            "why_it_matters": why, "interpretations": interpretations, "follow_up": follow_up,
            "claim_type": claim_type, "confidence": confidence, "score": round(score, 1),
            "columns": columns, "chart": chart,
            # v4 compatibility fields
            "text": f"{observation} {evidence_text}",
        }))


def generate_findings(df: pd.DataFrame, schema: Optional[List[Dict]] = None, issues: Optional[List[Dict]] = None,
                      target: Optional[str] = None, max_findings: int = 15) -> List[Dict[str, Any]]:
    schema = schema or infer_schema(df)
    sch = {c["name"]: c for c in schema}
    cols = analysis_columns(df, schema)
    n, m = df.shape
    b = _Builder(target if target in df.columns else None)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        _quality(df, sch, cols, issues or [], b)
        _distributions(df, cols, b)
        _relationships(df, cols, b)
        _segments(df, sch, cols, b)
        _trends(df, sch, cols, b)
        _concentration(df, cols, b)
    items = sorted(b.items, key=lambda x: -x["score"])
    for i, it in enumerate(items, 1):
        it["priority"] = i
    return items[:max_findings]


# ── generators ────────────────────────────────────────────────────────────────

def _quality(df, sch, cols, issues, b: _Builder):
    n, m = df.shape
    miss = df.isna().sum()
    total_missing = int(miss.sum())
    if total_missing:
        worst = miss.idxmax()
        pct = miss[worst] / n * 100
        b.add(category="data_quality", title=f"Missing data concentrated in '{worst}'",
              observation=f"{int((miss > 0).sum())} of {m} columns have missing values; '{worst}' is the most "
                          f"affected with {int(miss[worst]):,} missing ({pct:.1f}%).",
              evidence={"missing_cells": total_missing, "missing_pct_overall": total_missing / (n * m) * 100,
                        "worst_column": worst, "worst_pct": pct},
              evidence_text=f"Overall {total_missing / (n * m) * 100:.1f}% of cells are empty.",
              why="Statistics and models silently use fewer rows (or imputed estimates) for affected columns, "
                  "which can bias results if values are not missing at random.",
              interpretations=["Values may be optional fields that were not filled in.",
                               "Missingness may be systematic (e.g. one source or period lacks the field)."],
              follow_up=["Check whether missingness differs across segments or time periods.",
                         "Choose an imputation strategy in the Clean step, or let the model pipeline impute."],
              claim_type="observation", confidence="high", strength=min(1.0, 0.4 + pct / 100), impact=0.9,
              columns=[str(worst)])
    dups = int(df.duplicated().sum())
    if dups:
        b.add(category="data_quality", title=f"{dups:,} duplicate rows",
              observation=f"{dups:,} rows ({dups / n:.1%}) are exact copies of another row.",
              evidence={"duplicate_rows": dups, "pct": dups / n * 100}, evidence_text="Exact match on every column.",
              why="Duplicates inflate counts and totals and can leak between training and test data.",
              interpretations=["Records may have been loaded twice.", "They may be legitimate repeated events."],
              follow_up=["Inspect duplicates in the Data view before removing them."],
              claim_type="observation", confidence="high", strength=min(1.0, 0.5 + dups / n * 5), impact=0.85, columns=[])
    idc = cols["identifier"]
    if idc:
        b.add(category="data_quality", title="Identifier columns excluded from analysis",
              observation=f"{', '.join(idc[:5])} {'look' if len(idc) > 1 else 'looks'} like row identifiers.",
              evidence={"columns": idc}, evidence_text="Near-unique values per row.",
              why="Identifiers have no analytical meaning; models trained on them memorise rows and fail on new data.",
              interpretations=["They are useful for joins and lookups, not for statistics."],
              follow_up=["Use them as join keys in Multi-file analysis if needed."],
              claim_type="observation", confidence="high", strength=0.35, impact=0.5, columns=idc[:5])
    vals = [i for i in issues if i["issue"] in ("numeric_as_text", "dates_as_text", "inconsistent_labels",
                                                "null_tokens", "number_with_unit", "implausible_dates")]
    if vals:
        b.add(category="data_quality", title=f"{len(vals)} formatting issue(s) detected",
              observation="; ".join(i["title"] for i in vals[:4]) + ("…" if len(vals) > 4 else "") + ".",
              evidence={"issues": [i["issue"] for i in vals]}, evidence_text="Detected by the cleaning engine.",
              why="Mis-typed or inconsistent values are excluded from numeric analysis or split one category into several.",
              interpretations=["These are typical of manual data entry or exports from different systems."],
              follow_up=["Review and apply the corresponding suggestions in the Clean step."],
              claim_type="observation", confidence="high", strength=0.7, impact=0.8,
              columns=sorted({c for i in vals for c in i.get("columns", [])})[:6])


def _distributions(df, cols, b: _Builder):
    found = 0
    for c in cols["numeric"]:
        v = df[c].dropna()
        if len(v) < 20 or v.nunique() < 5:
            continue
        sk = float(v.skew())
        if abs(sk) > 1.5 and found < 2:
            mean, med = float(v.mean()), float(v.median())
            gap = (mean - med) / abs(med) * 100 if med else None
            b.add(category="distribution", title=f"'{c}' is strongly {'right' if sk > 0 else 'left'}-skewed",
                  observation=f"The mean ({_fmt(mean)}) is {'above' if mean > med else 'below'} the median ({_fmt(med)})"
                              + (f" by {abs(gap):.0f}%" if gap is not None else "") + ".",
                  evidence={"skewness": sk, "mean": mean, "median": med, "n": len(v)},
                  evidence_text=f"Skewness = {sk:.2f} (n = {len(v):,}).",
                  why="A few extreme values dominate the average; the median better describes a typical record.",
                  interpretations=["Natural for amounts, prices and durations (many small, few large values).",
                                   "Could also be caused by data-entry errors in the tail."],
                  follow_up=["Report medians alongside means.", "Consider a log transform for linear models.",
                             "Inspect the largest values in the Profile view."],
                  claim_type="observation", confidence="high", strength=min(1.0, abs(sk) / 4 + 0.3), impact=0.55,
                  columns=[c], chart={"type": "histogram", "x": c})
            found += 1
    best = None
    for c in cols["numeric"]:
        v = df[c].dropna()
        if len(v) < 20:
            continue
        o = st.outlier_summary(v)
        if o["iqr_extreme"] and (best is None or o["iqr_extreme"] / len(v) > best[1]):
            best = (c, o["iqr_extreme"] / len(v), o, len(v))
    if best:
        c, share, o, nv = best
        b.add(category="anomaly", title=f"Extreme values in '{c}'",
              observation=f"{o['iqr_extreme']} value(s) ({share:.1%}) lie beyond 3×IQR from the quartiles "
                          f"(outside {_fmt(o['fences_extreme'][0])} – {_fmt(o['fences_extreme'][1])}).",
              evidence={"extreme_count": o["iqr_extreme"], "mild_count": o["iqr_mild"], "fences": o["fences_extreme"], "n": nv},
              evidence_text=f"{o['iqr_mild']} value(s) exceed the usual 1.5×IQR fences.",
              why="Extreme values can dominate means, regression fits and charts.",
              interpretations=["Genuine rare events (large orders, VIP customers).", "Unit or entry errors."],
              follow_up=["Inspect the rows in Data view (sort by this column).",
                         "Run anomaly detection in the Model step to look at multivariate outliers."],
              claim_type="observation", confidence="medium", strength=min(1.0, 0.4 + share * 10), impact=0.6,
              columns=[c], chart={"type": "box", "y": c})


def _relationships(df, cols, b: _Builder):
    num = [c for c in cols["numeric"] if df[c].nunique() > 2]
    pairs = []
    for i, a in enumerate(num[:25]):
        for c in num[i + 1:25]:
            t = st.corr_test(df[a], df[c], "spearman")
            if t["r"] is not None:
                pairs.append((a, c, t))
    adj = st.benjamini_hochberg([p[2]["p_value"] for p in pairs])
    ranked = sorted([(a, c, t, q) for (a, c, t), q in zip(pairs, adj) if q is not None and q < 0.05 and abs(t["r"]) >= 0.3],
                    key=lambda x: -abs(x[2]["r"]))
    for a, c, t, q in ranked[:3]:
        pr = st.corr_test(df[a], df[c], "pearson")
        r = t["r"]
        redundant = abs(r) >= 0.9
        b.add(category="relationship",
              title=f"{st.strength_label(r).capitalize()} {'positive' if r > 0 else 'negative'} association: '{a}' and '{c}'",
              observation=f"Higher '{a}' tends to go with {'higher' if r > 0 else 'lower'} '{c}'.",
              evidence={"spearman_rho": r, "pearson_r": pr["r"], "p_adjusted": q, "n": t["n"], "ci95": t["ci95"]},
              evidence_text=f"Spearman ρ = {r:.2f} (95% CI {t['ci95'][0]:.2f} to {t['ci95'][1]:.2f}), "
                            f"Pearson r = {pr['r']:.2f}, n = {t['n']:,}, adjusted p {_p(q)}." if t["ci95"] and pr["r"] is not None
                            else f"Spearman ρ = {r:.2f}, n = {t['n']:,}.",
              why=("The two columns carry almost the same information; keeping both can destabilise linear models."
                   if redundant else "One variable may help explain or predict the other."),
              interpretations=["A direct relationship (one influences the other).",
                               "Both may be driven by a third variable (confounding).",
                               "Possibly a definitional link (e.g. revenue = price × quantity)."],
              follow_up=["Plot the scatter chart to check linearity and outliers.",
                         "Control for other variables (Diagnostics) before drawing conclusions."],
              claim_type="association", confidence=_confidence(q, t["n"]), strength=min(1.0, abs(r)), impact=0.75,
              columns=[a, c], chart={"type": "scatter", "x": a, "y": c})


def _key_measures(df, sch, cols, k: int = 3) -> List[str]:
    return rank_measures(cols["numeric"], sch)[:k]


def _segments(df, sch, cols, b: _Builder):
    measures = _key_measures(df, sch, cols, 4)
    dims = [c for c in cols["categorical"] if 2 <= df[c].nunique() <= 30]
    tests = []
    for mcol in measures:
        for d in dims[:10]:
            g = st.group_test(df[mcol], df[d], min_group=5)
            if g["kruskal_p"] is not None and g["eta_squared"] is not None:
                tests.append((mcol, d, g))
    adj = st.benjamini_hochberg([t[2]["kruskal_p"] for t in tests])
    sig = sorted([(mcol, d, g, q) for (mcol, d, g), q in zip(tests, adj) if q < 0.05 and (g["eta_squared"] or 0) >= 0.01],
                 key=lambda x: -x[2]["eta_squared"])
    used, used_dims = set(), set()
    for mcol, d, g, q in sig:
        if mcol in used or d in used_dims or len(used) >= 3:
            continue
        used.add(mcol)
        used_dims.add(d)
        grp = df.groupby(d, observed=True)[mcol].agg(["mean", "median", "count"])
        grp = grp[grp["count"] >= 5]
        if len(grp) < 2:
            continue
        hi, lo = grp["mean"].idxmax(), grp["mean"].idxmin()
        tg = st.two_group(df.loc[df[d] == hi, mcol], df.loc[df[d] == lo, mcol])
        gap = (grp.loc[hi, "mean"] - grp.loc[lo, "mean"]) / abs(grp.loc[lo, "mean"]) * 100 if grp.loc[lo, "mean"] else None
        b.add(category="segment", title=f"'{mcol}' differs across '{d}'",
              observation=f"Average '{mcol}' is highest for '{hi}' ({_fmt(grp.loc[hi, 'mean'])}, n={int(grp.loc[hi, 'count'])}) "
                          f"and lowest for '{lo}' ({_fmt(grp.loc[lo, 'mean'])}, n={int(grp.loc[lo, 'count'])})"
                          + (f" — a {abs(gap):.0f}% gap." if gap is not None else "."),
              evidence={"eta_squared": g["eta_squared"], "kruskal_p_adjusted": q, "groups": int(len(grp)),
                        "best_vs_worst": tg},
              evidence_text=f"'{d}' accounts for {g['eta_squared'] * 100:.1f}% of the variance in '{mcol}' "
                            f"(η² = {g['eta_squared']:.3f}, {st.effect_label_eta(g['eta_squared'])} effect; "
                            f"Kruskal–Wallis adjusted p {_p(q)}).",
              why="Segment differences point to where performance diverges and where to look first.",
              interpretations=["Segments may genuinely behave differently.",
                               "The gap may reflect differences in mix (e.g. product types) rather than the segment itself."],
              follow_up=[f"Run Diagnostics on '{mcol}' to check confounders and decompose the gap.",
                         "Compare medians to make sure a few extreme rows do not drive the gap."],
              claim_type="association", confidence=_confidence(q, int(grp["count"].sum()),
                                                                (g["eta_squared"] or 0) >= 0.06),
              strength=min(1.0, 0.3 + (g["eta_squared"] or 0) * 3), impact=0.85, columns=[mcol, d],
              chart={"type": "box", "y": mcol, "x": d})


def _trends(df, sch, cols, b: _Builder):
    if not cols["datetime"]:
        return
    dcol = cols["datetime"][0]
    measures = _key_measures(df, sch, cols, 2)
    for mcol in measures:
        freq, info = st.infer_frequency(df[dcol])
        if info.get("span_days", 0) < 60:
            continue
        agg = st.default_agg(mcol, sch.get(mcol, {}).get("semantic_type"))
        series = st.aggregate_series(df, dcol, mcol, freq, agg)
        counts = st.aggregate_series(df, dcol, None, freq, "count")
        # Drop partial edge periods with very few records (< 30% of median) to avoid fake drops.
        if len(counts) >= 4:
            med = counts.median()
            keep = counts >= 0.3 * med
            series = series[keep.reindex(series.index, fill_value=False)]
        series = series.dropna()
        if len(series) < 6:
            continue
        idx = np.arange(len(series))
        from scipy import stats as sps
        tau, p = sps.kendalltau(idx, series.to_numpy())
        if p is None or np.isnan(p) or p >= 0.05:
            continue
        slope, intercept = np.polyfit(idx, series.to_numpy(), 1)
        first, last = float(series.iloc[0]), float(series.iloc[-1])
        chg = (last - first) / abs(first) * 100 if first else None
        unit = st.FREQ_LABELS.get(freq, "period")
        b.add(category="trend", title=f"'{mcol}' shows a {'rising' if tau > 0 else 'falling'} trend",
              observation=f"{agg.title()} of '{mcol}' per {unit} moved from {_fmt(first)} to {_fmt(last)}"
                          + (f" ({chg:+.1f}%)" if chg is not None else "") + f" over {len(series)} {unit}s.",
              evidence={"kendall_tau": float(tau), "p_value": float(p), "slope_per_period": float(slope),
                        "periods": int(len(series)), "aggregation": agg, "frequency": freq},
              evidence_text=f"Mann–Kendall-style test: Kendall τ = {tau:.2f}, p {_p(float(p))}; "
                            f"linear slope {_fmt(float(slope))} per {unit}.",
              why="A sustained trend affects planning and should be checked before forecasting.",
              interpretations=["Real growth or decline.", "A change in how data was collected or a change in volume."],
              follow_up=["Open Forecasting to project the series with backtested models.",
                         "Check whether the number of records per period changed as well."],
              claim_type="observation", confidence="high" if p < 0.01 and len(series) >= 12 else "medium",
              strength=min(1.0, abs(tau) + 0.2), impact=0.8, columns=[mcol, dcol],
              chart={"type": "line", "x": dcol, "y": mcol, "agg": agg})


def _concentration(df, cols, b: _Builder):
    for c in cols["categorical"][:6]:
        vc = df[c].value_counts(normalize=True)
        if len(vc) >= 3 and vc.iloc[0] >= 0.6:
            b.add(category="concentration", title=f"'{c}' is dominated by one value",
                  observation=f"'{vc.index[0]}' accounts for {vc.iloc[0]:.0%} of non-missing rows.",
                  evidence={"top_value": str(vc.index[0]), "share": float(vc.iloc[0]), "levels": int(len(vc))},
                  evidence_text=f"{len(vc)} distinct values in total.",
                  why="Comparisons for the smaller groups rest on few observations and are less reliable.",
                  interpretations=["The population is genuinely concentrated.", "The sample may under-represent other groups."],
                  follow_up=["Look at confidence intervals before comparing small groups."],
                  claim_type="observation", confidence="high", strength=0.4 + (vc.iloc[0] - 0.6), impact=0.45, columns=[c],
                  chart={"type": "bar", "x": c})
            break


# v4 compatibility -----------------------------------------------------------------

def generate_insights(df: pd.DataFrame, eda: Optional[Dict] = None) -> List[Dict[str, Any]]:
    items = generate_findings(df)
    rows, cols = df.shape
    summary = {
        "type": "summary", "category": "summary", "icon": "database", "priority": 0, "confidence": "high",
        "title": "Dataset Overview",
        "text": f"{rows:,} rows and {cols} columns analysed.",
        "observation": f"{rows:,} rows and {cols} columns analysed.", "claim_type": "observation",
    }
    return [summary] + items
