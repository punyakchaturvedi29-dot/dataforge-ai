"""
Chart recommendation and data preparation.

Charts are chosen by column type, cardinality and purpose, and the data is
aggregated on the server so the browser never receives the raw dataset.

Guard rails against misleading charts:
* bar charts of many categories show the top N plus an explicit "Other" bar;
* pie charts are only allowed for ≤ 6 parts of a non-negative total;
* scatter plots are down-sampled (and say so) above ``CHART_MAX_POINTS``;
* time series flag incomplete edge periods and state the aggregation used;
* missing values are reported, never plotted as zeros.
"""
from __future__ import annotations

import warnings
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from core.config import settings
from core.errors import ValidationFailed
from core.jsonsafe import to_jsonable

from . import stats as st
from .schema import analysis_columns, infer_schema, rank_measures

CHART_TYPES = ["histogram", "bar", "line", "scatter", "box", "heatmap", "missing", "pie"]
AGGS = ["count", "sum", "mean", "median", "min", "max"]


def _lt(schema_map: Dict, c: str) -> str:
    return schema_map.get(c, {}).get("logical_type", "")


def histogram_chart(df, x: str) -> Dict:
    s = df[x]
    if not pd.api.types.is_numeric_dtype(s) or pd.api.types.is_bool_dtype(s):
        raise ValidationFailed(f"A histogram needs a numeric column; '{x}' is {s.dtype}.")
    h = st.histogram(s)
    if h.get("discrete"):
        data = [{"label": _fmt_num(v), "value": float(v), "count": int(c)} for v, c in zip(h["values"], h["counts"])]
    else:
        e = h["edges"]
        data = [{"x0": e[i], "x1": e[i + 1], "label": f"{_fmt_num(e[i])}–{_fmt_num(e[i + 1])}", "count": int(c)}
                for i, c in enumerate(h["counts"])]
    miss = int(s.isna().sum())
    return {"type": "histogram", "x": x, "data": data, "discrete": bool(h.get("discrete")),
            "n": int(s.notna().sum()), "missing": miss,
            "notes": ([f"{miss:,} missing value(s) not shown."] if miss else []) +
                     ([] if h.get("discrete") else ["Bin width chosen with the Freedman–Diaconis rule."])}


def bar_chart(df, x: str, y: Optional[str] = None, agg: str = "count", limit: int = 15, sort: str = "value") -> Dict:
    if agg not in AGGS:
        raise ValidationFailed(f"Unknown aggregation '{agg}'")
    s = df[x]
    if pd.api.types.is_numeric_dtype(s) and s.nunique() > 30 and not pd.api.types.is_bool_dtype(s):
        raise ValidationFailed(f"'{x}' is a continuous number with many values; use a histogram instead of a bar chart.")
    if y is None or agg == "count":
        g = s.astype("string").fillna("(missing)").value_counts()
        values = g.astype(float)
        counts = g
        agg, ylab = "count", "Count"
    else:
        if not pd.api.types.is_numeric_dtype(df[y]) or pd.api.types.is_bool_dtype(df[y]):
            raise ValidationFailed(f"'{y}' must be numeric to compute a {agg}.")
        key = s.astype("string").fillna("(missing)")
        grp = df[y].groupby(key)
        values = getattr(grp, agg)()
        counts = grp.count()
        ylab = f"{agg.title()} of {y}"
    order = values.sort_values(ascending=False).index if sort == "value" else sorted(values.index, key=str)
    limit = max(2, min(int(limit), 50))
    top = list(order[:limit])
    data = [{"label": str(k), "value": to_jsonable(values[k]), "n": int(counts.get(k, 0))} for k in top]
    notes = []
    rest = [k for k in order[limit:]]
    if rest:
        if agg in ("count", "sum"):
            other_val = float(values[rest].sum())
            data.append({"label": f"Other ({len(rest)})", "value": other_val, "n": int(counts[rest].sum()), "other": True})
            notes.append(f"{len(rest)} smaller categories are combined into 'Other'.")
        else:
            notes.append(f"Showing the top {limit} of {len(order)} categories by {agg}.")
    if agg == "mean":
        small = [d["label"] for d in data if d["n"] < 5]
        if small:
            notes.append(f"Means for {', '.join(small[:4])} are based on fewer than 5 rows.")
    return {"type": "bar", "x": x, "y": y, "agg": agg, "y_label": ylab, "data": to_jsonable(data), "notes": notes,
            "n_categories": int(len(order))}


def pie_chart(df, x: str, y: Optional[str] = None) -> Dict:
    base = bar_chart(df, x, y, "count" if y is None else "sum", limit=6)
    vals = [d["value"] for d in base["data"] if d["value"] is not None]
    if any(v < 0 for v in vals):
        raise ValidationFailed("A pie chart needs non-negative values; use a bar chart instead.")
    if base["n_categories"] > 6:
        base["notes"].append("Pie charts are limited to 6 slices; smaller categories are grouped.")
    base["type"] = "pie"
    total = sum(vals) or 1
    for d in base["data"]:
        d["pct"] = (d["value"] or 0) / total * 100
    return base


def line_chart(df, x: str, y: Optional[str] = None, agg: Optional[str] = None, freq: Optional[str] = None,
               group: Optional[str] = None, schema_map: Optional[Dict] = None) -> Dict:
    if not pd.api.types.is_datetime64_any_dtype(df[x]):
        raise ValidationFailed(f"A time-series line chart needs a date/time column; '{x}' is {df[x].dtype}. "
                               "Convert it in the Clean step first.")
    if y is not None and (not pd.api.types.is_numeric_dtype(df[y]) or pd.api.types.is_bool_dtype(df[y])):
        raise ValidationFailed(f"'{y}' must be numeric.")
    sem = (schema_map or {}).get(y, {}).get("semantic_type") if y else None
    agg = agg or (st.default_agg(y, sem) if y else "count")
    if agg not in AGGS:
        raise ValidationFailed(f"Unknown aggregation '{agg}'")
    auto_freq, info = st.infer_frequency(df[x])
    freq = freq or auto_freq
    if freq not in st.FREQ_LABELS:
        raise ValidationFailed("freq must be one of h, D, W, MS, QS, YS")
    counts = st.aggregate_series(df, x, None, freq, "count")
    series_list = []
    groups = [None]
    if group:
        top = df[group].value_counts().index[:6].tolist()
        groups = top
    for gv in groups:
        sub = df if gv is None else df[df[group] == gv]
        s = st.aggregate_series(sub, x, y, freq, agg) if y else st.aggregate_series(sub, x, None, freq, "count")
        c = st.aggregate_series(sub, x, None, freq, "count")
        pts = []
        for t, v in s.items():
            n_obs = int(c.get(t, 0))
            pts.append({"t": t.isoformat(), "value": to_jsonable(v) if n_obs > 0 or agg == "count" else None, "n": n_obs})
        series_list.append({"name": str(gv) if gv is not None else (y or "records"), "points": pts})
    notes = [f"{agg.title()} per {st.FREQ_LABELS.get(freq, freq)}."]
    partial = []
    if len(counts) >= 4:
        med = counts.median()
        for t in [counts.index[0], counts.index[-1]]:
            if counts[t] < 0.5 * med:
                partial.append(t.isoformat())
    if partial:
        notes.append("Edge period(s) with unusually few records may be incomplete and are marked.")
    empty = int(sum(1 for p in series_list[0]["points"] if p["n"] == 0))
    if empty:
        notes.append(f"{empty} period(s) have no records; they are shown as gaps, not zeros." if agg != "count"
                     else f"{empty} period(s) have no records.")
    if group and df[group].nunique() > 6:
        notes.append(f"Showing the 6 most frequent values of '{group}'.")
    return {"type": "line", "x": x, "y": y, "agg": agg, "freq": freq, "series": series_list,
            "partial_periods": partial, "notes": notes, "frequency_info": to_jsonable(info)}


def scatter_chart(df, x: str, y: str, color: Optional[str] = None) -> Dict:
    for c in (x, y):
        if not pd.api.types.is_numeric_dtype(df[c]) or pd.api.types.is_bool_dtype(df[c]):
            raise ValidationFailed(f"Scatter plots need numeric columns; '{c}' is {df[c].dtype}.")
    cols = [x, y] + ([color] if color else [])
    d = df[cols].dropna(subset=[x, y])
    n_total = len(d)
    maxp = settings.chart_max_points
    sampled = n_total > maxp
    if sampled:
        d = d.sample(maxp, random_state=0)
    trend = None
    if n_total >= 5 and df[x].nunique() > 1:
        full = df[[x, y]].dropna().astype(float)
        slope, intercept = np.polyfit(full[x], full[y], 1)
        r = full[x].corr(full[y])
        trend = {"slope": float(slope), "intercept": float(intercept), "r2": float(r ** 2) if pd.notna(r) else None,
                 "x_min": float(full[x].min()), "x_max": float(full[x].max())}
    pts = []
    for row in d.itertuples(index=False, name=None):
        p = {"x": to_jsonable(row[0]), "y": to_jsonable(row[1])}
        if color:
            p["c"] = None if pd.isna(row[2]) else str(row[2])
        pts.append(p)
    notes = []
    if sampled:
        notes.append(f"Showing a random sample of {maxp:,} of {n_total:,} points; the trend line uses all points.")
    notes.append("The trend line is an ordinary least-squares fit and describes association only.")
    return {"type": "scatter", "x": x, "y": y, "color": color, "points": pts, "n_total": n_total,
            "sampled": sampled, "trend": trend, "notes": notes}


def box_chart(df, y: str, x: Optional[str] = None, limit: int = 12) -> Dict:
    if not pd.api.types.is_numeric_dtype(df[y]) or pd.api.types.is_bool_dtype(df[y]):
        raise ValidationFailed(f"Box plots need a numeric column; '{y}' is {df[y].dtype}.")
    groups = []
    notes = []
    if x:
        if df[x].nunique() > 50:
            raise ValidationFailed(f"'{x}' has too many distinct values for a grouped box plot.")
        order = df.groupby(x, observed=True)[y].median().sort_values(ascending=False).index.tolist()
        if len(order) > limit:
            notes.append(f"Showing {limit} of {len(order)} groups (highest medians).")
        for gval in order[:limit]:
            bs = st.box_stats(df.loc[df[x] == gval, y], max_outliers=50)
            if bs:
                groups.append({"label": str(gval), **bs})
        small = [g["label"] for g in groups if g["n"] < 5]
        if small:
            notes.append(f"Groups with fewer than 5 rows: {', '.join(small[:4])}.")
        test = st.group_test(df[y], df[x])
        notes.append(f"Kruskal–Wallis p = {test['kruskal_p']:.3g}, η² = {test['eta_squared']:.3f}."
                     if test["kruskal_p"] is not None and test["eta_squared"] is not None else "Group test not available.")
    else:
        bs = st.box_stats(df[y])
        groups.append({"label": y, **bs})
    notes.append("Whiskers extend to the most extreme values within 1.5×IQR; points beyond are drawn individually.")
    return {"type": "box", "x": x, "y": y, "groups": to_jsonable(groups), "notes": notes}


def heatmap_chart(df, schema: List[Dict], method: str = "pearson", columns: Optional[List[str]] = None) -> Dict:
    from .profiling import correlations
    if method not in ("pearson", "spearman"):
        raise ValidationFailed("method must be pearson or spearman")
    if columns:
        sub = df[[c for c in columns if c in df.columns]]
        sch = [s for s in schema if s["name"] in sub.columns]
    else:
        sub, sch = df, schema
    c = correlations(sub, sch, max_cols=20)
    if len(c["columns"]) < 2:
        raise ValidationFailed("A correlation heatmap needs at least two numeric measure columns.")
    return {"type": "heatmap", "method": method, "columns": c["columns"], "matrix": c[method],
            "notes": (["Showing the 20 most variable numeric columns."] if c["truncated"] else []) + c["notes"][:1]}


def missing_chart(df) -> Dict:
    miss = df.isna().mean().sort_values(ascending=False) * 100
    miss = miss[miss > 0]
    data = [{"label": str(k), "pct": float(v), "count": int(df[k].isna().sum())} for k, v in miss.head(40).items()]
    return {"type": "missing", "data": data,
            "notes": [] if data else ["No missing values."]}


def build_chart(df: pd.DataFrame, spec: Dict[str, Any], schema: Optional[List[Dict]] = None) -> Dict[str, Any]:
    schema = schema or infer_schema(df)
    smap = {c["name"]: c for c in schema}
    ctype = spec.get("type")
    if ctype not in CHART_TYPES:
        raise ValidationFailed(f"Unknown chart type '{ctype}'. Use one of {', '.join(CHART_TYPES)}.")
    for key in ("x", "y", "group", "color"):
        v = spec.get(key)
        if v is not None and v not in df.columns:
            raise ValidationFailed(f"Unknown column '{v}'")
    x, y = spec.get("x"), spec.get("y")
    for c in (x, y):
        if c and _lt(smap, c) == "identifier" and ctype not in ("bar",):
            raise ValidationFailed(f"'{c}' is an identifier; charting it is not meaningful.")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        if ctype == "histogram":
            out = histogram_chart(df, x or y)
        elif ctype == "bar":
            if not x:
                raise ValidationFailed("Choose a category column (x) for the bar chart.")
            out = bar_chart(df, x, y, spec.get("agg") or ("mean" if y else "count"), spec.get("limit", 15))
        elif ctype == "pie":
            out = pie_chart(df, x, y)
        elif ctype == "line":
            if not x:
                raise ValidationFailed("Choose a date column (x) for the line chart.")
            out = line_chart(df, x, y, spec.get("agg"), spec.get("freq"), spec.get("group"), smap)
        elif ctype == "scatter":
            if not (x and y):
                raise ValidationFailed("Choose numeric x and y columns for the scatter plot.")
            out = scatter_chart(df, x, y, spec.get("color"))
        elif ctype == "box":
            if not y:
                raise ValidationFailed("Choose a numeric column (y) for the box plot.")
            out = box_chart(df, y, x)
        elif ctype == "heatmap":
            out = heatmap_chart(df, schema, spec.get("method", "pearson"), spec.get("columns"))
        else:
            out = missing_chart(df)
    out["title"] = spec.get("title") or _default_title(out)
    return to_jsonable(out)


def _default_title(c: Dict) -> str:
    t = c["type"]
    if t == "histogram":
        return f"Distribution of {c['x']}"
    if t in ("bar", "pie"):
        return f"{c['x']} — count" if c.get("agg") == "count" else f"{c['agg'].title()} of {c['y']} by {c['x']}"
    if t == "line":
        return f"{(c.get('agg') or '').title()} of {c['y'] or 'records'} over time" if c.get("y") else "Records over time"
    if t == "scatter":
        return f"{c['y']} vs {c['x']}"
    if t == "box":
        return f"{c['y']} by {c['x']}" if c.get("x") else f"Spread of {c['y']}"
    if t == "heatmap":
        return f"Correlation matrix ({c['method'].title()})"
    return "Missing values by column"


def _fmt_num(v: float) -> str:
    v = float(v)
    if abs(v) >= 1000:
        return f"{v:,.0f}"
    if abs(v) >= 10:
        return f"{v:.1f}"
    return f"{v:.2f}".rstrip("0").rstrip(".") if v != int(v) else str(int(v))


def recommend_charts(df: pd.DataFrame, schema: Optional[List[Dict]] = None, max_charts: int = 12) -> List[Dict[str, Any]]:
    """Automatically choose a sensible, non-redundant set of charts."""
    schema = schema or infer_schema(df)
    smap = {c["name"]: c for c in schema}
    cols = analysis_columns(df, schema)
    measures = rank_measures(cols["numeric"], smap)
    dims = [c for c in cols["categorical"] if 2 <= smap[c]["n_unique"] <= 30]
    dims = sorted(dims, key=lambda c: (abs(smap[c]["n_unique"] - 6), dims.index(c)))
    specs: List[Dict[str, Any]] = []

    def add(spec, reason):
        specs.append((spec, reason))

    if df.isna().any().any():
        add({"type": "missing"}, "Shows where data is missing before you interpret other charts.")
    if cols["datetime"] and measures:
        for mcol in measures[:2]:
            add({"type": "line", "x": cols["datetime"][0], "y": mcol},
                f"'{mcol}' over time, aggregated with {st.default_agg(mcol, smap[mcol]['semantic_type'])}.")
    for mcol in measures[:3]:
        add({"type": "histogram", "x": mcol}, f"Distribution shape and spread of '{mcol}'.")
    if measures and dims:
        best = None
        for mcol in measures[:3]:
            for d in dims[:6]:
                e = st.eta_squared(df[mcol], df[d])
                if e is not None and (best is None or e > best[2]):
                    best = (mcol, d, e)
        if best:
            add({"type": "box", "y": best[0], "x": best[1]},
                f"'{best[1]}' explains the most variation in '{best[0]}' (η² = {best[2]:.2f}).")
            agg = st.default_agg(best[0], smap[best[0]]["semantic_type"])
            add({"type": "bar", "x": best[1], "y": best[0], "agg": agg},
                f"{agg.title()} of '{best[0]}' per '{best[1]}'.")
    for d in dims[:3]:
        add({"type": "bar", "x": d, "agg": "count"}, f"How records are distributed across '{d}'.")
    if len(measures) >= 2:
        best_pair, best_r = None, 0
        for i, a in enumerate(measures[:12]):
            for c in measures[i + 1:12]:
                r = df[a].corr(df[c], method="spearman")
                if pd.notna(r) and abs(r) > abs(best_r):
                    best_pair, best_r = (a, c), r
        if best_pair and abs(best_r) >= 0.2:
            add({"type": "scatter", "x": best_pair[0], "y": best_pair[1]},
                f"Strongest monotonic association among measures (Spearman ρ = {best_r:.2f}).")
    if len(measures) >= 3:
        add({"type": "heatmap", "method": "spearman"}, "Pairwise associations between all numeric measures.")

    out = []
    for spec, reason in specs[:max_charts]:
        try:
            c = build_chart(df, spec, schema)
            c["reason"] = reason
            c["spec"] = spec
            out.append(c)
        except ValidationFailed:
            continue
    return out
