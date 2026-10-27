"""
v4-compatible EDA summary.

Kept for the legacy ``/api/analyze`` endpoint. Unlike v4 it excludes identifier
columns from numeric statistics and time series, uses an appropriate time
frequency and labels the aggregation used. New code should use
``agents.profiling`` directly.
"""
from __future__ import annotations

from typing import Any, Dict

import numpy as np
import pandas as pd

from core.jsonsafe import to_jsonable

from . import stats as st
from .schema import analysis_columns, infer_schema


def safe(v):
    return to_jsonable(v)


def run_eda(df: pd.DataFrame) -> Dict[str, Any]:
    schema = infer_schema(df)
    smap = {c["name"]: c for c in schema}
    cols = analysis_columns(df, schema)
    numeric_cols, cat_cols, date_cols = cols["numeric"], cols["all_categorical"], cols["datetime"]

    column_stats = []
    for c in df.columns:
        s = df[c]
        stat = {"name": c, "dtype": str(s.dtype), "logical_type": smap[c]["logical_type"],
                "missing": int(s.isna().sum()), "missing_pct": round(s.isna().mean() * 100, 1),
                "unique": int(smap[c]["n_unique"])}
        if c in numeric_cols:
            v = s.dropna().astype(float)
            if len(v):
                stat.update({"mean": safe(v.mean()), "median": safe(v.median()), "std": safe(v.std()),
                             "min": safe(v.min()), "max": safe(v.max()), "q25": safe(v.quantile(0.25)),
                             "q75": safe(v.quantile(0.75)), "skew": safe(v.skew()), "kurtosis": safe(v.kurtosis())})
        column_stats.append(stat)

    correlation: Dict[str, Dict[str, Any]] = {}
    if len(numeric_cols) >= 2:
        corr = df[numeric_cols].astype(float).corr()
        for a in numeric_cols:
            correlation[a] = {b: safe(corr.loc[a, b]) for b in numeric_cols
                              if b != a and pd.notna(corr.loc[a, b]) and abs(corr.loc[a, b]) > 0.3}

    distributions = {}
    for c in numeric_cols[:8]:
        v = df[c].dropna()
        if len(v) >= 5:
            counts, edges = np.histogram(v.astype(float), bins=20)
            distributions[c] = {"bins": [safe(e) for e in edges[:-1]], "counts": [int(x) for x in counts]}

    top_values = {}
    for c in cat_cols[:6]:
        vc = df[c].astype("string").value_counts().head(10)
        top_values[c] = {"labels": [str(x) for x in vc.index], "values": [int(x) for x in vc.values]}

    time_series = {}
    if date_cols:
        d = date_cols[0]
        freq, _ = st.infer_frequency(df[d])
        for c in numeric_cols[:3]:
            agg = st.default_agg(c, smap[c]["semantic_type"])
            ts = st.aggregate_series(df, d, c, freq, agg).dropna()
            if len(ts) >= 3:
                time_series[c] = {"dates": [str(t.date()) for t in ts.index], "values": [safe(v) for v in ts.values],
                                  "aggregation": agg, "frequency": freq}

    return {
        "shape": {"rows": int(len(df)), "cols": int(len(df.columns))},
        "numeric_cols": numeric_cols, "categorical_cols": cat_cols, "date_cols": date_cols,
        "identifier_cols": cols["identifier"], "text_cols": cols["text"],
        "column_stats": column_stats, "correlation": correlation, "distributions": distributions,
        "top_values": top_values, "time_series": time_series,
        "missing_summary": {c: int(df[c].isna().sum()) for c in df.columns if df[c].isna().any()},
    }
