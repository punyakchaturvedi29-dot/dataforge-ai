"""
Auto dashboard: domain detection, honest KPIs and a recommended chart set.

KPIs are computed directly from the data with an explicit aggregation label
("Total revenue", "Average customer_age"). Period-over-period change is only
reported when a date column exists, comparing the last *complete* period with
the one before it — never first-vs-last month of a partial range.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional

import pandas as pd

from core.jsonsafe import to_jsonable

from . import stats as st
from .charts import recommend_charts
from .schema import analysis_columns, infer_schema, name_tokens, rank_measures

DOMAIN_SIGNALS = {
    "sales": {"revenue", "sales", "order", "product", "customer", "quantity", "discount", "region", "channel", "sku"},
    "hr": {"employee", "salary", "department", "attrition", "tenure", "hire", "performance", "headcount", "leave"},
    "finance": {"profit", "expense", "budget", "cost", "cash", "balance", "tax", "invoice", "payment", "asset"},
    "healthcare": {"patient", "diagnosis", "hospital", "doctor", "treatment", "blood", "disease", "admission"},
    "marketing": {"campaign", "click", "clicks", "impression", "impressions", "conversion", "ctr", "roas", "lead", "funnel"},
}
DESCRIPTIONS = {
    "sales": "Sales & Revenue Analytics Dashboard", "hr": "Human Resources Analytics Dashboard",
    "finance": "Financial Performance Dashboard", "healthcare": "Healthcare Operations Dashboard",
    "marketing": "Marketing Campaign Dashboard", "generic": "General Analytics Dashboard",
}


def detect_domain(df: pd.DataFrame) -> Dict[str, Any]:
    toks = set()
    for c in df.columns:
        toks.update(name_tokens(c))
    scores = {d: len(toks & sig) for d, sig in DOMAIN_SIGNALS.items()}
    best = max(scores, key=scores.get)
    domain = best if scores[best] >= 2 else "generic"
    return {"domain": domain, "scores": scores,
            "matched_terms": sorted(toks & DOMAIN_SIGNALS.get(domain, set())),
            "note": "Detected from column names only; it only affects labels and chart ordering."}


def _last_full_period_change(df, dcol, mcol, agg) -> Optional[Dict[str, Any]]:
    freq, info = st.infer_frequency(df[dcol])
    if info.get("span_days", 0) < 60:
        return None
    period = "MS" if info.get("span_days", 0) > 90 else "W"
    vals = st.aggregate_series(df, dcol, mcol, period, agg)
    cnt = st.aggregate_series(df, dcol, None, period, "count")
    if len(vals) < 3:
        return None
    last_ts = pd.to_datetime(df[dcol]).max()
    end_of_last = vals.index[-1] + (pd.offsets.MonthBegin(1) if period == "MS" else pd.Timedelta(days=7))
    complete = vals if last_ts >= end_of_last - pd.Timedelta(days=1) else vals.iloc[:-1]
    if len(complete) < 2 or cnt.get(complete.index[-1], 0) == 0 or cnt.get(complete.index[-2], 0) == 0:
        return None
    cur, prev = float(complete.iloc[-1]), float(complete.iloc[-2])
    if prev == 0:
        return None
    label = "month" if period == "MS" else "week"
    return {"current": cur, "previous": prev, "change_pct": (cur - prev) / abs(prev) * 100,
            "current_period": complete.index[-1].strftime("%Y-%m-%d"),
            "previous_period": complete.index[-2].strftime("%Y-%m-%d"),
            "label": f"vs previous {label} (last complete {label})"}


def kpi_cards(df: pd.DataFrame, schema: Optional[List[Dict]] = None, k: int = 4) -> List[Dict[str, Any]]:
    schema = schema or infer_schema(df)
    smap = {c["name"]: c for c in schema}
    cols = analysis_columns(df, schema)
    measures = rank_measures(cols["numeric"], smap)[:k]
    cards = [{"label": "Records", "value": int(len(df)), "aggregation": "count", "format": "count", "trend": None}]
    for c in measures[: k - 1]:
        agg = st.default_agg(c, smap[c]["semantic_type"])
        val = float(df[c].sum()) if agg == "sum" else float(df[c].mean())
        trend = _last_full_period_change(df, cols["datetime"][0], c, agg) if cols["datetime"] else None
        cards.append({
            "label": f"{'Total' if agg == 'sum' else 'Average'} {c}", "column": c, "value": val, "aggregation": agg,
            "format": agg, "n": int(df[c].notna().sum()),
            "trend": round(trend["change_pct"], 1) if trend else None, "trend_detail": trend,
        })
    return to_jsonable(cards)


def generate_dashboard(df: pd.DataFrame, eda: Optional[Dict] = None, schema: Optional[List[Dict]] = None) -> Dict[str, Any]:
    schema = schema or infer_schema(df)
    dom = detect_domain(df)
    charts = recommend_charts(df, schema, max_charts=10)
    return to_jsonable({
        "domain": dom["domain"], "domain_label": dom["domain"].title(), "domain_detail": dom,
        "domain_description": DESCRIPTIONS.get(dom["domain"], "Analytics Dashboard"),
        "kpis": kpi_cards(df, schema), "charts": charts,
    })
