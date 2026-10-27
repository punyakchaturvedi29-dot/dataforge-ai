"""Profiling, correlations, insights, charts and the auto dashboard."""
from __future__ import annotations

from typing import Any, Dict, List, Optional

from fastapi import APIRouter, Query
from pydantic import BaseModel, Field

from agents.charts import build_chart
from agents.dashboard import generate_dashboard
from agents.profiling import target_analysis
from core.jsonsafe import to_jsonable
from services import analysis as A

from ._deps import csv_response, get_ws, stem

router = APIRouter()


@router.get("/{workspace_id}/profile")
def profile(workspace_id: str, dataset_id: Optional[str] = None):
    ws = get_ws(workspace_id)
    ds = ws.dataset(dataset_id)
    p = A.profile(ws, ds)
    ws.mark_stage("profile")
    return p


@router.get("/{workspace_id}/profile/export", summary="Column statistics as CSV")
def profile_csv(workspace_id: str):
    import pandas as pd
    ws = get_ws(workspace_id)
    ds = ws.active
    rows = []
    for c in A.profile(ws, ds)["columns"]:
        r = {"column": c["name"], "type": c["logical_type"], "dtype": c["dtype"], "count": c["count"],
             "missing": c["n_missing"], "missing_pct": c["missing_pct"], "unique": c["n_unique"]}
        n = c.get("numeric") or {}
        for k in ("mean", "median", "mode", "std", "variance", "min", "max", "range", "q1", "q3", "iqr", "skewness", "kurtosis"):
            r[k] = n.get(k)
        r["outliers_iqr"] = (n.get("outliers") or {}).get("iqr_mild")
        cat = c.get("categorical") or {}
        r["top_value"] = cat.get("mode")
        r["top_value_pct"] = cat.get("mode_pct")
        rows.append(r)
    return csv_response(pd.DataFrame(rows), f"{stem(ds.meta.get('filename'))}_profile.csv")


@router.get("/{workspace_id}/correlations")
def correlations(workspace_id: str, dataset_id: Optional[str] = None):
    ws = get_ws(workspace_id)
    return A.corr(ws, ws.dataset(dataset_id))


@router.get("/{workspace_id}/target-analysis")
def target(workspace_id: str, target: str = Query(..., max_length=300)):
    ws = get_ws(workspace_id)
    ds = ws.active
    return ws.cached("target", target, lambda: target_analysis(ds.working, target, A.schema(ws, ds)), ds)


@router.get("/{workspace_id}/insights")
def insights(workspace_id: str, target: Optional[str] = None, limit: int = Query(15, ge=1, le=40)):
    ws = get_ws(workspace_id)
    f = A.findings(ws, tgt=target, limit=limit)
    ws.mark_stage("analyze")
    return {"findings": f, "target": target or A.target(ws),
            "notes": ["Findings are ranked by a priority score = evidence strength × practical impact × relevance.",
                      "'association' findings are statistical relationships, not causal effects."]}


@router.get("/{workspace_id}/insights/export")
def insights_csv(workspace_id: str):
    import pandas as pd
    ws = get_ws(workspace_id)
    rows = [{"priority": f["priority"], "title": f["title"], "category": f["category"], "claim_type": f["claim_type"],
             "confidence": f["confidence"], "score": f["score"], "observation": f["observation"],
             "evidence": f["evidence_text"], "why_it_matters": f["why_it_matters"],
             "follow_up": " | ".join(f["follow_up"])} for f in A.findings(ws, limit=40)]
    return csv_response(pd.DataFrame(rows), "findings.csv")


@router.get("/{workspace_id}/charts/recommended")
def charts(workspace_id: str):
    ws = get_ws(workspace_id)
    c = A.recommended_charts(ws)
    ws.mark_stage("visualize")
    return {"charts": c}


class ChartSpec(BaseModel):
    type: str = Field(..., max_length=20)
    x: Optional[str] = None
    y: Optional[str] = None
    agg: Optional[str] = None
    group: Optional[str] = None
    color: Optional[str] = None
    freq: Optional[str] = None
    method: Optional[str] = None
    limit: int = Field(15, ge=2, le=50)
    columns: Optional[List[str]] = None
    title: Optional[str] = Field(None, max_length=200)


@router.post("/{workspace_id}/charts")
def chart(workspace_id: str, spec: ChartSpec):
    ws = get_ws(workspace_id)
    ds = ws.active
    s = spec.model_dump(exclude_none=True)
    out = ws.cached("chart", s, lambda: build_chart(ds.working, s, A.schema(ws, ds)), ds)
    ws.mark_stage("visualize")
    return out


@router.get("/{workspace_id}/dashboard", summary="Domain-aware auto dashboard (KPIs + charts)")
def dashboard(workspace_id: str):
    ws = get_ws(workspace_id)
    ds = ws.active
    out = ws.cached("dashboard", None, lambda: generate_dashboard(ds.working, None, A.schema(ws, ds)), ds)
    ws.mark_stage("visualize")
    return out
