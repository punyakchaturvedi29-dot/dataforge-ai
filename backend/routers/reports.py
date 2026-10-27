"""Executive summary, recommendations and report exports (PDF, Markdown, JSON)."""
from __future__ import annotations

from typing import Any, Dict, List, Optional

from fastapi import APIRouter
from fastapi.responses import JSONResponse, Response
from pydantic import BaseModel, Field

from agents import report_builder as RB
from agents.automl import library_versions
from agents.charts import build_chart
from core.config import settings
from core.errors import ValidationFailed
from core.security import content_disposition
from services import analysis as A
from services import models as registry

from ._deps import get_ws, stem

router = APIRouter()


@router.get("/{workspace_id}/summary")
def summary(workspace_id: str):
    ws = get_ws(workspace_id)
    return A.executive_summary(ws)


@router.get("/{workspace_id}/recommendations")
def recommendations(workspace_id: str):
    return {"recommendations": A.recs(get_ws(workspace_id))}


class ReportBody(BaseModel):
    title: Optional[str] = Field(None, max_length=160)
    format: str = Field("pdf", pattern="^(pdf|markdown|json)$")
    sections: Optional[List[str]] = None
    finding_ids: Optional[List[str]] = None
    model_id: Optional[str] = None
    chart_specs: Optional[List[Dict[str, Any]]] = Field(None, max_length=12)
    include_stale: bool = False


def _build(ws, body: ReportBody) -> Dict[str, Any]:
    ds = ws.active
    sch = A.schema(ws, ds)
    if body.sections:
        bad = [s for s in body.sections if s not in RB.ALL_SECTIONS]
        if bad:
            raise ValidationFailed(f"Unknown report section(s): {', '.join(bad)}")
    all_f = A.findings(ws, ds, limit=40)
    fnd = [f for f in all_f if f["id"] in set(body.finding_ids)] if body.finding_ids is not None else all_f[:8]
    if body.chart_specs is not None:
        charts = []
        for spec in body.chart_specs:
            try:
                charts.append(build_chart(ds.working, spec, sch))
            except ValidationFailed:
                continue
    else:
        charts = A.recommended_charts(ws, ds)[:4]
    model = registry.get_meta(ws, body.model_id) if body.model_id else A.latest_model(ws)
    fc = A.load_analysis(ws, "forecast", current_only=not body.include_stale)
    dg = A.load_analysis(ws, "diagnostics", current_only=not body.include_stale)
    rl = A.load_analysis(ws, "rules", current_only=not body.include_stale)
    m = ds.meta
    return RB.build_report(
        title=body.title or f"Analysis of {ds.name}",
        dataset={"name": ds.name, "filename": m.get("filename"), "rows": m.get("rows"), "columns": m.get("columns"),
                 "original_rows": m.get("original_rows"), "version": ds.version, "n_ops": len(ds.ops)},
        profile=A.profile(ws, ds), findings=fnd, summary=A.executive_summary(ws), ops=ds.ops,
        ingestion=m.get("ingestion", {}), charts=charts, model=model,
        forecast=fc["result"] if fc else None, diagnostics=dg["result"] if dg else None,
        rules=rl["result"] if rl else None, recs=A.recs(ws), versions=library_versions(),
        sections=body.sections, workspace_id=ws.id, app_version=settings.version)


@router.post("/{workspace_id}/report", summary="Generate a customised report")
def report(workspace_id: str, body: ReportBody):
    ws = get_ws(workspace_id)
    rep = _build(ws, body)
    ws.log("report", f"Generated a {body.format.upper()} report")
    ws.mark_stage("export")
    name = stem(ws.active.meta.get("filename")) + "_report"
    if body.format == "json":
        return JSONResponse(rep, headers={"Content-Disposition": content_disposition(f"{name}.json")})
    if body.format == "markdown":
        return Response(RB.render_markdown(rep), media_type="text/markdown; charset=utf-8",
                        headers={"Content-Disposition": content_disposition(f"{name}.md")})
    return Response(RB.render_pdf(rep), media_type="application/pdf",
                    headers={"Content-Disposition": content_disposition(f"{name}.pdf")})


@router.post("/{workspace_id}/report/preview", summary="Report structure (JSON) for on-screen preview")
def report_preview(workspace_id: str, body: ReportBody):
    return _build(get_ws(workspace_id), body)
