"""Multi-file analysis: add datasets, compare schemas, discover and validate joins, stack."""
from __future__ import annotations

from typing import List, Optional

from fastapi import APIRouter, File, Form, Query, UploadFile
from pydantic import BaseModel, Field

from agents.ingest import load_dataframe, read_upload
from agents.multifile import compare_schemas, execute_join, join_candidates, preview_join, stack
from core.errors import Conflict, ValidationFailed
from core.jsonsafe import to_jsonable

from ._deps import get_ws

router = APIRouter()


@router.post("/{workspace_id}/sources", status_code=201, summary="Add another file to the workspace")
async def add_source(workspace_id: str, file: UploadFile = File(...), sheet: Optional[str] = Form(None),
                     activate: bool = Form(False)):
    ws = get_ws(workspace_id)
    content = await read_upload(file)
    df, report = load_dataframe(content, file.filename or "", sheet or None)
    with ws.lock:
        ds = ws.add_dataset(df, filename=report["filename"], ingestion=report, make_active=activate)
        ws.log("source_added", f"Added '{report['filename']}' ({len(df):,} rows)", dataset_id=ds.id)
        ws.save()
    return {"dataset": ds.summary(), "ingestion": report}


@router.get("/{workspace_id}/sources/compare")
def compare(workspace_id: str, ids: Optional[str] = Query(None, description="Comma-separated dataset ids")):
    ws = get_ws(workspace_id)
    dss = [ws.dataset(i) for i in ids.split(",")] if ids else ws.datasets()
    return compare_schemas([(d.id, d.name, d.working) for d in dss])


@router.get("/{workspace_id}/sources/join-candidates")
def candidates(workspace_id: str, left: str, right: str):
    ws = get_ws(workspace_id)
    a, b = ws.dataset(left), ws.dataset(right)
    if a.id == b.id:
        raise ValidationFailed("Choose two different datasets.")
    return {"left": a.summary(), "right": b.summary(),
            "candidates": ws.cached("join", {"l": a.id, "lv": a.version, "r": b.id, "rv": b.version},
                                    lambda: join_candidates(a.working, b.working), a)}


class JoinBody(BaseModel):
    left_id: str
    right_id: str
    left_on: str = Field(..., max_length=300)
    right_on: str = Field(..., max_length=300)
    how: str = "left"
    case_insensitive: bool = False
    name: Optional[str] = Field(None, max_length=120)
    confirm: bool = False
    activate: bool = True


@router.post("/{workspace_id}/sources/join/preview")
def join_preview(workspace_id: str, body: JoinBody):
    ws = get_ws(workspace_id)
    a, b = ws.dataset(body.left_id), ws.dataset(body.right_id)
    return preview_join(a.working, b.working, body.left_on, body.right_on, body.how, body.case_insensitive)


@router.post("/{workspace_id}/sources/join", status_code=201)
def join(workspace_id: str, body: JoinBody):
    ws = get_ws(workspace_id)
    a, b = ws.dataset(body.left_id), ws.dataset(body.right_id)
    pv = preview_join(a.working, b.working, body.left_on, body.right_on, body.how, body.case_insensitive, sample=0)
    if pv["requires_confirmation"] and not body.confirm:
        raise Conflict("This join needs confirmation: " + " ".join(pv["warnings"]), code="confirmation_required",
                       details={"validation": {k: pv[k] for k in pv if k not in ("preview", "preview_columns")}})
    merged = execute_join(a.working, b.working, body.left_on, body.right_on, body.how, body.case_insensitive)
    lineage = {"type": "join", "how": body.how, "left_on": body.left_on, "right_on": body.right_on,
               "case_insensitive": body.case_insensitive,
               "sources": [{"dataset_id": a.id, "name": a.name, "version": a.version, "rows": int(len(a.working))},
                           {"dataset_id": b.id, "name": b.name, "version": b.version, "rows": int(len(b.working))}],
               "validation": {k: pv[k] for k in ("rows_result", "cardinality", "match_rate_left", "unmatched_left_rows",
                                                 "unmatched_right_rows", "explosion_factor", "warnings")},
               "confirmed": body.confirm}
    name = body.name or f"{a.name} ⋈ {b.name}"
    with ws.lock:
        ds = ws.add_dataset(merged, filename=name, name=name, source="join", lineage=lineage,
                            ingestion={"filename": name, "format": "join", "warnings": pv["warnings"]},
                            make_active=body.activate)
        ws.log("joined", f"Joined '{a.name}' and '{b.name}' on {body.left_on} = {body.right_on} ({body.how}): "
                         f"{len(merged):,} rows", dataset_id=ds.id)
        ws.invalidate_cache()
        ws.save()
    return to_jsonable({"dataset": ds.summary(), "validation": lineage["validation"]})


class StackBody(BaseModel):
    ids: List[str] = Field(..., min_length=2, max_length=20)
    mode: str = "union"
    name: Optional[str] = Field(None, max_length=120)
    activate: bool = True


@router.post("/{workspace_id}/sources/stack", status_code=201)
def stack_sources(workspace_id: str, body: StackBody):
    ws = get_ws(workspace_id)
    dss = [ws.dataset(i) for i in body.ids]
    out, report = stack([(d.name, d.working) for d in dss], body.mode)
    name = body.name or "Stacked: " + " + ".join(d.name for d in dss)[:100]
    lineage = {"type": "stack", "mode": body.mode, "sources": [{"dataset_id": d.id, "name": d.name, "version": d.version,
                                                                "rows": int(len(d.working))} for d in dss], "report": report}
    with ws.lock:
        ds = ws.add_dataset(out, filename=name, name=name, source="stack", lineage=lineage,
                            ingestion={"filename": name, "format": "stack", "warnings": report["warnings"]},
                            make_active=body.activate)
        ws.log("stacked", f"Stacked {len(dss)} datasets into '{name}' ({len(out):,} rows)", dataset_id=ds.id)
        ws.save()
    return {"dataset": ds.summary(), "report": report}
