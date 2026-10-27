"""v4-compatible feature endpoints (``/api/features/*``)."""
from typing import Optional

from fastapi import APIRouter, File, Form, UploadFile
from pydantic import BaseModel

from agents.dashboard import generate_dashboard
from agents.forecasting import run_forecasting
from agents.ingest import load_dataframe, read_upload
from agents.multifile import compare_schemas, execute_join, join_candidates, preview_join
from agents.rca import run_rca
from core.errors import Conflict
from core.jsonsafe import records, to_jsonable
from services import analysis as A
from services.workspace import store

from ._deps import get_ws

router = APIRouter()


@router.get("/executive-summary/{session_id}")
def executive_summary(session_id: str):
    return A.executive_summary(get_ws(session_id))


@router.get("/dashboard/{session_id}")
def auto_dashboard(session_id: str):
    ws = get_ws(session_id)
    return generate_dashboard(ws.active.working, None, A.schema(ws))


class ForecastRequest(BaseModel):
    session_id: str
    horizon: int = 6


@router.post("/forecast")
def forecast(req: ForecastRequest):
    return run_forecasting(get_ws(req.session_id).active.working, {}, horizon=max(1, min(req.horizon, 60)))


@router.get("/forecast/{session_id}")
def forecast_get(session_id: str, horizon: int = 6):
    return run_forecasting(get_ws(session_id).active.working, {}, horizon=max(1, min(horizon, 60)))


class RCARequest(BaseModel):
    session_id: str
    target_col: Optional[str] = None


@router.post("/rca")
def root_cause_analysis(req: RCARequest):
    return run_rca(get_ws(req.session_id).active.working, None, req.target_col)


@router.get("/rca/{session_id}")
def rca_get(session_id: str, target_col: Optional[str] = None):
    return run_rca(get_ws(session_id).active.working, None, target_col)


@router.post("/multifile/upload")
async def multifile_upload(file: UploadFile = File(...), session_id: Optional[str] = Form(None)):
    content = await read_upload(file)
    df, report = load_dataframe(content, file.filename or "")
    ws = store.get(session_id) if session_id else store.create(name=report["filename"])
    with ws.lock:
        ds = ws.add_dataset(df, filename=report["filename"], ingestion=report)
        ws.log("source_added", f"Added '{report['filename']}'")
        ws.save()
    files = [{"file_id": d.id, "filename": d.name, "rows": d.meta["rows"], "cols": d.meta["columns"],
              "columns": [str(c) for c in d.working.columns]} for d in ws.datasets()]
    return {"session_id": ws.id, "file_id": ds.id, "filename": report["filename"], "rows": len(df),
            "cols": len(df.columns), "columns": [str(c) for c in df.columns], "total_files": len(files), "files": files}


@router.get("/multifile/{session_id}")
def multifile_analysis(session_id: str):
    ws = get_ws(session_id)
    dss = ws.datasets()
    if len(dss) < 2:
        return {"available": False, "reason": "Upload at least 2 files to enable multi-file analysis.",
                "file_count": len(dss)}
    opts = []
    for i, a in enumerate(dss):
        for b in dss[i + 1:]:
            c = join_candidates(a.working, b.working)
            if c:
                opts.append({"file_a": a.name, "file_b": b.name, "fid_a": a.id, "fid_b": b.id, "keys": c,
                             "best_key": f"{c[0]['left_on']} ↔ {c[0]['right_on']}"})
    return to_jsonable({"available": True, "file_count": len(dss),
                        "files": [{"file_id": d.id, "filename": d.name, "rows": d.meta["rows"], "cols": d.meta["columns"],
                                   "columns": [str(c) for c in d.working.columns]} for d in dss],
                        "join_options": opts, "schema": compare_schemas([(d.id, d.name, d.working) for d in dss]),
                        "merged_preview": None,
                        "note": "Joins are no longer executed automatically; preview and confirm them explicitly."})


class MergeRequest(BaseModel):
    session_id: str
    fid_a: str
    fid_b: str
    col_a: str
    col_b: Optional[str] = None
    how: str = "left"
    confirm: bool = False


@router.post("/multifile/merge")
def multifile_merge(req: MergeRequest):
    ws = get_ws(req.session_id)
    a, b = ws.dataset(req.fid_a), ws.dataset(req.fid_b)
    col_b = req.col_b or req.col_a
    pv = preview_join(a.working, b.working, req.col_a, col_b, req.how, sample=10)
    if pv["requires_confirmation"] and not req.confirm:
        raise Conflict("This merge needs confirmation: " + " ".join(pv["warnings"]), code="confirmation_required",
                       details={"validation": {k: v for k, v in pv.items() if k != "preview"}})
    merged = execute_join(a.working, b.working, req.col_a, col_b, req.how)
    with ws.lock:
        ds = ws.add_dataset(merged, filename=f"merged_{a.name}_{b.name}", source="join", make_active=True,
                            lineage={"type": "join", "left_on": req.col_a, "right_on": col_b, "how": req.how,
                                     "sources": [{"dataset_id": a.id, "name": a.name}, {"dataset_id": b.id, "name": b.name}]})
        ws.log("joined", f"Merged '{a.name}' and '{b.name}'")
        ws.save()
    return to_jsonable({"merged_session_id": ws.id, "dataset_id": ds.id,
                        "message": f"Merged on '{req.col_a}' ({req.how} join): {len(merged):,} rows × {len(merged.columns)} columns",
                        "rows": len(merged), "cols": len(merged.columns), "columns": [str(c) for c in merged.columns],
                        "preview": records(merged.head(10)), "validation": {k: v for k, v in pv.items() if k != "preview"}})
