"""v4-compatible upload endpoint (``POST /api/upload/``).

Behaviour change from v4: data is no longer cleaned silently on upload. The
response lists cleaning *suggestions*; they are applied in the cleaning workflow.
"""
from fastapi import APIRouter, File, UploadFile

from agents.ingest import read_upload
from core.jsonsafe import records, to_jsonable
from services import analysis as A

from .workspaces import _ingest

router = APIRouter()


@router.post("/")
async def upload_file(file: UploadFile = File(...)):
    content = await read_upload(file)
    ws = _ingest(content, file.filename or "", None, None)
    ds = ws.active
    df = ds.working
    prof = A.profile(ws, ds)
    cols = []
    for c in prof["columns"]:
        info = {"name": c["name"], "type": c["dtype"], "logical_type": c["logical_type"],
                "semantic_type": c["semantic_type"], "missing": c["n_missing"], "missing_pct": c["missing_pct"],
                "unique": c["n_unique"]}
        if c.get("numeric"):
            info.update({"min": c["numeric"].get("min"), "max": c["numeric"].get("max"), "mean": c["numeric"].get("mean")})
        cols.append(info)
    return to_jsonable({
        "session_id": ws.id, "workspace_id": ws.id, "filename": ds.meta.get("filename"), "rows": len(df),
        "original_rows": len(ds.original), "columns": cols, "preview": records(df.head(10)),
        "cleaning_report": [], "cleaning_suggestions": A.issues(ws, ds),
        "ingestion": ds.meta.get("ingestion"), "health_score": round(prof["quality"]["score"]),
    })
