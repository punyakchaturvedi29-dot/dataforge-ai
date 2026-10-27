"""Workspaces, ingestion, data access and the overview dashboard."""
from __future__ import annotations

from typing import List, Optional

import numpy as np
import pandas as pd
from fastapi import APIRouter, File, Form, Query, UploadFile
from pydantic import BaseModel, Field

from agents.dashboard import kpi_cards
from agents.ingest import inspect_file, load_dataframe, read_upload
from agents.schema import is_text_dtype
from core.config import BACKEND_DIR, settings
from core.errors import NotFound, ValidationFailed
from core.jsonsafe import records, to_jsonable
from services import analysis as A
from services.workspace import STAGES, store

from ._deps import csv_response, get_ws, stem, xlsx_response

router = APIRouter()


def _workspace_payload(ws) -> dict:
    out = ws.summary()
    out["datasets"] = [d.summary() for d in ws.datasets()]
    out["stages"] = ws.stages()
    out["settings"] = ws.meta.get("settings", {})
    out["activity"] = list(reversed(ws.meta.get("activity", [])[-30:]))
    try:
        out["active_dataset"] = A.dataset_card(ws)
    except NotFound:
        out["active_dataset"] = None
    return to_jsonable(out)


def _ingest(content: bytes, filename: str, sheet: Optional[str], name: Optional[str]):
    df, report = load_dataframe(content, filename, sheet)
    ws = store.create(name=name or report["filename"])
    ds = ws.add_dataset(df, filename=report["filename"], ingestion=report, make_active=True)
    ws.log("uploaded", f"Uploaded '{report['filename']}' ({len(df):,} rows × {len(df.columns)} columns)",
           dataset_id=ds.id, warnings=len(report["warnings"]))
    ws.mark_stage("upload", save=False)
    ws.mark_stage("validate", save=False)
    ws.save()
    return ws


@router.post("/inspect", summary="Pre-check a file (format, Excel sheets, delimiter) before ingesting it")
async def inspect(file: UploadFile = File(...)):
    content = await read_upload(file)
    return inspect_file(content, file.filename or "")


@router.post("/upload", status_code=201, summary="Upload a dataset and create a workspace")
async def upload(file: UploadFile = File(...), sheet: Optional[str] = Form(None), name: Optional[str] = Form(None)):
    content = await read_upload(file)
    ws = _ingest(content, file.filename or "", sheet or None, name)
    return _workspace_payload(ws)


@router.post("/sample", status_code=201, summary="Create a workspace from the bundled sample dataset")
def sample():
    path = BACKEND_DIR.parent / "sample_data.csv"
    if not path.exists():
        raise NotFound("The sample dataset is not available on this server.")
    ws = _ingest(path.read_bytes(), "sample_data.csv", None, "Sample: retail orders")
    return _workspace_payload(ws)


@router.get("", summary="List workspaces (most recent first)")
def list_workspaces(limit: int = Query(50, ge=1, le=200)):
    return {"workspaces": store.list(limit)}


@router.get("/{workspace_id}")
def get_workspace(workspace_id: str):
    return _workspace_payload(get_ws(workspace_id))


class WorkspacePatch(BaseModel):
    name: Optional[str] = Field(None, max_length=120)
    target: Optional[str] = Field(None, max_length=300)
    clear_target: bool = False


@router.patch("/{workspace_id}")
def patch_workspace(workspace_id: str, body: WorkspacePatch):
    ws = get_ws(workspace_id)
    with ws.lock:
        if body.name is not None:
            if not body.name.strip():
                raise ValidationFailed("Name cannot be empty")
            ws.meta["name"] = body.name.strip()
        s = ws.meta.setdefault("settings", {})
        if body.clear_target:
            s.pop("target", None)
        elif body.target is not None:
            if body.target not in ws.active.working.columns:
                raise ValidationFailed(f"Unknown column '{body.target}'")
            s["target"] = body.target
        ws.save()
    return _workspace_payload(ws)


@router.delete("/{workspace_id}", status_code=204, summary="Permanently delete a workspace and all its files")
def delete_workspace(workspace_id: str):
    store.delete(workspace_id)
    from services import models as registry
    registry.clear_cache()


@router.post("/{workspace_id}/stages/{stage}")
def mark_stage(workspace_id: str, stage: str):
    if stage not in STAGES:
        raise ValidationFailed(f"Unknown stage '{stage}'")
    ws = get_ws(workspace_id)
    ws.mark_stage(stage)
    return {"stages": ws.stages()}


@router.get("/{workspace_id}/activity")
def activity(workspace_id: str, limit: int = Query(100, ge=1, le=300)):
    ws = get_ws(workspace_id)
    return {"activity": list(reversed(ws.meta.get("activity", [])))[:limit]}


# ── Datasets ──────────────────────────────────────────────────────────────────

@router.get("/{workspace_id}/datasets")
def datasets(workspace_id: str):
    ws = get_ws(workspace_id)
    return {"active_dataset_id": ws.meta.get("active_dataset_id"), "datasets": [d.summary() for d in ws.datasets()]}


@router.post("/{workspace_id}/datasets/{dataset_id}/activate")
def activate(workspace_id: str, dataset_id: str):
    ws = get_ws(workspace_id)
    ws.set_active(dataset_id)
    return _workspace_payload(ws)


@router.delete("/{workspace_id}/datasets/{dataset_id}", status_code=204)
def remove_dataset(workspace_id: str, dataset_id: str):
    ws = get_ws(workspace_id)
    name = ws.dataset(dataset_id).name
    ws.remove_dataset(dataset_id)
    ws.log("dataset_removed", f"Removed dataset '{name}'")
    ws.save()


@router.get("/{workspace_id}/schema")
def get_schema(workspace_id: str, dataset_id: Optional[str] = None):
    ws = get_ws(workspace_id)
    ds = ws.dataset(dataset_id)
    return {"dataset_id": ds.id, "version": ds.version, "columns": A.schema(ws, ds)}


@router.get("/{workspace_id}/data", summary="Paginated, sortable, searchable rows (server-side)")
def data(workspace_id: str, dataset_id: Optional[str] = None, version: str = Query("working", pattern="^(working|original)$"),
         page: int = Query(1, ge=1), page_size: int = Query(50, ge=1, le=500), sort: Optional[str] = None,
         desc: bool = False, search: Optional[str] = Query(None, max_length=200), filter_column: Optional[str] = None,
         filter_value: Optional[str] = Query(None, max_length=200), missing_only: bool = False,
         duplicates_only: bool = False):
    ws = get_ws(workspace_id)
    ds = ws.dataset(dataset_id)
    df = ds.working if version == "working" else ds.original
    view = df
    if duplicates_only:
        view = view[view.duplicated(keep=False)]
    if missing_only:
        view = view[view.isna().any(axis=1)]
    if filter_column:
        if filter_column not in df.columns:
            raise ValidationFailed(f"Unknown column '{filter_column}'")
        s = view[filter_column]
        if filter_value is None or filter_value == "":
            view = view[s.isna()]
        else:
            view = view[s.astype("string").str.lower() == filter_value.lower()]
    if search:
        cols = [c for c in view.columns if is_text_dtype(view[c])][:40] or list(view.columns[:40])
        mask = np.zeros(len(view), dtype=bool)
        for c in cols:
            mask |= view[c].astype("string").str.contains(search, case=False, regex=False, na=False).to_numpy()
        view = view[mask]
    if sort:
        if sort not in df.columns:
            raise ValidationFailed(f"Unknown column '{sort}'")
        view = view.sort_values(sort, ascending=not desc, na_position="last", kind="stable")
    total = len(view)
    start = (page - 1) * page_size
    chunk = view.iloc[start:start + page_size]
    return to_jsonable({
        "dataset_id": ds.id, "version": version, "dataset_version": ds.version, "total_rows": int(len(df)),
        "matched_rows": int(total), "page": page, "page_size": page_size,
        "pages": max(1, -(-total // page_size)), "columns": [{"name": str(c), "dtype": str(df[c].dtype)} for c in df.columns],
        "row_ids": [to_jsonable(i) for i in chunk.index], "rows": records(chunk),
    })


@router.get("/{workspace_id}/download", summary="Download the working (cleaned) or original dataset")
def download(workspace_id: str, dataset_id: Optional[str] = None,
             version: str = Query("working", pattern="^(working|original)$"),
             format: str = Query("csv", pattern="^(csv|xlsx)$")):
    ws = get_ws(workspace_id)
    ds = ws.dataset(dataset_id)
    df = ds.working if version == "working" else ds.original
    name = f"{stem(ds.meta.get('filename'))}_{'cleaned' if version == 'working' and ds.ops else version}.{format}"
    ws.log("exported", f"Downloaded {version} data as {format.upper()}")
    ws.mark_stage("export")
    return csv_response(df, name) if format == "csv" else xlsx_response(df, name)


# ── Overview dashboard ───────────────────────────────────────────────────────────

@router.get("/{workspace_id}/overview", summary="Dashboard metrics computed from the current dataset")
def overview(workspace_id: str):
    ws = get_ws(workspace_id)
    ds = ws.active
    prof = A.profile(ws, ds)
    sch = A.schema(ws, ds)
    iss = A.issues(ws, ds)
    model = A.latest_model(ws)
    fc = A.load_analysis(ws, "forecast")
    an = A.load_analysis(ws, "anomaly")
    dg = A.load_analysis(ws, "diagnostics")
    indicators = {}
    if fc:
        r = fc["result"]
        f0 = r["forecasts"][r["selected_method"]]
        indicators["forecast"] = {"target": r.get("target"), "aggregation": r["aggregation"], "method": r["selected_label"],
                                  "next_period": r["future_index"][0], "next_value": f0["forecast"][0],
                                  "lower80": f0["lower80"][0], "upper80": f0["upper80"][0], "stale": fc["stale"],
                                  "at": fc["at"]}
    if an:
        indicators["anomalies"] = {"n": an["result"]["n_anomalies"], "rate_pct": an["result"]["anomaly_rate_pct"],
                                   "stale": an["stale"], "at": an["at"]}
    if dg:
        f0 = (dg["result"].get("findings") or [None])[0]
        indicators["diagnostics"] = {"outcome": dg["result"]["outcome"], "top": f0["title"] if f0 else None,
                                     "label": f0["label"] if f0 else None, "stale": dg["stale"], "at": dg["at"]}
    from services import models as registry
    return to_jsonable({
        "dataset": A.dataset_card(ws, ds), "shape": prof["shape"], "quality": prof["quality"],
        "missing": {"cells": prof["missing"]["cells"], "pct": prof["missing"]["pct"],
                    "by_column": prof["missing"]["by_column"][:8], "complete_rows": prof["missing"]["complete_rows"]},
        "duplicates": prof["duplicates"], "type_counts": prof["type_counts"],
        "issues": {"counts": prof["issues_count"], "top": [i for i in iss if i["severity"] in ("high", "medium")][:5]},
        "kpis": kpi_cards(ds.working, sch), "findings": A.findings(ws, ds)[:4],
        "model": registry.summary(model) if model else None, "indicators": indicators,
        "recommendations": A.recs(ws)[:4], "stages": ws.stages(),
        "activity": list(reversed(ws.meta.get("activity", [])[-8:])),
        "ingestion_warnings": ds.meta.get("ingestion", {}).get("warnings", []),
    })
