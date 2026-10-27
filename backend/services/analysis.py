"""
Glue between workspaces and the analysis engines.

Derived results (schema, profile, findings, ...) are cached per
``(dataset id, dataset version, kind, params)``. Any cleaning operation bumps
the dataset version, which invalidates every cached result for that dataset.
Results the user explicitly runs (forecasts, diagnostics, anomaly detection)
are persisted to the workspace folder as the "latest" result of their kind.
"""
from __future__ import annotations

import json
import time
from typing import Any, Dict, List, Optional

import pandas as pd

from agents import charts as charts_mod
from agents.cleaner import detect_issues
from agents.executive_summary import build_summary, recommendations
from agents.insights import generate_findings
from agents.profiling import correlations, profile_dataset
from agents.schema import analysis_columns, infer_schema
from core.config import settings
from core.errors import NotFound
from core.jsonsafe import to_jsonable
from core.security import safe_child

from . import models as registry
from .workspace import Dataset, Workspace, _atomic_write_json


def schema(ws: Workspace, ds: Optional[Dataset] = None) -> List[Dict]:
    ds = ds or ws.active
    return ws.cached("schema", None, lambda: infer_schema(ds.working), ds)


def issues(ws: Workspace, ds: Optional[Dataset] = None) -> List[Dict]:
    ds = ds or ws.active
    return ws.cached("issues", None, lambda: detect_issues(ds.working, schema(ws, ds)), ds)


def profile(ws: Workspace, ds: Optional[Dataset] = None) -> Dict:
    ds = ds or ws.active
    return ws.cached("profile", None, lambda: profile_dataset(ds.working, schema(ws, ds), issues(ws, ds)), ds)


def corr(ws: Workspace, ds: Optional[Dataset] = None) -> Dict:
    ds = ds or ws.active
    return ws.cached("correlations", None, lambda: correlations(ds.working, schema(ws, ds)), ds)


def target(ws: Workspace) -> Optional[str]:
    return (ws.meta.get("settings") or {}).get("target")


def findings(ws: Workspace, ds: Optional[Dataset] = None, tgt: Optional[str] = None, limit: int = 15) -> List[Dict]:
    ds = ds or ws.active
    tgt = tgt if tgt is not None else target(ws)
    return ws.cached("findings", {"t": tgt, "l": limit},
                     lambda: generate_findings(ds.working, schema(ws, ds), issues(ws, ds), tgt, limit), ds)


def recommended_charts(ws: Workspace, ds: Optional[Dataset] = None) -> List[Dict]:
    ds = ds or ws.active
    return ws.cached("charts", None, lambda: charts_mod.recommend_charts(ds.working, schema(ws, ds)), ds)


# ── persisted analyses ─────────────────────────────────────────────────────────

def save_analysis(ws: Workspace, kind: str, params: Dict, result: Dict, ds: Optional[Dataset] = None) -> None:
    ds = ds or ws.active
    d = safe_child(ws.dir, "analyses")
    d.mkdir(parents=True, exist_ok=True)
    _atomic_write_json(d / f"{kind}.json", {"params": params, "result": result, "dataset_id": ds.id,
                                            "dataset_version": ds.version, "at": time.time()})
    with ws.lock:
        ws.meta.setdefault("analyses", {})[kind] = {"at": time.time(), "params": to_jsonable(params),
                                                    "dataset_id": ds.id, "dataset_version": ds.version}
        ws.save()


def load_analysis(ws: Workspace, kind: str, current_only: bool = False) -> Optional[Dict]:
    p = safe_child(ws.dir, "analyses", f"{kind}.json")
    if not p.exists():
        return None
    data = json.loads(p.read_text(encoding="utf-8"))
    try:
        ds = ws.active
        data["stale"] = data.get("dataset_id") != ds.id or data.get("dataset_version") != ds.version
    except NotFound:
        data["stale"] = True
    if current_only and data["stale"]:
        return None
    return data


def latest_model(ws: Workspace) -> Optional[Dict]:
    ids = ws.meta.get("models", [])
    for mid in reversed(ids):
        try:
            return registry.get_meta(ws, mid)
        except NotFound:
            continue
    return None


def state(ws: Workspace) -> Dict[str, Any]:
    ds = ws.active
    sch = schema(ws, ds)
    cols = analysis_columns(ds.working, sch)
    st = {s["id"]: s["completed"] for s in ws.stages()}
    return {
        "issues": issues(ws, ds), "n_ops": len(ds.ops), "profile_viewed": st.get("profile"),
        "findings": findings(ws, ds), "has_datetime": bool(cols["datetime"]),
        "has_forecast": bool(load_analysis(ws, "forecast")), "has_diagnostics": bool(load_analysis(ws, "diagnostics")),
        "has_model": bool(ws.meta.get("models")), "has_predictions": bool(ws.meta.get("predictions")),
        "n_rules": len(ws.meta.get("rules", [])), "rows": len(ds.working),
    }


def recs(ws: Workspace) -> List[Dict]:
    return recommendations(state(ws))


def executive_summary(ws: Workspace) -> Dict:
    ds = ws.active
    fc = load_analysis(ws, "forecast", current_only=True)
    dg = load_analysis(ws, "diagnostics", current_only=True)
    return build_summary(ds.name, profile(ws, ds), findings(ws, ds), ds.ops, latest_model(ws),
                         fc["result"] if fc else None, dg["result"] if dg else None, recs(ws))


def dataset_card(ws: Workspace, ds: Optional[Dataset] = None) -> Dict[str, Any]:
    ds = ds or ws.active
    return {**ds.summary(), "ingestion": ds.meta.get("ingestion", {})}
