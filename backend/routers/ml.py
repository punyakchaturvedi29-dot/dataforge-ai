"""Machine learning: AutoML jobs, experiments, model registry, predictions, clustering, anomalies."""
from __future__ import annotations

import io
import json
import time
from dataclasses import fields
from typing import Any, Dict, List, Optional

import pandas as pd
from fastapi import APIRouter, File, Form, Query, UploadFile
from pydantic import BaseModel, Field

from agents import automl
from agents.inference import predict, predictions_frame, prepare_input
from agents.ingest import load_dataframe, read_upload
from agents.ml_advanced import run_anomaly_detection, run_clustering
from core.config import settings
from core.errors import NotFound, PayloadTooLarge, ValidationFailed
from core.jsonsafe import records, to_jsonable
from core.security import new_id, normalize_id, safe_child
from services import analysis as A
from services import models as registry
from services.jobs import CancelledByUser, jobs
from services.workspace import _atomic_write_json, store, write_parquet

from ._deps import csv_response, get_ws

router = APIRouter()
jobs_router = APIRouter()


@router.get("/{workspace_id}/ml/options")
def options(workspace_id: str):
    ws = get_ws(workspace_id)
    ds = ws.active
    sch = A.schema(ws, ds)
    targets = []
    for c in sch:
        lt = c["logical_type"]
        if lt in ("numeric", "categorical", "boolean"):
            hint = ("regression" if lt == "numeric" and c["n_unique"] > 10 else
                    "binary" if c["n_unique"] == 2 else "multiclass")
            if lt == "categorical" and c["n_unique"] > 50:
                continue
            targets.append({"column": c["name"], "logical_type": lt, "n_unique": c["n_unique"], "task_hint": hint,
                            "missing_pct": c["missing_pct"]})
    return {"targets": targets, "columns": sch, "available_models": automl.available_models(),
            "metrics": automl.ALLOWED_PRIMARY, "metric_labels": automl.METRIC_LABELS,
            "defaults": {"test_size": 0.2, "cv_folds": 5, "n_iter": 8,
                         "time_budget_s": settings.training_time_budget_s, "max_rows": settings.training_max_rows},
            "datetime_columns": [c["name"] for c in sch if c["logical_type"] == "datetime"],
            "selected_target": A.target(ws)}


class TrainBody(BaseModel):
    target: str = Field(..., max_length=300)
    problem_type: str = "auto"
    features: Optional[List[str]] = None
    exclude: List[str] = Field(default_factory=list)
    test_size: float = Field(0.2, ge=0.05, le=0.5)
    cv_folds: int = Field(5, ge=2, le=10)
    time_column: Optional[str] = None
    models: Optional[List[str]] = None
    tune: bool = True
    n_iter: int = Field(8, ge=1, le=50)
    time_budget_s: Optional[int] = Field(None, ge=5, le=3600)
    primary_metric: Optional[str] = None
    class_weight: str = "auto"
    random_state: int = Field(42, ge=0, le=2**31 - 1)
    quick: bool = False


def _fingerprint(df: pd.DataFrame) -> str:
    import hashlib
    try:
        h = pd.util.hash_pandas_object(df, index=False).to_numpy().tobytes()
    except TypeError:
        h = df.astype(str).to_csv(index=False).encode()
    return hashlib.sha256(h).hexdigest()


def _run_training(job, ws_id: str, ds_id: str, cfg: automl.TrainConfig) -> Dict[str, Any]:
    ws = store.get(ws_id)
    ds = ws.dataset(ds_id)
    df = ds.working  # immutable snapshot: cleaning replaces, never mutates, the working frame
    sch = A.schema(ws, ds)

    def cancelled():
        return job.cancel_requested

    try:
        out = automl.train(df, cfg, progress=job.update, should_cancel=cancelled, schema=sch)
    except automl.Cancelled:
        raise CancelledByUser()
    r = out["result"]
    exp_id = new_id(16)
    meta = {"name": f"{r['selected']['model']} → {r['target']}", "result": r, "dataset_id": ds.id,
            "dataset_name": ds.name, "dataset_version": ds.version, "experiment_id": exp_id,
            "cleaning_ops": [o for o in ds.ops if o.get("applies_to_new_data")],
            "data_fingerprint": _fingerprint(df), "created_at": time.time()}
    mid = registry.save_model(ws, out["pipeline"], meta)
    exp = {"id": exp_id, "created_at": time.time(), "model_id": mid, "dataset_id": ds.id, "dataset_name": ds.name,
           "dataset_version": ds.version, "result": r}
    d = safe_child(ws.dir, "experiments")
    d.mkdir(parents=True, exist_ok=True)
    _atomic_write_json(d / f"{exp_id}.json", exp)
    summary = _exp_summary(exp)
    with ws.lock:
        ws.meta.setdefault("experiments", []).append(summary)
        ws.log("model_trained", f"Trained {r['selected']['model']} for '{r['target']}' "
                                f"({r['primary_metric_label']} CV {_cvfmt(r)})", experiment_id=exp_id, model_id=mid)
        ws.mark_stage("model", save=False)
        ws.save()
    return {"experiment_id": exp_id, "model_id": mid, "summary": summary}


def _cvfmt(r) -> str:
    sel = next((c for c in r["candidates"] if c["key"] == r["selected"]["key"]), None)
    v = sel["cv"][r["primary_metric"]]["mean"] if sel else None
    return f"{v:.4g}" if v is not None else "n/a"


def _exp_summary(exp: Dict) -> Dict[str, Any]:
    r = exp["result"]
    return to_jsonable({"id": exp["id"], "created_at": exp["created_at"], "model_id": exp["model_id"],
                        "dataset_name": exp["dataset_name"], "dataset_version": exp["dataset_version"],
                        "target": r["target"], "problem_type": r["problem_type"], "selected_model": r["selected"]["model"],
                        "primary_metric": r["primary_metric"], "primary_metric_label": r["primary_metric_label"],
                        "cv_score": _cvfmt(r), "test_metrics": r["test_metrics"], "duration_s": r["duration_s"],
                        "n_candidates": len(r["candidates"]), "config": r["config"]})


@router.post("/{workspace_id}/ml/train", status_code=202, summary="Start an AutoML training job")
def train(workspace_id: str, body: TrainBody):
    ws = get_ws(workspace_id)
    ds = ws.active
    cfg = automl.TrainConfig(**{f.name: getattr(body, f.name) for f in fields(automl.TrainConfig) if hasattr(body, f.name)})
    cfg.validate()
    if cfg.target not in ds.working.columns:
        raise ValidationFailed(f"Unknown target column '{cfg.target}'")
    job = jobs.submit("train", ws.id, f"Train models to predict '{cfg.target}'",
                      lambda j: _run_training(j, ws.id, ds.id, cfg))
    with ws.lock:
        ws.meta.setdefault("settings", {})["target"] = cfg.target
        ws.save()
    return {"job_id": job.id, "job": job.public()}


@router.get("/{workspace_id}/experiments")
def experiments(workspace_id: str):
    ws = get_ws(workspace_id)
    return {"experiments": list(reversed(ws.meta.get("experiments", [])))}


def _load_exp(ws, exp_id: str) -> Dict:
    p = safe_child(ws.dir, "experiments", f"{normalize_id(exp_id, 'experiment')}.json")
    if not p.exists():
        raise NotFound("Experiment not found")
    return json.loads(p.read_text(encoding="utf-8"))


@router.get("/{workspace_id}/experiments/{experiment_id}")
def experiment(workspace_id: str, experiment_id: str):
    return _load_exp(get_ws(workspace_id), experiment_id)


@router.get("/{workspace_id}/experiments/{experiment_id}/export")
def experiment_export(workspace_id: str, experiment_id: str, format: str = Query("csv", pattern="^(csv|json)$")):
    exp = _load_exp(get_ws(workspace_id), experiment_id)
    r = exp["result"]
    if format == "json":
        from fastapi.responses import JSONResponse
        from core.security import content_disposition
        return JSONResponse(exp, headers={"Content-Disposition": content_disposition(f"experiment_{experiment_id}.json")})
    rows = []
    for c in r["candidates"]:
        row = {"model": c["model"], "status": c["status"], "reason": c.get("reason"), "cv_seconds": c.get("cv_seconds"),
               "tuned": c.get("tuned", False)}
        for m, v in (c.get("cv") or {}).items():
            row[f"cv_{m}_mean"] = v["mean"]
            row[f"cv_{m}_std"] = v["std"]
        for m, v in (c.get("train") or {}).items():
            row[f"train_{m}"] = v
        row["selected"] = c["key"] == r["selected"]["key"]
        rows.append(row)
    return csv_response(pd.DataFrame(rows), f"model_comparison_{r['target']}.csv")


@router.delete("/{workspace_id}/experiments/{experiment_id}", status_code=204)
def delete_experiment(workspace_id: str, experiment_id: str):
    ws = get_ws(workspace_id)
    exp = _load_exp(ws, experiment_id)
    safe_child(ws.dir, "experiments", f"{exp['id']}.json").unlink(missing_ok=True)
    try:
        registry.delete_model(ws, exp["model_id"])
    except NotFound:
        pass
    with ws.lock:
        ws.meta["experiments"] = [e for e in ws.meta.get("experiments", []) if e["id"] != exp["id"]]
        ws.save()


@router.get("/{workspace_id}/models")
def models(workspace_id: str):
    return {"models": registry.list_models(get_ws(workspace_id))}


@router.get("/{workspace_id}/models/{model_id}")
def model(workspace_id: str, model_id: str):
    ws = get_ws(workspace_id)
    m = registry.get_meta(ws, model_id)
    return to_jsonable({**registry.summary(m), "input_schema": m["result"]["input_schema"],
                        "cleaning_ops": [{"label": o["label"], "params": o["params"]} for o in m.get("cleaning_ops", [])],
                        "result": m["result"], "data_fingerprint": m.get("data_fingerprint"),
                        "artifact_sha256": m.get("artifact_sha256")})


class PredictBody(BaseModel):
    rows: List[Dict[str, Any]] = Field(..., min_length=1, max_length=1000)
    explain: bool = False
    allow_missing_features: bool = False


@router.post("/{workspace_id}/models/{model_id}/predict")
def predict_rows(workspace_id: str, model_id: str, body: PredictBody):
    ws = get_ws(workspace_id)
    pipe, meta, warns = registry.load_model(ws, model_id)
    df_in = pd.DataFrame(body.rows)
    X, report = prepare_input(df_in, meta, strict=True, allow_missing_features=body.allow_missing_features)
    out = predict(pipe, X, meta)
    if body.explain and len(X) <= 20:
        out["explanations"] = [automl.explain_local(pipe, X.iloc[[i]], meta["result"], meta["result"].get("classes") or [])
                               for i in range(len(X))]
    out["validation"] = report
    out["warnings"] = warns + report.get("cleaning_notes", [])
    ws.mark_stage("model")
    return to_jsonable(out)


@router.post("/{workspace_id}/models/{model_id}/predict/batch", summary="Score a CSV/Excel file")
async def predict_batch(workspace_id: str, model_id: str, file: UploadFile = File(...), sheet: Optional[str] = Form(None),
                        allow_missing_features: bool = Form(False)):
    ws = get_ws(workspace_id)
    pipe, meta, warns = registry.load_model(ws, model_id)
    content = await read_upload(file)
    df_in, ing = load_dataframe(content, file.filename or "", sheet)
    if len(df_in) > settings.batch_predict_max_rows:
        raise PayloadTooLarge(f"Batch prediction is limited to {settings.batch_predict_max_rows:,} rows.")
    X, report = prepare_input(df_in, meta, strict=False, allow_missing_features=allow_missing_features)
    out = predict(pipe, X, meta)
    frame = predictions_frame(df_in, out, report, meta["result"]["target"])
    pid = new_id(16)
    d = safe_child(ws.dir, "predictions")
    d.mkdir(parents=True, exist_ok=True)
    write_parquet(frame, d / f"{pid}.parquet")
    summary: Dict[str, Any] = {"id": pid, "model_id": registry.summary(meta)["id"], "filename": ing["filename"],
                               "rows": int(len(frame)), "created_at": time.time(),
                               "rows_with_warnings": int(frame["input_warnings"].notna().sum())}
    if meta["result"]["problem_type"] == "regression":
        p = pd.Series(out["predictions"], dtype=float)
        summary["distribution"] = {"mean": p.mean(), "min": p.min(), "max": p.max(), "median": p.median()}
    else:
        summary["class_counts"] = pd.Series(out["predictions"]).value_counts().to_dict()
    with ws.lock:
        ws.meta.setdefault("predictions", []).append(to_jsonable(summary))
        ws.meta["predictions"] = ws.meta["predictions"][-20:]
        ws.log("batch_predicted", f"Scored {len(frame):,} rows from '{ing['filename']}'", prediction_id=pid)
        ws.save()
    return to_jsonable({"prediction_id": pid, "summary": summary, "validation": {k: v for k, v in report.items()
                                                                                 if k != "row_issues"},
                        "warnings": warns + report.get("cleaning_notes", []) + ing.get("warnings", []),
                        "notes": out["notes"], "columns": list(frame.columns), "preview": records(frame.head(50))})


@router.get("/{workspace_id}/predictions/{prediction_id}/download")
def download_predictions(workspace_id: str, prediction_id: str):
    ws = get_ws(workspace_id)
    p = safe_child(ws.dir, "predictions", f"{normalize_id(prediction_id, 'prediction')}.parquet")
    if not p.exists():
        raise NotFound("Prediction file not found")
    ws.mark_stage("export")
    return csv_response(pd.read_parquet(p), f"predictions_{prediction_id[:8]}.csv")


@router.delete("/{workspace_id}/models/{model_id}", status_code=204)
def delete_model(workspace_id: str, model_id: str):
    ws = get_ws(workspace_id)
    registry.delete_model(ws, model_id)
    ws.log("model_deleted", "Deleted a model")
    ws.save()


class ClusterBody(BaseModel):
    n_clusters: int = Field(3, ge=2, le=12)
    method: str = "kmeans"
    columns: Optional[List[str]] = None
    eps: Optional[float] = Field(None, gt=0)
    min_samples: Optional[int] = Field(None, ge=2, le=1000)


@router.post("/{workspace_id}/ml/cluster")
def cluster(workspace_id: str, body: ClusterBody):
    ws = get_ws(workspace_id)
    ds = ws.active
    r = run_clustering(ds.working, body.n_clusters, body.method, body.columns, body.eps, body.min_samples)
    A.save_analysis(ws, "cluster", body.model_dump(), r, ds)
    ws.log("clustered", f"Clustering ({body.method}) found {r['n_clusters']} cluster(s)")
    ws.save()
    return r


class AnomalyBody(BaseModel):
    contamination: float = Field(0.05, ge=0.001, le=0.5)
    columns: Optional[List[str]] = None


@router.post("/{workspace_id}/ml/anomaly")
def anomaly(workspace_id: str, body: AnomalyBody):
    ws = get_ws(workspace_id)
    ds = ws.active
    r = run_anomaly_detection(ds.working, body.contamination, body.columns)
    A.save_analysis(ws, "anomaly", body.model_dump(), r, ds)
    ws.log("anomalies", f"Anomaly detection flagged {r['n_anomalies']:,} row(s)")
    ws.save()
    return r


@router.get("/{workspace_id}/analyses/{kind}", summary="Latest saved result of an analysis (forecast, diagnostics, ...)")
def latest(workspace_id: str, kind: str):
    if kind not in ("forecast", "diagnostics", "cluster", "anomaly", "rules"):
        raise NotFound("Unknown analysis kind")
    r = A.load_analysis(get_ws(workspace_id), kind)
    if not r:
        raise NotFound("No saved result yet")
    return r


@router.get("/{workspace_id}/jobs")
def ws_jobs(workspace_id: str):
    ws = get_ws(workspace_id)
    return {"jobs": jobs.list(ws.id)}


@jobs_router.get("/{job_id}")
def job(job_id: str):
    return jobs.get(job_id).public()


@jobs_router.post("/{job_id}/cancel")
def cancel(job_id: str):
    return jobs.cancel(job_id).public()
