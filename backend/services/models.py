"""
Model registry with a secure, explicit lifecycle.

* Only models trained by this server are ever loaded. There is deliberately no
  endpoint to upload a serialized model: unpickling untrusted files can execute
  arbitrary code.
* Each artifact is signed with HMAC-SHA256 using the server secret
  (``DATAFORGE_SECRET_KEY`` or an auto-generated key in the data directory).
  The signature is verified before every load, so a tampered or replaced file
  is refused.
* Metadata (metrics, input schema, library versions, data fingerprint,
  cleaning steps) is stored next to the artifact for reproducibility.
"""
from __future__ import annotations

import io
import json
import logging
import shutil
import threading
import time
from typing import Any, Dict, List, Optional, Tuple

import joblib

from core.config import settings
from core.errors import NotFound, Conflict
from core.jsonsafe import to_jsonable
from core.security import new_id, normalize_id, safe_child, sha256_bytes, sign_bytes, verify_signature

from .workspace import Workspace, _atomic_write_json

log = logging.getLogger("dataforge.models")
_load_cache: Dict[Tuple[str, str], Any] = {}
_lock = threading.Lock()


def _dir(ws: Workspace, model_id: str):
    return safe_child(ws.dir, "models", normalize_id(model_id, "model"))


def save_model(ws: Workspace, pipeline: Any, meta: Dict[str, Any]) -> str:
    mid = new_id(16)
    d = _dir(ws, mid)
    d.mkdir(parents=True, exist_ok=True)
    buf = io.BytesIO()
    joblib.dump(pipeline, buf, compress=3)
    data = buf.getvalue()
    (d / "model.joblib").write_bytes(data)
    (d / "model.sig").write_text(sign_bytes(data, settings.get_secret()))
    meta = dict(meta)
    meta.update({"id": mid, "artifact_sha256": sha256_bytes(data), "artifact_bytes": len(data),
                 "created_at": meta.get("created_at", time.time())})
    _atomic_write_json(d / "meta.json", meta)
    with ws.lock:
        ws.meta.setdefault("models", []).append(mid)
        ws.save()
    return mid


def get_meta(ws: Workspace, model_id: str) -> Dict[str, Any]:
    p = _dir(ws, model_id) / "meta.json"
    if not p.exists():
        raise NotFound("Model not found")
    return json.loads(p.read_text(encoding="utf-8"))


def list_models(ws: Workspace) -> List[Dict[str, Any]]:
    out = []
    for mid in ws.meta.get("models", []):
        try:
            m = get_meta(ws, mid)
        except NotFound:
            continue
        out.append(summary(m))
    out.sort(key=lambda m: -(m.get("created_at") or 0))
    return out


def summary(m: Dict[str, Any]) -> Dict[str, Any]:
    r = m.get("result", {})
    return to_jsonable({
        "id": m["id"], "name": m.get("name"), "created_at": m.get("created_at"), "target": r.get("target"),
        "problem_type": r.get("problem_type"), "model": r.get("selected", {}).get("model"),
        "primary_metric": r.get("primary_metric"), "primary_metric_label": r.get("primary_metric_label"),
        "test_metrics": r.get("test_metrics"), "dataset_id": m.get("dataset_id"), "dataset_name": m.get("dataset_name"),
        "experiment_id": m.get("experiment_id"), "n_features": len(r.get("input_schema", {}).get("features", [])),
        "versions": r.get("versions"),
    })


def load_model(ws: Workspace, model_id: str) -> Tuple[Any, Dict[str, Any], List[str]]:
    mid = normalize_id(model_id, "model")
    meta = get_meta(ws, mid)
    key = (ws.id, mid)
    warnings: List[str] = []
    import sklearn
    trained = meta.get("result", {}).get("versions", {}).get("scikit-learn")
    if trained and trained != sklearn.__version__:
        warnings.append(f"This model was trained with scikit-learn {trained}; the server runs {sklearn.__version__}. "
                        "Retrain the model if predictions look wrong.")
    with _lock:
        if key in _load_cache:
            return _load_cache[key], meta, warnings
    d = _dir(ws, mid)
    data = (d / "model.joblib").read_bytes()
    sig = (d / "model.sig").read_text() if (d / "model.sig").exists() else ""
    if not verify_signature(data, sig, settings.get_secret()):
        log.error("Refusing to load model %s: signature mismatch", mid)
        raise Conflict("The model file failed its integrity check and was not loaded. Retrain the model.",
                       code="integrity_check_failed")
    model = joblib.load(io.BytesIO(data))
    with _lock:
        if len(_load_cache) > 16:
            _load_cache.pop(next(iter(_load_cache)))
        _load_cache[key] = model
    return model, meta, warnings


def delete_model(ws: Workspace, model_id: str) -> None:
    mid = normalize_id(model_id, "model")
    d = _dir(ws, mid)
    if not d.exists():
        raise NotFound("Model not found")
    shutil.rmtree(d, ignore_errors=True)
    with _lock:
        _load_cache.pop((ws.id, mid), None)
    with ws.lock:
        ws.meta["models"] = [m for m in ws.meta.get("models", []) if m != mid]
        ws.save()


def clear_cache() -> None:
    with _lock:
        _load_cache.clear()
