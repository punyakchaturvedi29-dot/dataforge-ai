"""
Workspace persistence.

A *workspace* groups everything about one analysis: one or more datasets, the
cleaning operation log, data-quality rules, experiments, trained models,
saved analysis results, chat history and an activity log.

Storage layout (``DATA_DIR/workspaces/<workspace_id>/``)::

    workspace.json                 metadata (atomic writes)
    datasets/<id>/original.parquet the data exactly as ingested (never modified)
    datasets/<id>/working.parquet  original + replayed cleaning operations (cache)
    models/<id>/...                signed model artifacts (see services/models.py)

Session state vs durable data
-----------------------------
* Durable: everything above. It survives restarts until the workspace is deleted
  or expires (``WORKSPACE_TTL_DAYS`` after its last update).
* Session-only: background job progress and in-memory caches.
"""
from __future__ import annotations

import json
import logging
import os
import shutil
import threading
import time
from collections import OrderedDict
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

import pandas as pd

from core.config import settings
from core.errors import NotFound, ResourceLimit, ValidationFailed, Forbidden
from core.jsonsafe import to_jsonable
from core.security import new_id, normalize_id, safe_child

log = logging.getLogger("dataforge.workspace")

STAGES = ["upload", "validate", "profile", "clean", "analyze", "model", "visualize", "export"]
MAX_ACTIVITY = 300
MAX_CHAT = 60


def _atomic_write_json(path: Path, data: Any) -> None:
    tmp = path.with_suffix(path.suffix + f".tmp{os.getpid()}{threading.get_ident()}")
    with open(tmp, "w", encoding="utf-8") as fh:
        json.dump(to_jsonable(data), fh, ensure_ascii=False, indent=1)
    os.replace(tmp, path)


def write_parquet(df: pd.DataFrame, path: Path) -> None:
    """Persist a dataframe; object columns that Arrow cannot type are stored as text."""
    tmp = path.with_suffix(".tmp.parquet")
    try:
        df.to_parquet(tmp, index=False)
    except Exception:
        fixed = df.copy()
        for c in fixed.columns:
            if pd.api.types.is_object_dtype(fixed[c]):
                fixed[c] = fixed[c].map(lambda v: v if v is None or (isinstance(v, float) and pd.isna(v)) else str(v))
        fixed.to_parquet(tmp, index=False)
    os.replace(tmp, path)


class Dataset:
    """One table inside a workspace."""

    def __init__(self, ws: "Workspace", meta: Dict[str, Any]):
        self.ws = ws
        self.meta = meta
        self._original: Optional[pd.DataFrame] = None
        self._working: Optional[pd.DataFrame] = None

    # -- identity -------------------------------------------------------------
    @property
    def id(self) -> str:
        return self.meta["id"]

    @property
    def name(self) -> str:
        return self.meta.get("name") or self.meta.get("filename") or self.id

    @property
    def ops(self) -> List[Dict[str, Any]]:
        return self.meta.setdefault("ops", [])

    @property
    def version(self) -> int:
        return int(self.meta.get("version", 0))

    @property
    def dir(self) -> Path:
        return safe_child(self.ws.dir, "datasets", self.id)

    # -- data -----------------------------------------------------------------
    @property
    def original(self) -> pd.DataFrame:
        if self._original is None:
            p = self.dir / "original.parquet"
            if not p.exists():
                raise NotFound("Dataset file is missing on disk")
            self._original = pd.read_parquet(p)
        return self._original

    @property
    def working(self) -> pd.DataFrame:
        if self._working is None:
            p = self.dir / "working.parquet"
            if p.exists() and self.meta.get("working_version") == self.version:
                self._working = pd.read_parquet(p)
            else:
                self._working = self._replay()
                self._persist_working()
        return self._working

    def _replay(self) -> pd.DataFrame:
        from agents.cleaner import replay_ops
        return replay_ops(self.original, self.ops)

    def _persist_working(self) -> None:
        write_parquet(self._working, self.dir / "working.parquet")
        self.meta["working_version"] = self.version

    def set_working(self, df: pd.DataFrame, ops: List[Dict[str, Any]]) -> None:
        """Replace the operation log and working data (called by the cleaning service)."""
        self.meta["ops"] = ops
        self.meta["version"] = self.version + 1
        self._working = df
        self._persist_working()
        self.meta["rows"] = int(len(df))
        self.meta["columns"] = int(len(df.columns))
        self.ws.invalidate_cache(self.id)

    def summary(self) -> Dict[str, Any]:
        m = self.meta
        return {
            "id": self.id, "name": self.name, "filename": m.get("filename"), "source": m.get("source"),
            "rows": m.get("rows"), "columns": m.get("columns"),
            "original_rows": m.get("original_rows"), "original_columns": m.get("original_columns"),
            "version": self.version, "n_ops": len(self.ops), "created_at": m.get("created_at"),
            "lineage": m.get("lineage"),
        }


class Workspace:
    def __init__(self, wid: str, meta: Dict[str, Any], base: Path):
        self.id = wid
        self.meta = meta
        self.dir = safe_child(base, wid)
        self.lock = threading.RLock()
        self._datasets: Dict[str, Dataset] = {}
        self._cache: Dict[tuple, Any] = {}

    # -- datasets -------------------------------------------------------------
    def dataset(self, dataset_id: Optional[str] = None) -> Dataset:
        did = dataset_id or self.meta.get("active_dataset_id")
        if not did:
            raise NotFound("This workspace has no dataset yet")
        did = normalize_id(did, "dataset")
        if did not in self.meta.get("datasets", {}):
            raise NotFound("Dataset not found in this workspace")
        if did not in self._datasets:
            self._datasets[did] = Dataset(self, self.meta["datasets"][did])
        return self._datasets[did]

    @property
    def active(self) -> Dataset:
        return self.dataset()

    def datasets(self) -> List[Dataset]:
        return [self.dataset(d) for d in self.meta.get("datasets", {})]

    def add_dataset(self, df: pd.DataFrame, *, filename: str, name: Optional[str] = None, source: str = "upload",
                    ingestion: Optional[Dict] = None, lineage: Optional[Dict] = None, make_active: bool = False) -> Dataset:
        if len(self.meta.get("datasets", {})) >= 20:
            raise ResourceLimit("A workspace can hold at most 20 datasets. Delete one first.")
        did = new_id(16)
        meta = {
            "id": did, "filename": filename, "name": name or filename, "source": source,
            "ingestion": ingestion or {}, "lineage": lineage or {}, "ops": [], "version": 0,
            "rows": int(len(df)), "columns": int(len(df.columns)),
            "original_rows": int(len(df)), "original_columns": int(len(df.columns)),
            "created_at": time.time(),
        }
        ds_dir = safe_child(self.dir, "datasets", did)
        ds_dir.mkdir(parents=True, exist_ok=True)
        write_parquet(df, ds_dir / "original.parquet")
        self.meta.setdefault("datasets", {})[did] = meta
        ds = Dataset(self, meta)
        ds._original = pd.read_parquet(ds_dir / "original.parquet")  # identical dtypes to later loads
        self._datasets[did] = ds
        if make_active or not self.meta.get("active_dataset_id"):
            self.meta["active_dataset_id"] = did
        self.save()
        return ds

    def remove_dataset(self, dataset_id: str) -> None:
        did = normalize_id(dataset_id, "dataset")
        if did == self.meta.get("active_dataset_id"):
            raise ValidationFailed("The active dataset cannot be deleted. Activate another dataset first.")
        if did not in self.meta.get("datasets", {}):
            raise NotFound("Dataset not found")
        self.meta["datasets"].pop(did)
        self._datasets.pop(did, None)
        shutil.rmtree(safe_child(self.dir, "datasets", did), ignore_errors=True)
        self.save()

    def set_active(self, dataset_id: str) -> Dataset:
        ds = self.dataset(dataset_id)
        self.meta["active_dataset_id"] = ds.id
        self.log("dataset_activated", f"Active dataset set to '{ds.name}'")
        self.save()
        return ds

    # -- caching ----------------------------------------------------------------
    def cached(self, kind: str, params: Any, fn: Callable[[], Any], dataset: Optional[Dataset] = None) -> Any:
        ds = dataset or self.active
        key = (ds.id, ds.version, kind, json.dumps(params, sort_keys=True, default=str))
        if key in self._cache:
            return self._cache[key]
        val = fn()
        if len(self._cache) > 64:
            self._cache.pop(next(iter(self._cache)))
        self._cache[key] = val
        return val

    def invalidate_cache(self, dataset_id: Optional[str] = None) -> None:
        if dataset_id is None:
            self._cache.clear()
        else:
            for k in [k for k in self._cache if k[0] == dataset_id]:
                self._cache.pop(k, None)

    # -- activity / stages -----------------------------------------------------
    def log(self, kind: str, message: str, **details: Any) -> None:
        act = self.meta.setdefault("activity", [])
        act.append({"ts": time.time(), "type": kind, "message": message, "details": to_jsonable(details)})
        if len(act) > MAX_ACTIVITY:
            del act[: len(act) - MAX_ACTIVITY]

    def mark_stage(self, stage: str, save: bool = True) -> None:
        if stage not in STAGES:
            return
        st = self.meta.setdefault("stages", {})
        if not st.get(stage, {}).get("completed"):
            st[stage] = {"completed": True, "at": time.time()}
            if save:
                self.save()

    def stages(self) -> List[Dict[str, Any]]:
        st = self.meta.get("stages", {})
        return [{"id": s, "completed": bool(st.get(s, {}).get("completed")), "at": st.get(s, {}).get("at")} for s in STAGES]

    # -- generic sections --------------------------------------------------------
    def section(self, name: str, default: Any) -> Any:
        return self.meta.setdefault(name, default)

    def save(self) -> None:
        self.meta["updated_at"] = time.time()
        self.dir.mkdir(parents=True, exist_ok=True)
        _atomic_write_json(self.dir / "workspace.json", self.meta)

    def summary(self) -> Dict[str, Any]:
        active = None
        try:
            active = self.active.summary() if self.meta.get("active_dataset_id") else None
        except NotFound:
            active = None
        return {
            "id": self.id, "name": self.meta.get("name"), "created_at": self.meta.get("created_at"),
            "updated_at": self.meta.get("updated_at"), "active_dataset": active,
            "n_datasets": len(self.meta.get("datasets", {})),
            "n_models": len(self.meta.get("models", [])),
            "expires_at": (self.meta.get("updated_at") or time.time()) + settings.workspace_ttl_days * 86400,
        }


class WorkspaceStore:
    def __init__(self) -> None:
        self._lock = threading.RLock()
        self._cache: "OrderedDict[str, Workspace]" = OrderedDict()
        self._last_purge = 0.0

    @property
    def base(self) -> Path:
        settings.ensure_dirs()
        return (settings.data_dir / "workspaces").resolve()

    def create(self, name: str = "Untitled workspace", wid: Optional[str] = None) -> Workspace:
        self.maybe_purge()
        wid = normalize_id(wid, "workspace") if wid else new_id(32)
        now = time.time()
        meta = {"id": wid, "name": name[:120], "created_at": now, "updated_at": now, "datasets": {},
                "active_dataset_id": None, "activity": [], "stages": {}, "chat": [], "rules": [],
                "experiments": [], "models": [], "analyses": {}}
        ws = Workspace(wid, meta, self.base)
        ws.dir.mkdir(parents=True, exist_ok=True)
        ws.save()
        with self._lock:
            self._remember(ws)
        return ws

    def _remember(self, ws: Workspace) -> None:
        self._cache[ws.id] = ws
        self._cache.move_to_end(ws.id)
        while len(self._cache) > max(1, settings.max_cached_workspaces):
            self._cache.popitem(last=False)

    def get(self, wid: str) -> Workspace:
        wid = normalize_id(wid, "workspace")
        with self._lock:
            if wid in self._cache:
                self._cache.move_to_end(wid)
                return self._cache[wid]
            path = safe_child(self.base, wid, "workspace.json")
            if not path.exists():
                raise NotFound("Workspace not found. It may have expired or been deleted.")
            with open(path, encoding="utf-8") as fh:
                meta = json.load(fh)
            ws = Workspace(wid, meta, self.base)
            self._remember(ws)
            return ws

    def exists(self, wid: str) -> bool:
        try:
            self.get(wid)
            return True
        except NotFound:
            return False

    def delete(self, wid: str) -> None:
        wid = normalize_id(wid, "workspace")
        with self._lock:
            self._cache.pop(wid, None)
            d = safe_child(self.base, wid)
            if not d.exists():
                raise NotFound("Workspace not found")
            shutil.rmtree(d, ignore_errors=True)

    def list(self, limit: int = 50) -> List[Dict[str, Any]]:
        if not settings.workspace_listing:
            raise Forbidden("Workspace listing is disabled on this server (WORKSPACE_LISTING=false).")
        self.maybe_purge()
        items = []
        for p in self.base.iterdir():
            f = p / "workspace.json"
            if not f.exists():
                continue
            try:
                ws = self.get(p.name)
                items.append(ws.summary())
            except Exception:  # corrupted folder: skip, keep listing usable
                log.warning("Skipping unreadable workspace %s", p.name)
        items.sort(key=lambda x: x.get("updated_at") or 0, reverse=True)
        return items[:limit]

    def maybe_purge(self, force: bool = False) -> int:
        """Delete workspaces not updated within the retention window."""
        now = time.time()
        if not force and now - self._last_purge < 3600:
            return 0
        self._last_purge = now
        ttl = settings.workspace_ttl_days * 86400
        if ttl <= 0:
            return 0
        removed = 0
        for p in self.base.iterdir():
            f = p / "workspace.json"
            try:
                updated = json.loads(f.read_text(encoding="utf-8")).get("updated_at", 0) if f.exists() else 0
            except Exception:
                continue
            if f.exists() and now - updated > ttl:
                with self._lock:
                    self._cache.pop(p.name, None)
                shutil.rmtree(p, ignore_errors=True)
                removed += 1
        if removed:
            log.info("Retention: removed %d expired workspace(s)", removed)
        return removed

    def clear_cache(self) -> None:
        with self._lock:
            self._cache.clear()


store = WorkspaceStore()
