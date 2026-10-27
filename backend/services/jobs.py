"""
Background jobs for long-running, CPU-bound work (model training).

FastAPI ``async`` endpoints do not make CPU-bound work non-blocking, so training
runs in a bounded thread pool. The API returns a job id immediately and the
client polls ``/api/jobs/{id}`` for progress.

Job state is *session state*: it lives in memory and is lost on restart. Job
*results* (experiments and models) are persisted by the job function itself.
"""
from __future__ import annotations

import logging
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Callable, Dict, List, Optional

from core.config import settings
from core.errors import AppError, NotFound, ResourceLimit
from core.jsonsafe import to_jsonable
from core.security import new_id, normalize_id

log = logging.getLogger("dataforge.jobs")


class Job:
    def __init__(self, kind: str, workspace_id: str, description: str):
        self.id = new_id(16)
        self.kind = kind
        self.workspace_id = workspace_id
        self.description = description
        self.status = "queued"
        self.progress = 0.0
        self.message = "Waiting to start"
        self.log: List[Dict[str, Any]] = []
        self.result: Any = None
        self.error: Optional[Dict[str, Any]] = None
        self.created_at = time.time()
        self.started_at: Optional[float] = None
        self.finished_at: Optional[float] = None
        self.cancel_requested = False

    def update(self, fraction: float, message: str) -> None:
        self.progress = max(self.progress, min(1.0, float(fraction)))
        if message != self.message:
            self.log.append({"t": round(time.time() - (self.started_at or self.created_at), 2), "message": message})
            self.log = self.log[-50:]
        self.message = message

    def public(self) -> Dict[str, Any]:
        return to_jsonable({
            "id": self.id, "kind": self.kind, "workspace_id": self.workspace_id, "description": self.description,
            "status": self.status, "progress": self.progress, "message": self.message, "log": self.log,
            "result": self.result if self.status == "succeeded" else None, "error": self.error,
            "created_at": self.created_at, "started_at": self.started_at, "finished_at": self.finished_at,
            "elapsed_s": (self.finished_at or time.time()) - (self.started_at or self.created_at),
        })


class JobManager:
    def __init__(self) -> None:
        self._jobs: Dict[str, Job] = {}
        self._lock = threading.Lock()
        self._pool: Optional[ThreadPoolExecutor] = None

    def _executor(self) -> ThreadPoolExecutor:
        if self._pool is None:
            self._pool = ThreadPoolExecutor(max_workers=max(1, settings.max_concurrent_jobs), thread_name_prefix="df-job")
        return self._pool

    def submit(self, kind: str, workspace_id: str, description: str,
               fn: Callable[[Job], Any]) -> Job:
        with self._lock:
            active = [j for j in self._jobs.values() if j.status in ("queued", "running")]
            if len(active) >= settings.max_queued_jobs:
                raise ResourceLimit(f"Too many jobs are running or queued ({len(active)}). Try again shortly.")
            same_ws = [j for j in active if j.workspace_id == workspace_id and j.kind == kind]
            if len(same_ws) >= 2:
                raise ResourceLimit("This workspace already has training jobs in progress. Wait for them to finish.")
            job = Job(kind, workspace_id, description)
            self._jobs[job.id] = job
            self._gc()

        def run():
            if job.cancel_requested:
                job.status, job.finished_at = "cancelled", time.time()
                return
            job.status, job.started_at = "running", time.time()
            job.update(0.01, "Starting")
            try:
                job.result = fn(job)
                job.status = "succeeded"
                job.progress = 1.0
                job.message = "Completed"
            except CancelledByUser:
                job.status = "cancelled"
                job.message = "Cancelled"
            except AppError as e:
                job.status = "failed"
                job.error = {"code": e.code, "message": e.message}
                job.message = e.message
            except Exception as e:  # pragma: no cover - logged for diagnosis
                log.exception("Job %s failed", job.id)
                job.status = "failed"
                job.error = {"code": "internal_error",
                             "message": f"The job failed unexpectedly ({type(e).__name__}). Details are in the server log."}
                job.message = "Failed"
            finally:
                job.finished_at = time.time()

        self._executor().submit(run)
        return job

    def get(self, job_id: str) -> Job:
        jid = normalize_id(job_id, "job")
        j = self._jobs.get(jid)
        if not j:
            raise NotFound("Job not found (jobs are kept in memory and are cleared on restart)")
        return j

    def cancel(self, job_id: str) -> Job:
        j = self.get(job_id)
        j.cancel_requested = True
        if j.status == "queued":
            j.status, j.finished_at = "cancelled", time.time()
        return j

    def list(self, workspace_id: Optional[str] = None) -> List[Dict[str, Any]]:
        jobs = [j for j in self._jobs.values() if workspace_id is None or j.workspace_id == workspace_id]
        jobs.sort(key=lambda j: -j.created_at)
        return [dict(j.public(), result=None) for j in jobs[:30]]

    def _gc(self) -> None:
        cutoff = time.time() - 6 * 3600
        for jid in [k for k, j in self._jobs.items() if j.finished_at and j.finished_at < cutoff]:
            self._jobs.pop(jid, None)

    def wait(self, job_id: str, timeout: float = 120) -> Job:
        """Used by tests and the legacy synchronous endpoints."""
        t0 = time.time()
        j = self.get(job_id)
        while j.status in ("queued", "running") and time.time() - t0 < timeout:
            time.sleep(0.05)
        return j


class CancelledByUser(Exception):
    pass


jobs = JobManager()
