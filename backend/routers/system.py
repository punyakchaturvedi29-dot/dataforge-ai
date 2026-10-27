"""Health checks and public configuration."""
from __future__ import annotations

import os
import shutil
import time

from fastapi import APIRouter

from agents.automl import available_models, library_versions
from core.config import settings

router = APIRouter()
_STARTED = time.time()


@router.get("/health", summary="Liveness and readiness information")
def health():
    settings.ensure_dirs()
    writable = os.access(settings.data_dir, os.W_OK)
    usage = shutil.disk_usage(settings.data_dir)
    checks = {
        "data_dir_writable": writable,
        "disk_free_mb": round(usage.free / 1e6),
        "models": available_models(),
        "llm_configured": settings.llm_enabled,
    }
    try:
        import statsmodels  # noqa: F401
        checks["statsmodels"] = True
    except Exception:
        checks["statsmodels"] = False
    return {"status": "ok" if writable else "degraded", "version": settings.version,
            "uptime_s": round(time.time() - _STARTED), "checks": checks, "versions": library_versions()}


@router.get("/config", summary="Non-secret server configuration for the UI")
def config():
    return {**settings.public_dict(), "models": available_models()}
