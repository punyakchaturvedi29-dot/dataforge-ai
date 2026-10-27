"""
DataForge AI — FastAPI application.

Run from the ``backend`` directory::

    uvicorn main:app --port 8000

API documentation is served at /docs (Swagger) and /redoc.
"""
from __future__ import annotations

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from core.config import settings
from core.errors import install_error_handlers
from core.middleware import RequestContextMiddleware, setup_logging
from routers import (advanced, analyses, analyze, assistant, chat, clean, explore, features, ml, predict, report,
                     reports, sources, system, upload, workspaces)

setup_logging()
log = logging.getLogger("dataforge")


@asynccontextmanager
async def lifespan(app: FastAPI):
    settings.ensure_dirs()
    settings.get_secret()
    from services.workspace import store
    removed = store.maybe_purge(force=True)
    log.info("DataForge AI %s ready (data dir: %s, LLM %s, %d expired workspace(s) purged)", settings.version,
             settings.data_dir, "enabled" if settings.llm_enabled else "disabled", removed)
    yield


def create_app() -> FastAPI:
    app = FastAPI(
        title="DataForge AI", version=settings.version, lifespan=lifespan,
        description="AI-assisted data analytics: ingestion, profiling, auditable cleaning, insights, AutoML, "
                    "forecasting, diagnostics, multi-file joins, a grounded copilot and reports.",
    )
    app.add_middleware(RequestContextMiddleware)
    app.add_middleware(CORSMiddleware, allow_origins=settings.cors_origins, allow_credentials=False,
                       allow_methods=["GET", "POST", "PATCH", "DELETE"], allow_headers=["*"],
                       expose_headers=["Content-Disposition", "x-request-id"])
    install_error_handlers(app)

    ws = "/api/workspaces"
    app.include_router(system.router, prefix="/api", tags=["system"])
    app.include_router(workspaces.router, prefix=ws, tags=["workspaces & data"])
    app.include_router(clean.router, prefix=ws, tags=["cleaning & quality rules"])
    app.include_router(explore.router, prefix=ws, tags=["profiling, insights & charts"])
    app.include_router(ml.router, prefix=ws, tags=["machine learning"])
    app.include_router(ml.jobs_router, prefix="/api/jobs", tags=["jobs"])
    app.include_router(analyses.router, prefix=ws, tags=["forecasting & diagnostics"])
    app.include_router(sources.router, prefix=ws, tags=["multi-file"])
    app.include_router(assistant.router, prefix=ws, tags=["copilot"])
    app.include_router(reports.router, prefix=ws, tags=["reports"])

    # v4-compatible endpoints (session_id == workspace id)
    app.include_router(upload.router, prefix="/api/upload", tags=["v4 compatibility"])
    app.include_router(analyze.router, prefix="/api/analyze", tags=["v4 compatibility"])
    app.include_router(chat.router, prefix="/api/chat", tags=["v4 compatibility"])
    app.include_router(report.router, prefix="/api/report", tags=["v4 compatibility"])
    app.include_router(predict.router, prefix="/api/predict", tags=["v4 compatibility"])
    app.include_router(features.router, prefix="/api/features", tags=["v4 compatibility"])
    app.include_router(advanced.router, prefix="/api/advanced", tags=["v4 compatibility"])

    dist = settings.frontend_dist
    if (dist / "index.html").exists():
        # Single-process deployment: serve the built React app.
        app.mount("/assets", StaticFiles(directory=dist / "assets"), name="assets")

        @app.get("/{path:path}", include_in_schema=False)
        def spa(path: str):
            if path.startswith("api/"):
                return JSONResponse({"detail": "Not found"}, status_code=404)
            candidate = (dist / path).resolve()
            if path and dist.resolve() in candidate.parents and candidate.is_file():
                return FileResponse(candidate)
            return FileResponse(dist / "index.html")
    else:
        @app.get("/", include_in_schema=False)
        def root():
            return {"status": "DataForge AI API running", "version": settings.version, "docs": "/docs"}

    return app


app = create_app()
