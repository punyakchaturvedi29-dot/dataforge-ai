"""HTTP middleware: request ids, access logging and request body size limits."""
from __future__ import annotations

import logging
import time
import uuid

from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import JSONResponse

from .config import settings

log = logging.getLogger("dataforge.http")

# Routes that legitimately receive large multipart bodies. Their size is enforced
# by the streaming upload reader instead (see agents/ingest.read_upload).
_UPLOAD_SUFFIXES = ("/upload", "/upload/", "/sources", "/predict/batch", "/multifile/upload", "/inspect")


class RequestContextMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        rid = request.headers.get("x-request-id", "")[:64] or uuid.uuid4().hex[:12]
        request.state.request_id = rid

        # Reject oversized JSON / form bodies early using Content-Length.
        cl = request.headers.get("content-length")
        path = request.url.path
        is_upload = path.rstrip("/").endswith(tuple(s.rstrip("/") for s in _UPLOAD_SUFFIXES))
        if cl and cl.isdigit():
            limit = settings.max_upload_bytes + 1024 * 1024 if is_upload else settings.max_json_body_kb * 1024
            if int(cl) > limit:
                return JSONResponse(
                    status_code=413,
                    content={"error": {"code": "payload_too_large",
                                       "message": f"Request body exceeds the {limit // 1024} KB limit",
                                       "details": {}, "request_id": rid},
                             "detail": "Request body too large"},
                )

        t0 = time.perf_counter()
        response = await call_next(request)
        dt = (time.perf_counter() - t0) * 1000
        response.headers["x-request-id"] = rid
        response.headers.setdefault("X-Content-Type-Options", "nosniff")
        response.headers.setdefault("Referrer-Policy", "no-referrer")
        if path.startswith("/api"):
            response.headers.setdefault("Cache-Control", "no-store")
        log.info("%s %s -> %s (%.0f ms) rid=%s", request.method, path, response.status_code, dt, rid)
        return response


def setup_logging() -> None:
    level = getattr(logging, settings.log_level.upper(), logging.INFO)
    logging.basicConfig(
        level=level,
        format="%(asctime)s %(levelname)-7s %(name)s: %(message)s",
    )
    # Keep third-party noise down.
    for noisy in ("matplotlib", "PIL", "urllib3", "httpx", "httpcore"):
        logging.getLogger(noisy).setLevel(logging.WARNING)
