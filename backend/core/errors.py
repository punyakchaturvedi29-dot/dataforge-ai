"""
Consistent API errors.

Every error response has the shape::

    {"error": {"code": "...", "message": "...", "details": {...}, "request_id": "..."},
     "detail": "<message>"}

``detail`` is kept for backwards compatibility with the v4 frontend, which read
``e.response.data.detail``. Internal exceptions never leak tracebacks or
exception text to clients; they are logged with the request id instead.
"""
from __future__ import annotations

import logging
from typing import Any, Dict, Optional

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from starlette.exceptions import HTTPException as StarletteHTTPException

log = logging.getLogger("dataforge.errors")


class AppError(Exception):
    status_code = 400
    code = "bad_request"

    def __init__(self, message: str, *, code: Optional[str] = None, status_code: Optional[int] = None,
                 details: Optional[Dict[str, Any]] = None):
        super().__init__(message)
        self.message = message
        if code:
            self.code = code
        if status_code:
            self.status_code = status_code
        self.details = details or {}


class NotFound(AppError):
    status_code = 404
    code = "not_found"


class ValidationFailed(AppError):
    status_code = 422
    code = "validation_failed"


class Conflict(AppError):
    status_code = 409
    code = "conflict"


class PayloadTooLarge(AppError):
    status_code = 413
    code = "payload_too_large"


class UnsupportedMedia(AppError):
    status_code = 415
    code = "unsupported_media_type"


class ResourceLimit(AppError):
    status_code = 429
    code = "resource_limit"


class Forbidden(AppError):
    status_code = 403
    code = "forbidden"


def _body(request: Request, code: str, message: str, details: Optional[Dict] = None) -> Dict[str, Any]:
    rid = getattr(request.state, "request_id", None)
    return {
        "error": {"code": code, "message": message, "details": details or {}, "request_id": rid},
        "detail": message,
    }


def install_error_handlers(app: FastAPI) -> None:
    @app.exception_handler(AppError)
    async def _app_error(request: Request, exc: AppError):
        return JSONResponse(status_code=exc.status_code, content=_body(request, exc.code, exc.message, exc.details))

    @app.exception_handler(StarletteHTTPException)
    async def _http_error(request: Request, exc: StarletteHTTPException):
        code = {404: "not_found", 405: "method_not_allowed", 401: "unauthorized", 403: "forbidden"}.get(exc.status_code, "http_error")
        message = exc.detail if isinstance(exc.detail, str) else "Request failed"
        return JSONResponse(status_code=exc.status_code, content=_body(request, code, message))

    @app.exception_handler(RequestValidationError)
    async def _validation_error(request: Request, exc: RequestValidationError):
        problems = []
        for e in exc.errors()[:20]:
            loc = ".".join(str(p) for p in e.get("loc", []) if p != "body")
            problems.append({"field": loc, "message": e.get("msg", "invalid value")})
        msg = "Invalid request: " + "; ".join(f"{p['field']}: {p['message']}" for p in problems[:5])
        return JSONResponse(status_code=422, content=_body(request, "invalid_request", msg, {"problems": problems}))

    @app.exception_handler(Exception)
    async def _unhandled(request: Request, exc: Exception):
        rid = getattr(request.state, "request_id", None)
        log.exception("Unhandled error (request_id=%s) on %s %s", rid, request.method, request.url.path)
        return JSONResponse(
            status_code=500,
            content=_body(request, "internal_error",
                          "An unexpected error occurred. The details were logged on the server "
                          f"(request id {rid})."),
        )
