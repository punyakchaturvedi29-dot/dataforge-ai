"""v4-compatible PDF report (``GET /api/report/{session_id}``)."""
from fastapi import APIRouter

from ._deps import get_ws
from .reports import ReportBody, report as build

router = APIRouter()


@router.get("/{session_id}")
def generate_report(session_id: str):
    ws = get_ws(session_id)
    return build(ws.id, ReportBody(format="pdf"))
