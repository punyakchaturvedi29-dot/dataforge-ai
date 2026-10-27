"""v4-compatible analysis endpoint (``GET /api/analyze/{session_id}``)."""
from fastapi import APIRouter

from agents.eda import run_eda
from agents.insights import generate_insights

from ._deps import get_ws

router = APIRouter()


@router.get("/{session_id}")
def analyze(session_id: str):
    ws = get_ws(session_id)
    df = ws.active.working
    eda = ws.cached("legacy_eda", None, lambda: run_eda(df))
    return {"session_id": ws.id, "eda": eda, "insights": generate_insights(df, eda)}
