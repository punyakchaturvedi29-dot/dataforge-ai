"""v4-compatible chat endpoint (``POST /api/chat/``) backed by the grounded copilot.

The response stays newline-delimited JSON (``{"text": ...}``) for old clients;
the full structured reply (evidence, clarifications) is sent as a final
``{"reply": ...}`` line. Use ``POST /api/workspaces/{id}/copilot`` for new code.
"""
import json

from fastapi import APIRouter
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from agents import copilot

from ._deps import get_ws
from .assistant import build_ctx

router = APIRouter()


class ChatRequest(BaseModel):
    session_id: str
    message: str = Field(..., min_length=1, max_length=2000)
    history: list = Field(default_factory=list, max_length=50)


@router.post("/")
def chat(req: ChatRequest):
    ws = get_ws(req.session_id)
    hist = [{"role": h.get("role"), "content": str(h.get("content", ""))[:2000]} for h in req.history
            if isinstance(h, dict) and h.get("role") in ("user", "assistant")]
    reply = copilot.answer(build_ctx(ws), req.message, hist)

    def gen():
        yield json.dumps({"text": reply["answer"]}) + "\n"
        yield json.dumps({"reply": reply}, default=str) + "\n"

    return StreamingResponse(gen(), media_type="application/x-ndjson")
