"""Analytics copilot endpoints."""
from __future__ import annotations

import time
from typing import Any, Dict

from fastapi import APIRouter
from pydantic import BaseModel, Field

from agents import copilot
from agents.forecasting import run_forecast
from core.jsonsafe import to_jsonable
from services import analysis as A

from ._deps import get_ws

router = APIRouter()


def build_ctx(ws) -> copilot.Ctx:
    ds = ws.active
    sch = A.schema(ws, ds)
    mem = ws.meta.setdefault("chat_context", {})
    tgt = A.target(ws)
    if tgt:
        mem.setdefault("target", tgt)
    return copilot.Ctx(
        df=ds.working, schema=sch, dataset_name=ds.name,
        findings=lambda: A.findings(ws, ds), profile=lambda: A.profile(ws, ds),
        latest_model=A.latest_model(ws),
        recommendations=lambda: A.recs(ws), memory=mem,
        forecast_fn=lambda **kw: run_forecast(ds.working, kw["date_col"], kw.get("target"), kw.get("horizon", 6),
                                              schema=sch),
        exec_summary=lambda: A.executive_summary(ws),
    )


class Ask(BaseModel):
    message: str = Field(..., min_length=1, max_length=2000)


@router.post("/{workspace_id}/copilot")
def ask(workspace_id: str, body: Ask):
    ws = get_ws(workspace_id)
    ctx = build_ctx(ws)
    history = ws.meta.get("chat", [])
    reply = copilot.answer(ctx, body.message, [{"role": h["role"], "content": h["content"]} for h in history])
    with ws.lock:
        chat = ws.meta.setdefault("chat", [])
        chat.append({"role": "user", "content": body.message, "ts": time.time()})
        chat.append({"role": "assistant", "content": reply["answer"], "ts": time.time(), "payload": reply})
        ws.meta["chat"] = chat[-60:]
        ws.meta["chat_context"] = to_jsonable(ctx.memory)
        ws.save()
    return reply


@router.get("/{workspace_id}/copilot/history")
def history(workspace_id: str):
    return {"messages": get_ws(workspace_id).meta.get("chat", [])}


@router.delete("/{workspace_id}/copilot/history", status_code=204)
def clear(workspace_id: str):
    ws = get_ws(workspace_id)
    with ws.lock:
        ws.meta["chat"] = []
        ws.meta["chat_context"] = {}
        ws.save()
