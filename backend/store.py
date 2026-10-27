"""
v4 compatibility shim.

v4 kept datasets in an unbounded in-memory dict keyed by ``session_id``. v5
stores them in persistent workspaces (see ``services/workspace.py``); a v4
``session_id`` is simply a workspace id. These helpers keep old imports working.
"""
from __future__ import annotations

from typing import Any, Dict, Optional

import pandas as pd

from core.errors import NotFound
from services.workspace import store


def save_dataset(session_id: str, df: pd.DataFrame, filename: str, raw_df: pd.DataFrame = None) -> str:
    ws = store.create(name=filename, wid=session_id.replace("-", "") if session_id else None)
    ws.add_dataset(raw_df if raw_df is not None else df, filename=filename, make_active=True)
    return ws.id


def get_dataset(session_id: str) -> Optional[Dict[str, Any]]:
    try:
        ws = store.get(session_id)
        ds = ws.active
    except NotFound:
        return None
    return {"df": ds.working, "raw_df": ds.original, "filename": ds.meta.get("filename"), "workspace": ws}


def get_df(session_id: str) -> Optional[pd.DataFrame]:
    d = get_dataset(session_id)
    return d["df"] if d else None
