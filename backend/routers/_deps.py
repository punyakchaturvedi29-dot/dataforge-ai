"""Shared helpers for routers."""
from __future__ import annotations

import io
from typing import Iterable, Optional

import pandas as pd
from fastapi.responses import StreamingResponse

from core.config import settings
from core.security import content_disposition, escape_formulas
from services.workspace import Workspace, store


def get_ws(workspace_id: str) -> Workspace:
    return store.get(workspace_id)


def csv_response(df: pd.DataFrame, filename: str, index: bool = False) -> StreamingResponse:
    """Stream a dataframe as CSV with spreadsheet formula injection neutralised."""
    out = escape_formulas(df) if settings.csv_formula_escape else df

    def gen() -> Iterable[bytes]:
        yield "﻿".encode("utf-8")  # BOM so Excel detects UTF-8
        chunk = 50_000
        for start in range(0, max(len(out), 1), chunk):
            buf = io.StringIO()
            out.iloc[start:start + chunk].to_csv(buf, index=index, header=start == 0)
            yield buf.getvalue().encode("utf-8")

    return StreamingResponse(gen(), media_type="text/csv; charset=utf-8",
                             headers={"Content-Disposition": content_disposition(filename)})


def xlsx_response(df: pd.DataFrame, filename: str) -> StreamingResponse:
    out = escape_formulas(df) if settings.csv_formula_escape else df
    if len(out) > 1_000_000:
        from core.errors import PayloadTooLarge
        raise PayloadTooLarge("Excel exports are limited to 1,000,000 rows; download CSV instead.")
    buf = io.BytesIO()
    tmp = out.copy()
    for c in tmp.columns:
        if pd.api.types.is_datetime64_any_dtype(tmp[c]) and getattr(tmp[c].dt, "tz", None) is not None:
            tmp[c] = tmp[c].dt.tz_localize(None)
    tmp.to_excel(buf, index=False, engine="openpyxl")
    buf.seek(0)
    return StreamingResponse(buf, media_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                             headers={"Content-Disposition": content_disposition(filename)})


def stem(filename: Optional[str]) -> str:
    f = filename or "dataset"
    return f.rsplit(".", 1)[0] if "." in f else f
