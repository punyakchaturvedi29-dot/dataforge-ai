"""Conversion of numpy / pandas objects into strict JSON-safe Python values."""
from __future__ import annotations

import datetime as _dt
import math
from typing import Any

import numpy as np
import pandas as pd


def to_jsonable(obj: Any, ndigits: int | None = None) -> Any:
    """Recursively convert to JSON-safe values. NaN/Inf become ``None``."""
    if obj is None or isinstance(obj, (str, bool)):
        return obj
    if isinstance(obj, (np.bool_,)):
        return bool(obj)
    if isinstance(obj, (int, np.integer)):
        return int(obj)
    if isinstance(obj, (float, np.floating)):
        f = float(obj)
        if math.isnan(f) or math.isinf(f):
            return None
        return round(f, ndigits) if ndigits is not None else f
    if isinstance(obj, dict):
        return {str(k): to_jsonable(v, ndigits) for k, v in obj.items()}
    if isinstance(obj, (list, tuple, set)):
        return [to_jsonable(v, ndigits) for v in obj]
    if isinstance(obj, np.ndarray):
        return [to_jsonable(v, ndigits) for v in obj.tolist()]
    if isinstance(obj, (pd.Timestamp, _dt.datetime, _dt.date)):
        try:
            if pd.isna(obj):
                return None
        except Exception:
            pass
        return obj.isoformat()
    if isinstance(obj, pd.Timedelta):
        return str(obj)
    if isinstance(obj, (pd.Series, pd.Index)):
        return [to_jsonable(v, ndigits) for v in obj.tolist()]
    if isinstance(obj, pd.DataFrame):
        return records(obj)
    if obj is pd.NA or obj is pd.NaT:
        return None
    try:
        if pd.isna(obj):
            return None
    except Exception:
        pass
    return str(obj)


def records(df: pd.DataFrame, ndigits: int | None = None) -> list:
    """DataFrame -> list of JSON-safe row dicts."""
    out = []
    cols = [str(c) for c in df.columns]
    for row in df.itertuples(index=False, name=None):
        out.append({c: to_jsonable(v, ndigits) for c, v in zip(cols, row)})
    return out


def num(v: Any, ndigits: int | None = None):
    """Single scalar -> float/None."""
    return to_jsonable(v, ndigits)
