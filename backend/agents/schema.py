"""
Column type inference.

Two complementary views are produced for every column:

* ``logical_type`` – how the column should be *analysed*:
  numeric, categorical, boolean, datetime, text, identifier, constant, empty.
  Inferred from the values (dtype, cardinality, uniqueness, shape), with the
  column name used only as a tie-breaker.
* ``semantic_type`` – a business label (revenue, date, location, ...) inferred
  from the column name. Kept from v4 because the dashboard and copilot use it.
"""
from __future__ import annotations

import re
import warnings
from typing import Dict, List

import numpy as np
import pandas as pd

# Keyword sets matched against *tokens* of the column name (not substrings), with the
# last token (the head noun, e.g. "channel" in "sales_channel") taking priority.
SEMANTIC_KEYWORDS = [
    ("id",         {"id", "uuid", "guid", "key", "sku", "identifier", "code"}),
    ("date",       {"date", "time", "timestamp", "datetime", "created", "updated", "dob", "birthdate",
                    "month", "year", "week", "day", "period", "dt"}),
    ("duration",   {"days", "hours", "minutes", "seconds", "duration", "tenure", "leadtime", "weeks", "months", "years"}),
    ("percentage", {"pct", "percent", "percentage", "ratio", "rate", "margin", "share", "growth"}),
    ("revenue",    {"revenue", "sales", "income", "earnings", "earning", "amount", "price", "cost", "fee", "payment",
                    "spend", "budget", "profit", "gmv", "salary", "value", "total", "usd", "eur"}),
    ("quantity",   {"quantity", "qty", "count", "units", "unit", "volume", "stock", "inventory", "num", "number"}),
    ("name",       {"name", "firstname", "lastname", "fullname", "title"}),
    ("email",      {"email", "mail"}),
    ("location",   {"city", "state", "country", "region", "zip", "postal", "postcode", "address", "location",
                    "territory", "area", "province"}),
    ("age",        {"age"}),
    ("gender",     {"gender", "sex"}),
    ("rating",     {"rating", "score", "grade", "rank", "satisfaction", "nps", "stars"}),
    ("category",   {"category", "type", "segment", "class", "status", "tag", "channel", "group", "tier", "label",
                    "department", "dept", "kind", "source", "method"}),
]
# v4 compatibility: regex patterns are still exported for callers that used them.
PATTERNS = {sem: r"\b(" + "|".join(sorted(kws)) + r")\b" for sem, kws in SEMANTIC_KEYWORDS}
ID_KEYWORDS = SEMANTIC_KEYWORDS[0][1]


def name_tokens(col: str) -> List[str]:
    s = re.sub(r"([a-z])([A-Z])", r"\1_\2", str(col))
    return [t for t in re.split(r"[^a-z0-9]+", s.lower()) if t]

BOOL_TOKENS = {"true", "false", "yes", "no", "y", "n", "t", "f"}
NULL_TOKENS = ["", "na", "n/a", "nan", "null", "none", "-", "--", "?", "missing", "#n/a"]

LOGICAL_TYPES = ["numeric", "categorical", "boolean", "datetime", "text", "identifier", "constant", "empty"]


def is_text_dtype(s: pd.Series) -> bool:
    """True for object and pandas string dtypes (pandas 2 and 3 compatible)."""
    return (pd.api.types.is_object_dtype(s) or pd.api.types.is_string_dtype(s)) and not pd.api.types.is_bool_dtype(s)


def is_numeric_dtype(s: pd.Series) -> bool:
    return pd.api.types.is_numeric_dtype(s) and not pd.api.types.is_bool_dtype(s)


def semantic_type(col: str, s: pd.Series | None = None) -> str:
    toks = name_tokens(col)
    if toks and toks[0] in ("is", "has", "was", "flag"):
        return "boolean"
    ordered = ([toks[-1]] + toks[:-1]) if toks else []
    for t in ordered:
        for sem, kws in SEMANTIC_KEYWORDS:
            if t in kws:
                return sem
    if s is not None:
        if pd.api.types.is_datetime64_any_dtype(s):
            return "date"
        if pd.api.types.is_bool_dtype(s):
            return "boolean"
        if pd.api.types.is_numeric_dtype(s):
            return "numeric"
    return "general"


def detect_schema(df: pd.DataFrame) -> dict:
    """v4-compatible: {column: semantic_type}."""
    return {col: semantic_type(col, df[col]) for col in df.columns}


def _sample(s: pd.Series, n: int = 5000) -> pd.Series:
    nn = s.dropna()
    if len(nn) > n:
        return nn.sample(n, random_state=0)
    return nn


def numeric_parse_rate(s: pd.Series) -> float:
    """Share of non-null text values that become numbers after removing currency/percent/thousand separators."""
    smp = _sample(s).astype(str).str.strip()
    smp = smp[~smp.str.lower().isin(NULL_TOKENS)]
    if len(smp) == 0:
        return 0.0
    cleaned = smp.str.replace(r"^\((.*)\)$", r"-\1", regex=True)  # (123) accounting negatives
    cleaned = cleaned.str.replace(r"[\$£€¥₹%\s,]", "", regex=True)
    parsed = pd.to_numeric(cleaned, errors="coerce")
    return float(parsed.notna().mean())


def datetime_parse_rate(s: pd.Series) -> float:
    smp = _sample(s, 2000).astype(str).str.strip()
    smp = smp[~smp.str.lower().isin(NULL_TOKENS)]
    if len(smp) == 0:
        return 0.0
    # Pure numbers are not dates (avoid treating 2024, 15 etc. as dates)
    if pd.to_numeric(smp, errors="coerce").notna().mean() > 0.5:
        return 0.0
    # Must contain a date-like separator or month name
    if not smp.str.contains(r"[-/.:]|jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec", case=False, regex=True).mean() > 0.8:
        return 0.0
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        parsed = pd.to_datetime(smp, errors="coerce", format="mixed")
    return float(parsed.notna().mean())


def _looks_sequential_id(s: pd.Series) -> bool:
    if not pd.api.types.is_integer_dtype(s) or len(s) < 20:
        return False
    v = s.dropna().to_numpy()
    if len(v) < 20 or len(np.unique(v)) != len(v):
        return False
    d = np.diff(np.sort(v))
    return bool(np.median(d) == 1 and (d == 1).mean() > 0.9)


def infer_column(name: str, s: pd.Series) -> Dict:
    n = int(len(s))
    nn = s.dropna()
    n_missing = n - int(len(nn))
    try:
        n_unique = int(nn.nunique())
    except TypeError:  # unhashable values (e.g. lists from JSON)
        n_unique = int(nn.astype(str).nunique())
    unique_ratio = (n_unique / len(nn)) if len(nn) else 0.0
    sem = semantic_type(name, s)
    hints: List[str] = []
    toks = name_tokens(name)
    name_is_id = bool(toks) and (toks[-1] in ID_KEYWORDS or (len(toks) == 1 and toks[0] in ID_KEYWORDS))

    if len(nn) == 0:
        logical = "empty"
    elif n_unique == 1:
        logical = "constant"
    elif pd.api.types.is_bool_dtype(s):
        logical = "boolean"
    elif pd.api.types.is_datetime64_any_dtype(s):
        logical = "datetime"
    elif pd.api.types.is_numeric_dtype(s):
        vals = set(pd.unique(nn)) if n_unique <= 2 else set()
        if n_unique == 2 and vals <= {0, 1, 0.0, 1.0}:
            logical = "boolean"
        elif (name_is_id and unique_ratio > 0.95 and pd.api.types.is_integer_dtype(s)) or _looks_sequential_id(s):
            logical = "identifier"
        else:
            logical = "numeric"
    elif is_text_dtype(s):
        txt = _sample(s).astype(str)
        lower_vals = set(txt.str.strip().str.lower().unique()[:10])
        avg_len = float(txt.str.len().mean()) if len(txt) else 0.0
        avg_words = float(txt.str.split().str.len().mean()) if len(txt) else 0.0
        has_space = float(txt.str.contains(" ").mean()) if len(txt) else 0.0
        if n_unique == 2 and lower_vals <= BOOL_TOKENS:
            logical = "boolean"
        elif n_unique <= 2 and len(nn) >= 2:
            logical = "categorical"
        elif unique_ratio > 0.9 and len(nn) >= 20 and has_space < 0.2 and avg_len < 40:
            logical = "identifier"
        elif name_is_id and unique_ratio > 0.95:
            logical = "identifier"
        elif avg_len > 60 or (avg_words >= 4 and unique_ratio > 0.5):
            logical = "text"
        elif unique_ratio > 0.9 and len(nn) >= 20:
            logical = "text"  # e.g. person names: free-form, near-unique
        else:
            logical = "categorical"

        # Hints for the cleaning engine – never applied automatically.
        if logical in ("categorical", "text", "identifier"):
            npr = numeric_parse_rate(s)
            if npr >= 0.8:
                hints.append("numeric_string")
            else:
                dpr = datetime_parse_rate(s)
                if dpr >= 0.8:
                    hints.append("datetime_string")
    else:
        logical = "categorical"

    if logical in ("categorical",) and n_unique > 50:
        hints.append("high_cardinality")
    if sem == "date" and logical not in ("datetime",) and "datetime_string" not in hints and is_text_dtype(s):
        pass

    return {
        "name": str(name),
        "dtype": str(s.dtype),
        "logical_type": logical,
        "semantic_type": sem,
        "n_missing": n_missing,
        "missing_pct": round(n_missing / n * 100, 2) if n else 0.0,
        "n_unique": n_unique,
        "unique_ratio": round(unique_ratio, 4),
        "hints": hints,
    }


def infer_schema(df: pd.DataFrame) -> List[Dict]:
    return [infer_column(c, df[c]) for c in df.columns]


def columns_by_type(schema: List[Dict]) -> Dict[str, List[str]]:
    out: Dict[str, List[str]] = {t: [] for t in LOGICAL_TYPES}
    for c in schema:
        out.setdefault(c["logical_type"], []).append(c["name"])
    return out


_MEASURE_PRI = {"revenue": 0, "quantity": 2, "rating": 3, "percentage": 4, "age": 6, "duration": 5}
_HEADLINE_TOKENS = {"revenue", "sales", "amount", "profit", "income", "total", "gmv", "spend", "value"}


def measure_priority(col: str, semantic: str | None) -> tuple:
    """Sort key putting headline business measures (revenue, sales) before unit-level ones (unit price)."""
    toks = set(name_tokens(col))
    base = _MEASURE_PRI.get(semantic or "", 7)
    if semantic == "revenue" and not (toks & _HEADLINE_TOKENS):
        base = 1  # price / cost / fee: monetary but per-unit
    return (base,)


def rank_measures(cols: List[str], schema_map: Dict[str, Dict]) -> List[str]:
    return sorted(cols, key=lambda c: (measure_priority(c, schema_map.get(c, {}).get("semantic_type")), cols.index(c)))


def analysis_columns(df: pd.DataFrame, schema: List[Dict] | None = None) -> Dict[str, List[str]]:
    """Columns suitable for each kind of analysis.

    * numeric:      measures (identifiers and booleans excluded)
    * categorical:  groupable dimensions (<= 50 levels) incl. booleans
    * datetime:     time axes
    """
    schema = schema or infer_schema(df)
    by = columns_by_type(schema)
    card = {c["name"]: c["n_unique"] for c in schema}
    cats = [c for c in by["categorical"] + by["boolean"] if card.get(c, 0) <= 50]
    return {
        "numeric": by["numeric"],
        "categorical": cats,
        "datetime": by["datetime"],
        "text": by["text"],
        "identifier": by["identifier"],
        "constant": by["constant"] + by["empty"],
        "all_categorical": by["categorical"] + by["boolean"],
    }
