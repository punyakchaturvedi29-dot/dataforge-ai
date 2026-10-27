"""
Auditable data-preparation engine.

Cleaning is an ordered *operation log* applied to the untouched original
dataset. Every operation:

* is previewed before it is applied (before/after sample + measured impact),
* stores the parameters it *fitted* (e.g. the median used for imputation), so
  replaying the log is deterministic and the same transformation can be applied
  to new data at prediction time,
* reports its consequences: rows removed, non-null values lost, cells changed,
  values that could not be converted,
* requires explicit confirmation when it removes rows/columns or destroys
  non-null values.

The original dataset is never modified; undo simply replays a shorter log.
Row identity is preserved through the index (``source row``), which gives
row-level lineage.
"""
from __future__ import annotations

import re
import time
import warnings
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from core.errors import ValidationFailed
from core.jsonsafe import to_jsonable
from core.security import new_id

from .schema import NULL_TOKENS, BOOL_TOKENS, infer_schema, is_text_dtype, numeric_parse_rate, datetime_parse_rate

# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────


def _require_cols(df: pd.DataFrame, cols: Any, key: str = "columns", allow_empty: bool = False) -> List[str]:
    if cols is None:
        return []
    if isinstance(cols, str):
        cols = [cols]
    if not isinstance(cols, list) or (not cols and not allow_empty):
        raise ValidationFailed(f"'{key}' must be a non-empty list of column names")
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise ValidationFailed(f"Unknown column(s): {', '.join(map(str, missing[:5]))}")
    return list(cols)


def _require_col(df: pd.DataFrame, params: Dict, key: str = "column") -> str:
    c = params.get(key)
    if not isinstance(c, str) or c not in df.columns:
        raise ValidationFailed(f"Unknown or missing column for '{key}': {c!r}")
    return c


def _restore_text(out: pd.Series, like: pd.Series) -> pd.Series:
    """Convert a nullable 'string' result back to the column's original text dtype, keeping NaN (not pd.NA)."""
    r = out.astype(object).where(out.notna(), np.nan)
    if like.dtype != object:
        try:
            return r.astype(like.dtype)
        except (TypeError, ValueError):
            return r
    return r


def _text_cols(df: pd.DataFrame) -> List[str]:
    return [c for c in df.columns if is_text_dtype(df[c])]


def _clean_numeric_text(s: pd.Series, decimal: str = "auto") -> pd.Series:
    t = s.astype("string").str.strip()
    t = t.mask(t.str.lower().isin(NULL_TOKENS))
    t = t.str.replace(r"^\((.*)\)$", r"-\1", regex=True)
    t = t.str.replace(r"[\$£€¥₹%\s]", "", regex=True)
    if decimal == "auto":
        smp = t.dropna().head(2000)
        eu = smp.str.match(r"^-?\d{1,3}(\.\d{3})+(,\d+)?$|^-?\d+,\d+$").mean() if len(smp) else 0
        decimal = "," if eu > 0.5 else "."
    if decimal == ",":
        t = t.str.replace(".", "", regex=False).str.replace(",", ".", regex=False)
    else:
        t = t.str.replace(",", "", regex=False)
    return pd.to_numeric(t, errors="coerce").astype("float64")


def _auto_dayfirst(s: pd.Series) -> bool:
    smp = s.dropna().astype(str).head(2000)
    m = smp.str.extract(r"^(\d{1,2})[/.-](\d{1,2})[/.-]\d{2,4}")
    if m.dropna().empty:
        return False
    a = pd.to_numeric(m[0], errors="coerce")
    b = pd.to_numeric(m[1], errors="coerce")
    return bool((a > 12).any() and not (b > 12).any())


def _bounds(s: pd.Series, method: str, k: float = 1.5, lower_pct: float = 0.01, upper_pct: float = 0.99) -> Tuple[float, float]:
    v = s.dropna().astype(float)
    if len(v) == 0:
        raise ValidationFailed("Column has no numeric values")
    if method == "iqr":
        q1, q3 = v.quantile(0.25), v.quantile(0.75)
        iqr = q3 - q1
        return float(q1 - k * iqr), float(q3 + k * iqr)
    if method == "percentile":
        return float(v.quantile(lower_pct)), float(v.quantile(upper_pct))
    if method == "zscore":
        mu, sd = v.mean(), v.std(ddof=0)
        return float(mu - k * sd), float(mu + k * sd)
    if method == "mad":
        med = v.median()
        mad = (v - med).abs().median() * 1.4826
        return float(med - k * mad), float(med + k * mad)
    raise ValidationFailed(f"Unknown outlier method '{method}'")


def _condition_mask(df: pd.DataFrame, params: Dict) -> pd.Series:
    col = _require_col(df, params)
    op = params.get("operator")
    val = params.get("value")
    s = df[col]
    if op == "is_null":
        return s.isna()
    if op == "not_null":
        return s.notna()
    if op in ("in", "not_in"):
        if not isinstance(val, list):
            raise ValidationFailed("'value' must be a list for in / not_in")
        m = s.astype(str).isin([str(v) for v in val]) & s.notna()
        return m if op == "in" else (~m & s.notna())
    if op == "between":
        if not isinstance(val, list) or len(val) != 2:
            raise ValidationFailed("'value' must be [low, high] for between")
        lo, hi = val
        sv = _coerce_like(s, lo), _coerce_like(s, hi)
        return s.notna() & (s >= sv[0]) & (s <= sv[1])
    if op == "outside":
        if not isinstance(val, list) or len(val) != 2:
            raise ValidationFailed("'value' must be [low, high] for outside")
        lo, hi = _coerce_like(s, val[0]), _coerce_like(s, val[1])
        return s.notna() & ((s < lo) | (s > hi))
    if op == "matches":
        from core.security import check_user_regex
        rx = check_user_regex(str(val))
        return s.astype(str).str.slice(0, 1000).str.contains(rx, regex=True, na=False) & s.notna()
    ops: Dict[str, Callable] = {"==": lambda a, b: a == b, "!=": lambda a, b: a != b, ">": lambda a, b: a > b,
                                "<": lambda a, b: a < b, ">=": lambda a, b: a >= b, "<=": lambda a, b: a <= b}
    if op not in ops:
        raise ValidationFailed(f"Unknown operator '{op}'")
    v = _coerce_like(s, val)
    try:
        return s.notna() & ops[op](s, v).fillna(False).astype(bool)
    except TypeError:
        raise ValidationFailed(f"Cannot compare column '{col}' ({s.dtype}) with {val!r}")


def _coerce_like(s: pd.Series, v: Any) -> Any:
    if pd.api.types.is_datetime64_any_dtype(s):
        return pd.Timestamp(v)
    if pd.api.types.is_numeric_dtype(s) and not pd.api.types.is_bool_dtype(s):
        try:
            return float(v)
        except (TypeError, ValueError):
            raise ValidationFailed(f"Expected a number for column '{s.name}', got {v!r}")
    return v if not isinstance(v, (int, float)) or is_text_dtype(s) is False else str(v)


# ─────────────────────────────────────────────────────────────────────────────
# Operations: fn(df, params, fitted) -> (df, fitted, notes)
# ─────────────────────────────────────────────────────────────────────────────

def op_drop_duplicates(df, p, fitted):
    subset = _require_cols(df, p.get("subset"), "subset") or None
    keep = p.get("keep", "first")
    if keep not in ("first", "last"):
        raise ValidationFailed("keep must be 'first' or 'last'")
    return df.drop_duplicates(subset=subset, keep=keep), {}, []


def op_drop_empty_rows(df, p, fitted):
    return df[~df.isna().all(axis=1)], {}, []


def op_drop_rows_missing(df, p, fitted):
    cols = _require_cols(df, p.get("columns"), allow_empty=True) or list(df.columns)
    how = p.get("how", "any")
    if how not in ("any", "all"):
        raise ValidationFailed("how must be 'any' or 'all'")
    return df.dropna(subset=cols, how=how), {}, []


def op_drop_columns(df, p, fitted):
    cols = _require_cols(df, p.get("columns"))
    return df.drop(columns=cols), {}, []


def op_rename_column(df, p, fitted):
    col = _require_col(df, p)
    new = str(p.get("new_name", "")).strip()
    if not new:
        raise ValidationFailed("new_name is required")
    if new in df.columns and new != col:
        raise ValidationFailed(f"A column named '{new}' already exists")
    return df.rename(columns={col: new}), {}, []


def op_impute(df, p, fitted):
    cols = _require_cols(df, p.get("columns") or ([p["column"]] if p.get("column") else None))
    strategy = p.get("strategy", "median")
    add_indicator = bool(p.get("add_indicator", False))
    df = df.copy()
    fitted = dict(fitted or {})
    notes = []
    if strategy == "knn":
        num = [c for c in cols if pd.api.types.is_numeric_dtype(df[c]) and not pd.api.types.is_bool_dtype(df[c])]
        if len(num) != len(cols):
            raise ValidationFailed("KNN imputation supports numeric columns only")
        from sklearn.impute import KNNImputer
        others = [c for c in df.columns if c not in cols and pd.api.types.is_numeric_dtype(df[c])
                  and not pd.api.types.is_bool_dtype(df[c])]
        feats = cols + others[:20]
        if len(df) > 50_000:
            raise ValidationFailed("KNN imputation is limited to 50,000 rows; use median imputation instead")
        k = int(p.get("k", 5))
        if add_indicator:
            for c in cols:
                df[f"{c}_was_missing"] = df[c].isna()
        X = df[feats].astype(float)
        mu, sd = X.mean(), X.std(ddof=0).replace(0, 1)
        imp = KNNImputer(n_neighbors=max(1, min(k, 50)))
        filled = imp.fit_transform(((X - mu) / sd).to_numpy())
        filled = pd.DataFrame(filled, index=df.index, columns=feats) * sd + mu
        for c in cols:
            df[c] = df[c].where(df[c].notna(), filled[c])
        notes.append("KNN imputation uses other numeric columns; it is not re-applied to new data at prediction time.")
        return df, {"strategy": "knn"}, notes
    for c in cols:
        s = df[c]
        if add_indicator:
            df[f"{c}_was_missing"] = s.isna()
        if strategy in ("ffill", "bfill"):
            df[c] = s.ffill() if strategy == "ffill" else s.bfill()
            continue
        if strategy == "interpolate":
            if not pd.api.types.is_numeric_dtype(s):
                raise ValidationFailed("Interpolation needs a numeric column")
            df[c] = s.interpolate(limit_direction="both")
            continue
        if c in fitted:
            val = fitted[c]
        elif strategy == "mean":
            if not pd.api.types.is_numeric_dtype(s) or pd.api.types.is_bool_dtype(s):
                raise ValidationFailed(f"Mean imputation needs a numeric column ('{c}' is {s.dtype})")
            val = float(s.mean())
        elif strategy == "median":
            if not pd.api.types.is_numeric_dtype(s) or pd.api.types.is_bool_dtype(s):
                raise ValidationFailed(f"Median imputation needs a numeric column ('{c}' is {s.dtype})")
            val = float(s.median())
        elif strategy == "mode":
            m = s.mode(dropna=True)
            if m.empty:
                raise ValidationFailed(f"Column '{c}' has no values to compute a mode")
            val = m.iloc[0]
            if isinstance(val, pd.Timestamp):
                val = val.isoformat()
        elif strategy == "constant":
            if "value" not in p:
                raise ValidationFailed("'value' is required for constant imputation")
            val = p["value"]
        else:
            raise ValidationFailed(f"Unknown imputation strategy '{strategy}'")
        fitted[c] = to_jsonable(val)
        fill = val
        if pd.api.types.is_datetime64_any_dtype(s) and fill is not None:
            fill = pd.Timestamp(fill)
        elif pd.api.types.is_integer_dtype(s) and isinstance(fill, float) and not float(fill).is_integer():
            df[c] = s.astype(float)
        if pd.api.types.is_numeric_dtype(s) and not pd.api.types.is_bool_dtype(s) and isinstance(fill, str):
            try:
                fill = float(fill)
            except ValueError:
                raise ValidationFailed(f"Constant '{fill}' is not numeric but column '{c}' is")
        df[c] = df[c].fillna(fill)
    if strategy in ("mean", "median", "mode", "constant"):
        notes.append("Imputation before the train/test split slightly leaks information into evaluation. "
                     "The AutoML pipeline imputes inside cross-validation, so you can also leave missing values as-is.")
    return df, fitted, notes


def op_convert_type(df, p, fitted):
    col = _require_col(df, p)
    to = p.get("to")
    df = df.copy()
    s = df[col]
    notes: List[str] = []
    fitted = dict(fitted or {})
    if to == "numeric":
        if pd.api.types.is_numeric_dtype(s) and not pd.api.types.is_bool_dtype(s):
            return df, fitted, ["Column is already numeric"]
        dec = fitted.get("decimal") or p.get("decimal", "auto")
        if dec == "auto":
            smp = s.dropna().astype(str).head(2000)
            eu = smp.str.match(r"^-?\d{1,3}(\.\d{3})+(,\d+)?$|^-?\d+,\d+$").mean() if len(smp) else 0
            dec = "," if eu > 0.5 else "."
        fitted["decimal"] = dec
        out = _clean_numeric_text(s, dec)
        if s.dropna().astype(str).str.contains("%").any():
            notes.append("Percent signs were removed; values stay on the 0–100 scale.")
        df[col] = out
    elif to == "integer":
        out = _clean_numeric_text(s) if is_text_dtype(s) else pd.to_numeric(s, errors="coerce")
        frac = out.dropna() % 1 != 0
        if frac.any():
            notes.append(f"{int(frac.sum())} value(s) had decimals and were rounded.")
        df[col] = out.round().astype("Int64")
    elif to == "datetime":
        if pd.api.types.is_datetime64_any_dtype(s):
            return df, fitted, ["Column is already a datetime"]
        dayfirst = fitted.get("dayfirst", p.get("dayfirst"))
        if dayfirst is None:
            dayfirst = _auto_dayfirst(s)
        fitted["dayfirst"] = bool(dayfirst)
        fmt = p.get("format") or "mixed"
        t = s.astype("string").str.strip()
        t = t.mask(t.str.lower().isin(NULL_TOKENS))
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            out = pd.to_datetime(t, errors="coerce", dayfirst=bool(dayfirst), format=fmt)
        if getattr(out.dt, "tz", None) is not None:
            out = out.dt.tz_convert("UTC").dt.tz_localize(None)
        df[col] = out
        notes.append(f"Parsed with dayfirst={bool(dayfirst)}.")
    elif to == "text":
        df[col] = s.map(lambda v: v if pd.isna(v) else (v.isoformat() if isinstance(v, pd.Timestamp) else str(v))).astype(object)
    elif to == "category":
        df[col] = s.map(lambda v: v if pd.isna(v) else str(v)).astype(object)
        notes.append("Stored as text; it will be treated as a categorical dimension.")
    elif to == "boolean":
        t = s.astype("string").str.strip().str.lower()
        mapping = {"true": True, "t": True, "yes": True, "y": True, "1": True, "1.0": True,
                   "false": False, "f": False, "no": False, "n": False, "0": False, "0.0": False}
        df[col] = t.map(mapping).astype("boolean")
    else:
        raise ValidationFailed("'to' must be one of numeric, integer, datetime, text, category, boolean")
    return df, fitted, notes


def op_trim_whitespace(df, p, fitted):
    cols = _require_cols(df, p.get("columns"), allow_empty=True) or _text_cols(df)
    collapse = bool(p.get("collapse_internal", True))
    df = df.copy()
    for c in cols:
        if not is_text_dtype(df[c]):
            continue
        s = df[c].astype("string").str.strip()
        if collapse:
            s = s.str.replace(r"\s+", " ", regex=True)
        df[c] = _restore_text(s, df[c])
    return df, {}, []


def op_standardize_case(df, p, fitted):
    cols = _require_cols(df, p.get("columns") or ([p["column"]] if p.get("column") else None))
    case = p.get("case", "lower")
    fn = {"lower": "lower", "upper": "upper", "title": "title"}.get(case)
    if not fn:
        raise ValidationFailed("case must be lower, upper or title")
    df = df.copy()
    for c in cols:
        if is_text_dtype(df[c]):
            df[c] = _restore_text(getattr(df[c].astype("string").str, fn)(), df[c])
    return df, {}, []


def op_map_values(df, p, fitted):
    col = _require_col(df, p)
    mapping = p.get("mapping")
    if not isinstance(mapping, dict) or not mapping:
        raise ValidationFailed("'mapping' must be a non-empty object of {old: new}")
    if len(mapping) > 5000:
        raise ValidationFailed("Mapping is too large (max 5000 entries)")
    df = df.copy()
    s = df[col]
    if is_text_dtype(s):
        df[col] = _restore_text(s.astype("string").replace({str(k): v for k, v in mapping.items()}), s)
    else:
        conv = {}
        for k, v in mapping.items():
            try:
                conv[type(s.dropna().iloc[0])(k) if len(s.dropna()) else k] = v
            except Exception:
                conv[k] = v
        df[col] = s.replace(conv)
    return df, {}, []


def op_replace_null_tokens(df, p, fitted):
    cols = _require_cols(df, p.get("columns"), allow_empty=True) or _text_cols(df)
    tokens = p.get("tokens") or NULL_TOKENS
    tokens = {str(t).strip().lower() for t in tokens}
    df = df.copy()
    for c in cols:
        if not is_text_dtype(df[c]):
            continue
        s = df[c]
        key = s.astype("string").str.strip().str.lower()
        df[c] = s.mask(key.isin(tokens))
    return df, {}, []


def op_clip_outliers(df, p, fitted):
    col = _require_col(df, p)
    s = df[col]
    if not pd.api.types.is_numeric_dtype(s) or pd.api.types.is_bool_dtype(s):
        raise ValidationFailed("Clipping needs a numeric column")
    fitted = dict(fitted or {})
    if "lower" not in fitted:
        lo, hi = _bounds(s, p.get("method", "iqr"), float(p.get("k", 1.5)),
                         float(p.get("lower_pct", 0.01)), float(p.get("upper_pct", 0.99)))
        fitted = {"lower": lo, "upper": hi}
    df = df.copy()
    if pd.api.types.is_integer_dtype(s):
        df[col] = s.astype(float)
    df[col] = df[col].clip(lower=fitted["lower"], upper=fitted["upper"])
    return df, fitted, [f"Values outside [{fitted['lower']:.4g}, {fitted['upper']:.4g}] were capped (winsorised)."]


def op_remove_outliers(df, p, fitted):
    col = _require_col(df, p)
    s = df[col]
    if not pd.api.types.is_numeric_dtype(s) or pd.api.types.is_bool_dtype(s):
        raise ValidationFailed("Outlier removal needs a numeric column")
    fitted = dict(fitted or {})
    if "lower" not in fitted:
        method = p.get("method", "iqr")
        k = float(p.get("k", 3.0 if method in ("iqr",) else 3.5))
        lo, hi = _bounds(s, method, k)
        fitted = {"lower": lo, "upper": hi}
    keep = s.isna() | ((s >= fitted["lower"]) & (s <= fitted["upper"]))
    return df[keep], fitted, [f"Rows with '{col}' outside [{fitted['lower']:.4g}, {fitted['upper']:.4g}] were removed."]


def op_remove_rows_where(df, p, fitted):
    m = _condition_mask(df, p)
    return df[~m], {}, []


def op_nullify_where(df, p, fitted):
    col = _require_col(df, p)
    m = _condition_mask(df, p)
    df = df.copy()
    df[col] = df[col].mask(m)
    return df, {}, []


def op_split_number_unit(df, p, fitted):
    col = _require_col(df, p)
    s = df[col]
    if not is_text_dtype(s):
        raise ValidationFailed("Column must be text, e.g. '12 kg'")
    m = s.astype("string").str.strip().str.extract(r"^(-?[\d.,]+)\s*([A-Za-z%°/]+)?$")
    df = df.copy()
    num = _clean_numeric_text(m[0])
    unit_col = p.get("unit_column") or f"{col}_unit"
    if unit_col in df.columns and unit_col != col:
        raise ValidationFailed(f"Column '{unit_col}' already exists")
    df[col] = num
    df[unit_col] = _restore_text(m[1].str.lower(), pd.Series(dtype=object))
    units = df[unit_col].dropna().unique().tolist()
    notes = [f"Units found: {', '.join(map(str, units[:10]))}."]
    if len(units) > 1:
        notes.append("Several units were found. Values are not converted to a common unit automatically; "
                     "convert them explicitly before comparing values.")
    return df, {"unit_column": unit_col}, notes


OPS: Dict[str, Dict[str, Any]] = {
    "drop_duplicates":    {"fn": op_drop_duplicates, "label": "Remove duplicate rows", "inference": False},
    "drop_empty_rows":    {"fn": op_drop_empty_rows, "label": "Remove empty rows", "inference": False},
    "drop_rows_missing":  {"fn": op_drop_rows_missing, "label": "Remove rows with missing values", "inference": False},
    "drop_columns":       {"fn": op_drop_columns, "label": "Drop columns", "inference": True},
    "rename_column":      {"fn": op_rename_column, "label": "Rename column", "inference": True},
    "impute":             {"fn": op_impute, "label": "Fill missing values", "inference": True},
    "convert_type":       {"fn": op_convert_type, "label": "Convert data type", "inference": True},
    "trim_whitespace":    {"fn": op_trim_whitespace, "label": "Trim whitespace", "inference": True},
    "standardize_case":   {"fn": op_standardize_case, "label": "Standardize letter case", "inference": True},
    "map_values":         {"fn": op_map_values, "label": "Merge / relabel values", "inference": True},
    "replace_null_tokens": {"fn": op_replace_null_tokens, "label": "Treat placeholder text as missing", "inference": True},
    "clip_outliers":      {"fn": op_clip_outliers, "label": "Cap outliers (winsorise)", "inference": True},
    "remove_outliers":    {"fn": op_remove_outliers, "label": "Remove outlier rows", "inference": False},
    "remove_rows_where":  {"fn": op_remove_rows_where, "label": "Remove rows matching a condition", "inference": False},
    "nullify_where":      {"fn": op_nullify_where, "label": "Mark values matching a condition as missing", "inference": True},
    "split_number_unit":  {"fn": op_split_number_unit, "label": "Split number and unit", "inference": True},
}

# Strategies that depend on row order / other rows and must not run on prediction inputs.
_NON_INFERENCE_IMPUTE = {"ffill", "bfill", "interpolate", "knn"}


# ─────────────────────────────────────────────────────────────────────────────
# Impact measurement
# ─────────────────────────────────────────────────────────────────────────────

def measure_impact(before: pd.DataFrame, after: pd.DataFrame, focus: Optional[List[str]] = None) -> Dict[str, Any]:
    removed_rows_idx = before.index.difference(after.index)
    removed_cols = [c for c in before.columns if c not in after.columns]
    added_cols = [c for c in after.columns if c not in before.columns]
    common_cols = [c for c in before.columns if c in after.columns]
    kept_idx = before.index.intersection(after.index)

    values_lost = 0
    if len(removed_rows_idx):
        values_lost += int(before.loc[removed_rows_idx].notna().sum().sum())
    if removed_cols:
        values_lost += int(before.loc[kept_idx, removed_cols].notna().sum().sum())

    cells_changed = 0
    nullified = 0
    filled = 0
    changed_by_col: Dict[str, int] = {}
    cols_to_check = [c for c in (focus or common_cols) if c in common_cols]
    for c in cols_to_check:
        b = before.loc[kept_idx, c]
        a = after.loc[kept_idx, c]
        bn, an = b.isna().to_numpy(), a.isna().to_numpy()
        nul = int((~bn & an).sum())
        fil = int((bn & ~an).sum())
        both = ~bn & ~an
        try:
            if b.dtype != a.dtype:
                bv = b.astype(str).to_numpy()
                av = a.astype(str).to_numpy()
                if pd.api.types.is_numeric_dtype(a) and not pd.api.types.is_bool_dtype(a):
                    bnum = pd.to_numeric(b, errors="coerce").to_numpy(dtype=float, na_value=np.nan)
                    anum = a.to_numpy(dtype=float, na_value=np.nan)
                    diff = both & ~np.isclose(bnum, anum, equal_nan=True)
                elif pd.api.types.is_datetime64_any_dtype(a):
                    diff = np.zeros(len(b), dtype=bool)  # representation change only
                else:
                    diff = both & (bv != av)
            else:
                diff = both & (b.to_numpy() != a.to_numpy())
            mod = int(np.asarray(diff, dtype=bool).sum())
        except Exception:
            mod = 0
        nullified += nul
        filled += fil
        n = mod + nul + fil
        if n:
            changed_by_col[str(c)] = n
        cells_changed += mod
    values_lost += nullified
    requires_confirmation = bool(len(removed_rows_idx) or removed_cols or nullified)
    return {
        "rows_before": int(len(before)), "rows_after": int(len(after)),
        "rows_removed": int(len(removed_rows_idx)),
        "columns_removed": [str(c) for c in removed_cols], "columns_added": [str(c) for c in added_cols],
        "cells_modified": int(cells_changed), "values_filled": int(filled), "values_nullified": int(nullified),
        "non_null_values_lost": int(values_lost),
        "changed_by_column": changed_by_col,
        "requires_confirmation": requires_confirmation,
    }


def _sample_changes(before: pd.DataFrame, after: pd.DataFrame, impact: Dict, n: int = 8) -> Dict[str, Any]:
    out: Dict[str, Any] = {"changed_rows": [], "removed_rows": [], "columns": []}
    cols = list(impact["changed_by_column"].keys())[:6]
    kept = before.index.intersection(after.index)
    if cols:
        mask = np.zeros(len(kept), dtype=bool)
        for c in cols:
            b, a = before.loc[kept, c], after.loc[kept, c]
            both_na = (b.isna() & a.isna()).to_numpy()
            mask |= (b.astype(str).to_numpy() != a.astype(str).to_numpy()) & ~both_na
        idx = kept[mask][:n]
        out["columns"] = cols
        for i in idx:
            out["changed_rows"].append({
                "row": to_jsonable(i),
                "before": {c: to_jsonable(before.at[i, c]) for c in cols},
                "after": {c: to_jsonable(after.at[i, c]) for c in cols},
            })
    removed = before.index.difference(after.index)[:n]
    if len(removed):
        show = list(before.columns[:8])
        for i in removed:
            out["removed_rows"].append({"row": to_jsonable(i), "values": {c: to_jsonable(before.at[i, c]) for c in show}})
    return out


# ─────────────────────────────────────────────────────────────────────────────
# Public API
# ─────────────────────────────────────────────────────────────────────────────

def _run(df: pd.DataFrame, op: str, params: Dict, fitted: Optional[Dict] = None):
    if op not in OPS:
        raise ValidationFailed(f"Unknown cleaning operation '{op}'")
    if not isinstance(params, dict):
        raise ValidationFailed("params must be an object")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return OPS[op]["fn"](df, params, fitted or {})


def preview_op(df: pd.DataFrame, op: str, params: Dict) -> Dict[str, Any]:
    after, fitted, notes = _run(df, op, params)
    impact = measure_impact(df, after)
    return {
        "op": op, "label": OPS[op]["label"], "params": params, "fitted": to_jsonable(fitted),
        "impact": impact, "notes": notes, "sample": _sample_changes(df, after, impact),
        "applies_to_new_data": _applies_to_inference(op, params),
    }


def apply_op(df: pd.DataFrame, op: str, params: Dict) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    after, fitted, notes = _run(df, op, params)
    impact = measure_impact(df, after)
    record = {
        "id": new_id(12), "op": op, "label": OPS[op]["label"], "params": to_jsonable(params),
        "fitted": to_jsonable(fitted), "impact": impact, "notes": notes, "created_at": time.time(),
        "applies_to_new_data": _applies_to_inference(op, params),
    }
    return after, record


def _applies_to_inference(op: str, params: Dict) -> bool:
    if not OPS[op]["inference"]:
        return False
    if op == "impute" and params.get("strategy") in _NON_INFERENCE_IMPUTE:
        return False
    return True


def replay_ops(original: pd.DataFrame, ops: List[Dict]) -> pd.DataFrame:
    df = original
    for rec in ops:
        df, _, _ = _run(df, rec["op"], rec.get("params", {}), rec.get("fitted") or {})
    return df


def apply_inference_ops(df: pd.DataFrame, ops: List[Dict]) -> Tuple[pd.DataFrame, List[str]]:
    """Apply column-level transformations to new data (prediction inputs).

    Row-removing operations are skipped by design; operations whose columns are
    absent are skipped with a note."""
    notes: List[str] = []
    for rec in ops:
        if not rec.get("applies_to_new_data"):
            continue
        try:
            df, _, _ = _run(df, rec["op"], rec.get("params", {}), rec.get("fitted") or {})
        except ValidationFailed as e:
            notes.append(f"Skipped '{rec.get('label')}' on new data: {e.message}")
    return df, notes


# ─────────────────────────────────────────────────────────────────────────────
# Issue detection → cleaning suggestions
# ─────────────────────────────────────────────────────────────────────────────

def _normalize_label(v: str) -> str:
    return re.sub(r"[^0-9a-z]+", "", str(v).lower())


def detect_issues(df: pd.DataFrame, schema: Optional[List[Dict]] = None) -> List[Dict[str, Any]]:
    schema = schema or infer_schema(df)
    sch = {c["name"]: c for c in schema}
    n = len(df)
    out: List[Dict[str, Any]] = []

    def add(kind, severity, title, description, evidence, op=None, alternatives=None, columns=None, safe=False,
            consequence=None):
        out.append({
            "id": f"{kind}:{'|'.join(columns or [])}", "issue": kind, "severity": severity, "title": title,
            "description": description, "evidence": to_jsonable(evidence), "columns": columns or [],
            "recommended": op, "alternatives": alternatives or [], "safe": safe,
            "consequence": consequence,
        })

    empty_rows = int(df.isna().all(axis=1).sum())
    if empty_rows:
        add("empty_rows", "medium", f"{empty_rows} completely empty row(s)",
            "Rows with no values add nothing to the analysis and distort counts.",
            {"rows": empty_rows}, {"op": "drop_empty_rows", "params": {}}, safe=True,
            consequence="Removes rows that contain no values; no information is lost.")

    dups = int(df.duplicated().sum())
    if dups:
        add("duplicates", "high" if dups / max(n, 1) > 0.01 else "medium",
            f"{dups:,} duplicate row(s) ({dups / max(n, 1):.1%})",
            "Exact duplicates over-weight some records in statistics and can leak between train and test sets.",
            {"duplicate_rows": dups, "pct": dups / max(n, 1) * 100},
            {"op": "drop_duplicates", "params": {"keep": "first"}},
            consequence="Keeps the first occurrence of each duplicated row. Verify duplicates are not legitimate "
                        "repeated events (e.g. two identical orders) before removing them.")

    for c in df.columns:
        info = sch.get(c, {})
        s = df[c]
        lt = info.get("logical_type")
        miss = int(s.isna().sum())
        pct = miss / n * 100 if n else 0

        if lt == "empty":
            add("empty_column", "medium", f"'{c}' is completely empty", "The column has no values.",
                {"missing_pct": 100}, {"op": "drop_columns", "params": {"columns": [c]}}, columns=[c], safe=True,
                consequence="Removes a column that holds no data.")
            continue
        if lt == "constant":
            add("constant_column", "low", f"'{c}' has a single value",
                "A constant column carries no information for comparisons or models.",
                {"value": to_jsonable(s.dropna().iloc[0]), "missing": miss},
                {"op": "drop_columns", "params": {"columns": [c]}}, columns=[c],
                consequence="Removes the column. Keep it if the constant value itself is meaningful documentation.")
            continue
        if lt == "identifier":
            add("identifier", "info", f"'{c}' looks like an identifier",
                "Identifiers are unique per row. They are excluded from statistics and models automatically "
                "(they would let a model memorise rows instead of learning patterns).",
                {"unique_ratio": info.get("unique_ratio")}, None, columns=[c])

        if is_text_dtype(s):
            nn = s.dropna()
            if len(nn):
                st = nn.astype("string")
                stripped = st.str.strip()
                tok = int(stripped.str.lower().isin(NULL_TOKENS).sum())
                if tok:
                    examples = stripped[stripped.str.lower().isin(NULL_TOKENS)].value_counts().head(5).index.tolist()
                    add("null_tokens", "medium", f"'{c}' uses placeholder text for missing values",
                        f"{tok:,} value(s) such as {', '.join(repr(e) for e in examples)} represent missing data "
                        "but are counted as real categories.",
                        {"count": tok, "examples": examples},
                        {"op": "replace_null_tokens", "params": {"columns": [c]}}, columns=[c], safe=True,
                        consequence="These values become missing (NaN) so they are counted as missing.")
                ws = int((st != stripped).sum() + stripped.str.contains(r"\s{2,}", regex=True).sum())
                if ws:
                    add("whitespace", "low", f"'{c}' has extra whitespace",
                        "Leading/trailing or repeated spaces make identical labels look different.",
                        {"values": ws}, {"op": "trim_whitespace", "params": {"columns": [c]}}, columns=[c], safe=True,
                        consequence="Spaces are trimmed; text content is otherwise unchanged.")
                if lt in ("categorical", "boolean") and info.get("n_unique", 0) <= 500:
                    vc = stripped.value_counts()
                    groups: Dict[str, List[str]] = {}
                    for v in vc.index:
                        groups.setdefault(_normalize_label(v), []).append(v)
                    variants = {k: g for k, g in groups.items() if len(g) > 1 and k}
                    if variants:
                        mapping = {}
                        for g in variants.values():
                            canon = max(g, key=lambda v: vc.get(v, 0))
                            for v in g:
                                if v != canon:
                                    mapping[v] = canon
                        ex = list(variants.values())[:3]
                        add("inconsistent_labels", "medium", f"'{c}' has inconsistent labels",
                            "Some categories differ only by case, spacing or punctuation: "
                            + "; ".join(" / ".join(repr(x) for x in g) for g in ex) + ".",
                            {"groups": len(variants), "values_affected": int(sum(vc.get(k, 0) for k in mapping))},
                            {"op": "map_values", "params": {"column": c, "mapping": mapping}}, columns=[c],
                            alternatives=[{"op": "standardize_case", "params": {"columns": [c], "case": "title"}}],
                            consequence="Each variant is relabelled to its most frequent spelling.")
                # Units like "12 kg" / "3 lb"
                um = stripped.head(3000).str.extract(r"^-?[\d.,]+\s*([A-Za-z%°/]+)$")[0]
                if um.notna().mean() > 0.8:
                    units = um.dropna().str.lower().unique().tolist()
                    add("number_with_unit", "medium" if len(units) > 1 else "low",
                        f"'{c}' stores numbers with units ({', '.join(units[:5])})",
                        "Numbers stored together with units cannot be summed or compared."
                        + (" Several different units are mixed in this column." if len(units) > 1 else ""),
                        {"units": units[:10]}, {"op": "split_number_unit", "params": {"column": c}}, columns=[c],
                        consequence="Creates a numeric column and a separate unit column. Units are not converted.")
                elif "numeric_string" in info.get("hints", []):
                    parsed = _clean_numeric_text(s)
                    fail = int(s.notna().sum() - parsed.notna().sum() - stripped.str.lower().isin(NULL_TOKENS).sum())
                    add("numeric_as_text", "high" if fail == 0 else "medium",
                        f"'{c}' contains numbers stored as text",
                        "Currency symbols, thousands separators or percent signs prevent numeric analysis.",
                        {"convertible_pct": numeric_parse_rate(s) * 100, "would_become_missing": max(fail, 0)},
                        {"op": "convert_type", "params": {"column": c, "to": "numeric"}}, columns=[c],
                        safe=fail <= 0,
                        consequence=(f"{fail} value(s) cannot be read as numbers and would become missing."
                                     if fail > 0 else "All values convert without loss."))
                elif "datetime_string" in info.get("hints", []):
                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore")
                        parsed = pd.to_datetime(stripped.mask(stripped.str.lower().isin(NULL_TOKENS)),
                                                errors="coerce", format="mixed", dayfirst=_auto_dayfirst(s))
                    bad = stripped[parsed.isna() & ~stripped.str.lower().isin(NULL_TOKENS)]
                    add("dates_as_text", "high" if len(bad) == 0 else "medium", f"'{c}' contains dates stored as text",
                        "Dates stored as text cannot be used for time-series analysis or forecasting.",
                        {"invalid_dates": int(len(bad)), "invalid_examples": bad.head(5).tolist()},
                        {"op": "convert_type", "params": {"column": c, "to": "datetime"}}, columns=[c],
                        safe=len(bad) == 0,
                        consequence=(f"{len(bad)} value(s) are not valid dates and would become missing."
                                     if len(bad) else "All values convert without loss."))

        if pd.api.types.is_datetime64_any_dtype(s) and s.notna().any():
            now = pd.Timestamp.now()
            future = int((s > now + pd.Timedelta(days=366)).sum())
            ancient = int((s < pd.Timestamp("1900-01-01")).sum())
            if future or ancient:
                add("implausible_dates", "medium", f"'{c}' has implausible dates",
                    "Dates more than a year in the future or before 1900 are usually entry errors.",
                    {"future": future, "before_1900": ancient},
                    {"op": "nullify_where", "params": {"column": c, "operator": "outside",
                                                        "value": ["1900-01-01", str((now + pd.Timedelta(days=366)).date())]}},
                    alternatives=[{"op": "remove_rows_where", "params": {"column": c, "operator": "outside",
                                   "value": ["1900-01-01", str((now + pd.Timedelta(days=366)).date())]}}],
                    columns=[c], consequence="The implausible dates become missing values.")

        if lt == "numeric":
            sem = info.get("semantic_type")
            if sem in ("age", "quantity", "revenue") or re.search(r"(age|qty|quantity|price|count|units|days|duration)", str(c).lower()):
                neg = int((s < 0).sum())
                if neg and not re.search(r"(profit|margin|change|delta|balance|growth|return|diff)", str(c).lower()):
                    add("negative_values", "medium", f"'{c}' has {neg} negative value(s)",
                        "Negative values are unusual for this kind of measure and may be entry errors or returns.",
                        {"negative": neg, "min": to_jsonable(s.min())},
                        {"op": "nullify_where", "params": {"column": c, "operator": "<", "value": 0}},
                        alternatives=[{"op": "remove_rows_where", "params": {"column": c, "operator": "<", "value": 0}}],
                        columns=[c], consequence="Negative values become missing. Keep them if they are legitimate "
                                                 "(e.g. refunds).")
            v = s.dropna()
            if len(v) >= 20:
                q1, q3 = v.quantile(0.25), v.quantile(0.75)
                iqr = q3 - q1
                if iqr > 0:
                    ext = int(((v < q1 - 3 * iqr) | (v > q3 + 3 * iqr)).sum())
                    if ext:
                        add("outliers", "medium", f"'{c}' has {ext} extreme value(s)",
                            "Values beyond 3×IQR from the quartiles. They may be errors or genuine rare events — "
                            "investigate before changing them.",
                            {"extreme": ext, "pct": ext / len(v) * 100, "lower_fence": q1 - 3 * iqr,
                             "upper_fence": q3 + 3 * iqr, "max": v.max(), "min": v.min()},
                            None, alternatives=[
                                {"op": "clip_outliers", "params": {"column": c, "method": "iqr", "k": 3}},
                                {"op": "remove_outliers", "params": {"column": c, "method": "iqr", "k": 3}}],
                            columns=[c], consequence="No automatic action is recommended for outliers.")
            top_share = s.value_counts(normalize=True, dropna=True).iloc[0] if s.notna().any() else 0
            if top_share >= 0.99 and info.get("n_unique", 0) > 1:
                add("near_constant", "low", f"'{c}' is almost constant",
                    f"{top_share:.1%} of values are identical; the column carries little information.",
                    {"top_share_pct": top_share * 100}, None,
                    alternatives=[{"op": "drop_columns", "params": {"columns": [c]}}], columns=[c],
                    consequence="Rare values may still matter (e.g. a fraud flag); dropping is optional.")

        convert_first = bool(set(info.get("hints", [])) & {"numeric_string", "datetime_string"})
        if 0 < pct <= 60 and lt not in ("empty", "identifier") and not convert_first:
            if lt == "numeric":
                try:
                    skew = float(s.skew())
                except Exception:
                    skew = 0.0
                strat = "median" if abs(skew) > 1 else "mean"
                rec = {"op": "impute", "params": {"columns": [c], "strategy": strat}}
                alts = [{"op": "impute", "params": {"columns": [c], "strategy": "median" if strat == "mean" else "mean"}},
                        {"op": "impute", "params": {"columns": [c], "strategy": "knn"}},
                        {"op": "drop_rows_missing", "params": {"columns": [c]}}]
                why = (f"Skewness is {skew:.2f}, so the median is more robust than the mean."
                       if strat == "median" else f"Distribution is roughly symmetric (skew {skew:.2f}); the mean is suitable.")
            elif lt in ("categorical", "boolean", "text"):
                rec = {"op": "impute", "params": {"columns": [c], "strategy": "constant", "value": "Missing"}}
                alts = [{"op": "impute", "params": {"columns": [c], "strategy": "mode"}},
                        {"op": "drop_rows_missing", "params": {"columns": [c]}}]
                why = "Labelling missing values explicitly keeps the information that the value was missing."
            else:
                rec = None
                alts = [{"op": "drop_rows_missing", "params": {"columns": [c]}}]
                why = "Missing dates are usually best left missing or the rows removed."
            add("missing_values", "high" if pct > 20 else "medium" if pct > 5 else "low",
                f"'{c}' is {pct:.1f}% missing", f"{miss:,} missing value(s). {why} The model pipeline can also "
                "handle missing values on its own.", {"missing": miss, "missing_pct": pct}, rec, alternatives=alts,
                columns=[c], consequence="Filled values are estimates, not observations; this reduces apparent variance.")
        elif pct > 60:
            add("mostly_missing", "high", f"'{c}' is {pct:.1f}% missing",
                "With most values missing, imputation would mostly invent data.",
                {"missing": miss, "missing_pct": pct},
                {"op": "drop_columns", "params": {"columns": [c]}},
                alternatives=[{"op": "impute", "params": {"columns": [c], "strategy": "constant",
                                                          "value": 0 if lt == "numeric" else "Missing",
                                                          "add_indicator": True}}],
                columns=[c], consequence=f"Dropping removes {n - miss:,} observed value(s).")

        if lt == "categorical" and info.get("n_unique", 0) > 50:
            add("high_cardinality", "info", f"'{c}' has {info['n_unique']:,} distinct values",
                "Charts group rare categories into 'Other' and models cap one-hot features at the most frequent levels.",
                {"unique": info["n_unique"]}, None, columns=[c])

    order = {"high": 0, "medium": 1, "low": 2, "info": 3}
    out.sort(key=lambda x: (order.get(x["severity"], 9), x["issue"]))
    return out


# ─────────────────────────────────────────────────────────────────────────────
# v4 compatibility
# ─────────────────────────────────────────────────────────────────────────────

def clean_dataframe(df: pd.DataFrame) -> Tuple[pd.DataFrame, List[Dict[str, Any]]]:
    """v4 API: apply only *safe* suggestions and return (df, report).

    Unlike v4 this no longer drops columns, removes duplicates or imputes values
    silently; those require explicit approval in the cleaning workflow.
    """
    report: List[Dict[str, Any]] = []
    for sug in detect_issues(df):
        if sug.get("safe") and sug.get("recommended"):
            r = sug["recommended"]
            try:
                df, rec = apply_op(df, r["op"], r["params"])
                report.append({"action": r["op"], "detail": sug["title"] + " — " + (sug.get("consequence") or ""),
                               "severity": "info"})
            except ValidationFailed:
                continue
    return df, report
