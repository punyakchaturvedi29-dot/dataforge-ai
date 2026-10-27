"""
User-defined data-quality rules.

Rules are declarative JSON objects stored in the workspace and evaluated on
demand against the working dataset. Each evaluation returns the number and
share of violating rows plus a small sample, so users can see *which* records
break *which* rule. Regular expressions are screened against catastrophic
backtracking (see ``core.security.check_user_regex``).
"""
from __future__ import annotations

import re
from typing import Any, Dict, List

import pandas as pd

from core.errors import ValidationFailed
from core.jsonsafe import to_jsonable
from core.security import check_user_regex, new_id

PRESET_PATTERNS = {
    "email": r"^[^@\s]+@[^@\s]+\.[A-Za-z]{2,}$",
    "url": r"^https?://[^\s/$.?#].[^\s]*$",
    "phone": r"^\+?[0-9 ()\-]{7,20}$",
    "postal_code_us": r"^\d{5}(-\d{4})?$",
    "iso_date": r"^\d{4}-\d{2}-\d{2}$",
    "alphanumeric": r"^[A-Za-z0-9]+$",
}

RULE_TYPES = {
    "not_null": "Value must be present",
    "unique": "Values must be unique",
    "range": "Value must be within a numeric/date range",
    "allowed_values": "Value must be one of a list",
    "pattern": "Text must match a pattern",
    "max_missing_pct": "Missing share must not exceed a threshold",
    "compare_columns": "Relationship between two columns must hold",
    "max_length": "Text length must not exceed a limit",
}


def validate_rule(rule: Dict[str, Any], df: pd.DataFrame) -> Dict[str, Any]:
    if not isinstance(rule, dict):
        raise ValidationFailed("Rule must be an object")
    t = rule.get("type")
    if t not in RULE_TYPES:
        raise ValidationFailed(f"Unknown rule type '{t}'. Use one of: {', '.join(RULE_TYPES)}")
    col = rule.get("column")
    if col not in df.columns:
        raise ValidationFailed(f"Unknown column '{col}'")
    clean = {"id": rule.get("id") or new_id(10), "type": t, "column": col,
             "name": str(rule.get("name") or f"{RULE_TYPES[t]} ({col})")[:120],
             "severity": rule.get("severity") if rule.get("severity") in ("error", "warning") else "error"}
    if t == "range":
        lo, hi = rule.get("min"), rule.get("max")
        if lo is None and hi is None:
            raise ValidationFailed("Range rules need min and/or max")
        clean.update({"min": lo, "max": hi})
    elif t == "allowed_values":
        vals = rule.get("values")
        if not isinstance(vals, list) or not vals or len(vals) > 1000:
            raise ValidationFailed("allowed_values needs a list of 1–1000 values")
        clean["values"] = [str(v) for v in vals]
        clean["case_sensitive"] = bool(rule.get("case_sensitive", True))
    elif t == "pattern":
        preset = rule.get("preset")
        if preset:
            if preset not in PRESET_PATTERNS:
                raise ValidationFailed(f"Unknown preset '{preset}'")
            clean["preset"] = preset
            clean["pattern"] = PRESET_PATTERNS[preset]
        else:
            check_user_regex(rule.get("pattern"))
            clean["pattern"] = rule["pattern"]
    elif t == "max_missing_pct":
        try:
            clean["threshold"] = float(rule.get("threshold"))
        except (TypeError, ValueError):
            raise ValidationFailed("threshold must be a number (percent)")
    elif t == "compare_columns":
        other = rule.get("other_column")
        if other not in df.columns:
            raise ValidationFailed(f"Unknown column '{other}'")
        op = rule.get("operator")
        if op not in ("<", "<=", ">", ">=", "==", "!="):
            raise ValidationFailed("operator must be one of < <= > >= == !=")
        clean.update({"other_column": other, "operator": op})
    elif t == "max_length":
        try:
            clean["max_length"] = int(rule.get("max_length"))
        except (TypeError, ValueError):
            raise ValidationFailed("max_length must be an integer")
    return clean


def _violations(rule: Dict[str, Any], df: pd.DataFrame) -> pd.Series:
    s = df[rule["column"]]
    t = rule["type"]
    if t == "not_null":
        return s.isna()
    if t == "unique":
        return s.notna() & s.duplicated(keep=False)
    if t == "range":
        if pd.api.types.is_datetime64_any_dtype(s):
            lo = pd.Timestamp(rule["min"]) if rule.get("min") is not None else None
            hi = pd.Timestamp(rule["max"]) if rule.get("max") is not None else None
            v = s
        else:
            v = pd.to_numeric(s, errors="coerce")
            lo = float(rule["min"]) if rule.get("min") is not None else None
            hi = float(rule["max"]) if rule.get("max") is not None else None
        bad = pd.Series(False, index=s.index)
        if lo is not None:
            bad |= v < lo
        if hi is not None:
            bad |= v > hi
        if not pd.api.types.is_datetime64_any_dtype(s):
            bad |= s.notna() & v.isna()  # non-numeric values violate a numeric range
        return bad.fillna(False)
    if t == "allowed_values":
        allowed = set(rule["values"]) if rule.get("case_sensitive", True) else {v.lower() for v in rule["values"]}
        sv = s.astype("string")
        if not rule.get("case_sensitive", True):
            sv = sv.str.lower()
        return (s.notna() & ~sv.isin(allowed)).fillna(False)
    if t == "pattern":
        rx = re.compile(rule["pattern"])
        sv = s.astype("string").str.slice(0, 1000)
        return (s.notna() & ~sv.str.fullmatch(rx).fillna(False)).astype(bool)
    if t == "max_length":
        return (s.notna() & (s.astype("string").str.len() > rule["max_length"])).fillna(False)
    if t == "compare_columns":
        a, b = s, df[rule["other_column"]]
        ops = {"<": a < b, "<=": a <= b, ">": a > b, ">=": a >= b, "==": a == b, "!=": a != b}
        ok = ops[rule["operator"]]
        return (a.notna() & b.notna() & ~ok.fillna(False)).astype(bool)
    return pd.Series(False, index=s.index)


def evaluate_rules(rules: List[Dict[str, Any]], df: pd.DataFrame, sample: int = 10) -> Dict[str, Any]:
    results = []
    n = len(df)
    any_bad = pd.Series(False, index=df.index)
    for r in rules:
        if r.get("column") not in df.columns or (r.get("other_column") and r["other_column"] not in df.columns):
            results.append({"rule": r, "status": "skipped", "message": "Column no longer exists in the dataset"})
            continue
        if r["type"] == "max_missing_pct":
            pct = float(df[r["column"]].isna().mean() * 100)
            passed = pct <= r["threshold"]
            results.append({"rule": r, "status": "pass" if passed else "fail", "violations": 0 if passed else 1,
                            "violation_pct": pct, "message": f"{pct:.1f}% missing (limit {r['threshold']}%)",
                            "sample": []})
            continue
        try:
            bad = _violations(r, df)
        except Exception as e:  # e.g. comparing incompatible types
            results.append({"rule": r, "status": "error", "message": f"Rule could not be evaluated: {type(e).__name__}"})
            continue
        k = int(bad.sum())
        any_bad |= bad
        cols = [r["column"]] + ([r["other_column"]] if r.get("other_column") else [])
        smp = df.loc[bad, cols].head(sample)
        results.append({
            "rule": r, "status": "pass" if k == 0 else "fail", "violations": k,
            "violation_pct": k / n * 100 if n else 0,
            "sample": [{"row": to_jsonable(i), **{c: to_jsonable(v) for c, v in row.items()}}
                       for i, row in smp.to_dict(orient="index").items()],
        })
    passed = sum(1 for r in results if r["status"] == "pass")
    return to_jsonable({"results": results, "rules_passed": passed, "rules_total": len(results),
                        "rows_with_violations": int(any_bad.sum()), "rows": n})
