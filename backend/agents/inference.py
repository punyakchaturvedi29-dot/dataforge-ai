"""
Prediction with schema validation.

New data goes through the same column-level cleaning steps that were applied
to the training data (recorded in the model metadata), then is validated
against the model's input schema before the stored pipeline (preprocessing +
model) produces predictions.
"""
from __future__ import annotations

import warnings
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from core.errors import ValidationFailed
from core.jsonsafe import to_jsonable

from .cleaner import apply_inference_ops


def prepare_input(df_in: pd.DataFrame, meta: Dict[str, Any], strict: bool = False,
                  allow_missing_features: bool = False) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    result = meta["result"]
    schema = result["input_schema"]
    report: Dict[str, Any] = {"missing_features": [], "unexpected_columns": [], "invalid_values": [],
                              "unseen_categories": [], "out_of_range": [], "cleaning_notes": [], "row_issues": {}}
    df = df_in.copy()
    df.columns = [str(c).strip() for c in df.columns]
    ops = meta.get("cleaning_ops") or []
    if ops:
        df, notes = apply_inference_ops(df, ops)
        report["cleaning_notes"] = notes
    names = schema["order"]
    target = result["target"]
    missing = [n for n in names if n not in df.columns]
    if missing:
        report["missing_features"] = missing
        if not allow_missing_features:
            raise ValidationFailed(
                f"Missing required feature(s): {', '.join(missing[:10])}" + ("…" if len(missing) > 10 else ""),
                code="missing_features", details={"missing_features": missing})
        for n in missing:
            df[n] = np.nan
    report["unexpected_columns"] = [c for c in df.columns if c not in names and c != target][:50]

    X = pd.DataFrame(index=df.index)
    issues: Dict[Any, List[str]] = {}

    def flag(i, msg):
        issues.setdefault(i, []).append(msg)

    for f in schema["features"]:
        n, kind = f["name"], f["kind"]
        s = df[n]
        if kind == "numeric":
            conv = pd.to_numeric(s, errors="coerce")
            bad = s.notna() & conv.isna()
            if bad.any():
                ex = s[bad].astype(str).head(3).tolist()
                report["invalid_values"].append({"feature": n, "count": int(bad.sum()), "examples": ex,
                                                 "expected": "number"})
                for i in s.index[bad][:1000]:
                    flag(i, f"'{n}' is not a number")
            lo, hi = f.get("min"), f.get("max")
            if lo is not None and hi is not None:
                span = hi - lo
                oor = conv.notna() & ((conv < lo - 0.1 * span) | (conv > hi + 0.1 * span))
                if oor.any():
                    report["out_of_range"].append({"feature": n, "count": int(oor.sum()),
                                                   "training_range": [lo, hi]})
                    for i in s.index[oor][:1000]:
                        flag(i, f"'{n}' is outside the training range (extrapolation)")
            X[n] = conv.astype(float)
        elif kind == "datetime":
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                conv = pd.to_datetime(s, errors="coerce", format="mixed")
            bad = s.notna() & conv.isna()
            if bad.any():
                report["invalid_values"].append({"feature": n, "count": int(bad.sum()),
                                                 "examples": s[bad].astype(str).head(3).tolist(), "expected": "date"})
                for i in s.index[bad][:1000]:
                    flag(i, f"'{n}' is not a valid date")
            X[n] = conv
        elif kind == "boolean":
            X[n] = s
        else:
            sv = s.map(lambda v: v if v is None or (isinstance(v, float) and np.isnan(v)) else str(v).strip())
            known = set(f.get("categories") or [])
            if known and f.get("n_unique", 0) <= len(known):
                unseen = sv.notna() & ~sv.isin(known)
                if unseen.any():
                    report["unseen_categories"].append({"feature": n, "count": int(unseen.sum()),
                                                        "examples": sv[unseen].astype(str).unique()[:5].tolist()})
                    for i in s.index[unseen][:1000]:
                        flag(i, f"'{n}' has a value not seen in training")
            X[n] = sv.astype(object)
    if strict and report["invalid_values"]:
        raise ValidationFailed("Some values have the wrong type: " + "; ".join(
            f"{v['feature']} expects a {v['expected']} (got {', '.join(map(repr, v['examples']))})"
            for v in report["invalid_values"]), code="invalid_values", details={"invalid_values": report["invalid_values"]})
    report["row_issues"] = {str(k): v for k, v in list(issues.items())[:2000]}
    report["rows_with_issues"] = len(issues)
    return X[names], report


def predict(pipeline: Any, X: pd.DataFrame, meta: Dict[str, Any]) -> Dict[str, Any]:
    result = meta["result"]
    problem = result["problem_type"]
    classes = result.get("classes") or []
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        raw = pipeline.predict(X)
        out: Dict[str, Any] = {"problem_type": problem}
        if problem == "regression":
            out["predictions"] = [to_jsonable(v) for v in np.asarray(raw, dtype=float)]
        else:
            out["predictions"] = [classes[int(i)] for i in raw]
            if hasattr(pipeline, "predict_proba"):
                proba = pipeline.predict_proba(X)
                out["probabilities"] = [{classes[j]: float(p[j]) for j in range(len(classes))} for p in proba]
                out["confidence"] = [float(np.max(p)) for p in proba]
    out["notes"] = (["Probabilities are model scores and are not necessarily calibrated."]
                    if problem != "regression" else
                    [f"Typical error on held-out data: MAE ≈ {result['test_metrics'].get('mae'):.4g}."
                     if result.get("test_metrics", {}).get("mae") is not None else ""])
    return out


def predictions_frame(df_in: pd.DataFrame, pred: Dict[str, Any], report: Dict[str, Any], target: str) -> pd.DataFrame:
    out = df_in.copy()
    col = f"predicted_{target}"
    out[col] = pred["predictions"]
    if pred.get("probabilities"):
        for cls in pred["probabilities"][0].keys():
            out[f"probability_{cls}"] = [p[cls] for p in pred["probabilities"]]
        out["confidence"] = pred["confidence"]
    issues = report.get("row_issues", {})
    out["input_warnings"] = [("; ".join(issues.get(str(i), [])) or None) for i in df_in.index]
    return out
