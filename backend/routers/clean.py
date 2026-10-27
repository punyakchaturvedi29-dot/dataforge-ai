"""Cleaning workflow (preview → confirm → apply → undo) and data-quality rules."""
from __future__ import annotations

import time
from typing import Any, Dict, List, Optional

from fastapi import APIRouter
from pydantic import BaseModel, Field

from agents.cleaner import OPS, apply_op, preview_op, replay_ops
from agents.quality_rules import PRESET_PATTERNS, RULE_TYPES, evaluate_rules, validate_rule
from core.errors import Conflict, NotFound, ValidationFailed
from core.jsonsafe import to_jsonable
from services import analysis as A

from ._deps import get_ws

router = APIRouter()

MAX_OPS = 200

OP_CATALOG = [
    {"op": "drop_duplicates", "params": {"subset": "columns|null", "keep": "first|last"}, "destructive": True},
    {"op": "drop_empty_rows", "params": {}, "destructive": True},
    {"op": "drop_rows_missing", "params": {"columns": "columns|null", "how": "any|all"}, "destructive": True},
    {"op": "drop_columns", "params": {"columns": "columns"}, "destructive": True},
    {"op": "rename_column", "params": {"column": "column", "new_name": "text"}, "destructive": False},
    {"op": "impute", "params": {"columns": "columns", "strategy": "mean|median|mode|constant|ffill|bfill|interpolate|knn",
                                "value": "any (constant)", "add_indicator": "bool"}, "destructive": False},
    {"op": "convert_type", "params": {"column": "column", "to": "numeric|integer|datetime|text|category|boolean",
                                      "dayfirst": "bool|null", "decimal": "auto|.|,"}, "destructive": "if values cannot be converted"},
    {"op": "trim_whitespace", "params": {"columns": "columns|null", "collapse_internal": "bool"}, "destructive": False},
    {"op": "standardize_case", "params": {"columns": "columns", "case": "lower|upper|title"}, "destructive": False},
    {"op": "map_values", "params": {"column": "column", "mapping": "object"}, "destructive": False},
    {"op": "replace_null_tokens", "params": {"columns": "columns|null", "tokens": "list|null"}, "destructive": "yes (values become missing)"},
    {"op": "clip_outliers", "params": {"column": "column", "method": "iqr|percentile|zscore|mad", "k": "number"}, "destructive": False},
    {"op": "remove_outliers", "params": {"column": "column", "method": "iqr|zscore|mad", "k": "number"}, "destructive": True},
    {"op": "remove_rows_where", "params": {"column": "column", "operator": "==|!=|>|<|>=|<=|in|not_in|is_null|not_null|between|outside|matches",
                                           "value": "any"}, "destructive": True},
    {"op": "nullify_where", "params": {"column": "column", "operator": "see remove_rows_where", "value": "any"}, "destructive": True},
    {"op": "split_number_unit", "params": {"column": "column", "unit_column": "text|null"}, "destructive": False},
]


class OpRequest(BaseModel):
    op: str = Field(..., max_length=40)
    params: Dict[str, Any] = Field(default_factory=dict)
    confirm: bool = False
    dataset_id: Optional[str] = None


@router.get("/{workspace_id}/cleaning/operations")
def operations():
    return {"operations": [{**o, "label": OPS[o["op"]]["label"], "applies_to_new_data": OPS[o["op"]]["inference"]}
                           for o in OP_CATALOG]}


@router.get("/{workspace_id}/cleaning/suggestions")
def suggestions(workspace_id: str, dataset_id: Optional[str] = None):
    ws = get_ws(workspace_id)
    ds = ws.dataset(dataset_id)
    return {"dataset_id": ds.id, "version": ds.version, "suggestions": A.issues(ws, ds)}


@router.post("/{workspace_id}/cleaning/preview")
def preview(workspace_id: str, body: OpRequest):
    ws = get_ws(workspace_id)
    ds = ws.dataset(body.dataset_id)
    return preview_op(ds.working, body.op, body.params)


def _apply(ws, ds, op: str, params: Dict, confirm: bool) -> Dict[str, Any]:
    if len(ds.ops) >= MAX_OPS:
        raise ValidationFailed(f"A dataset can have at most {MAX_OPS} cleaning steps. Reset or start a new workspace.")
    with ws.lock:
        new_df, record = apply_op(ds.working, op, params)
        imp = record["impact"]
        if imp["requires_confirmation"] and not confirm:
            raise Conflict("This step removes rows/columns or turns values into missing values. "
                           "Review the impact and confirm to apply it.", code="confirmation_required",
                           details={"impact": imp, "notes": record["notes"]})
        if len(new_df) == 0:
            raise ValidationFailed("This step would remove every row; it was not applied.")
        ds.set_working(new_df, ds.ops + [record])
        ws.log("cleaning_applied", f"{record['label']}: {_impact_text(imp)}", op=op, dataset_id=ds.id)
        ws.mark_stage("clean", save=False)
        ws.save()
    return record


def _impact_text(imp: Dict) -> str:
    parts = []
    if imp["rows_removed"]:
        parts.append(f"{imp['rows_removed']:,} row(s) removed")
    if imp["columns_removed"]:
        parts.append(f"{len(imp['columns_removed'])} column(s) removed")
    if imp["columns_added"]:
        parts.append(f"{len(imp['columns_added'])} column(s) added")
    if imp["cells_modified"]:
        parts.append(f"{imp['cells_modified']:,} value(s) changed")
    if imp["values_filled"]:
        parts.append(f"{imp['values_filled']:,} missing value(s) filled")
    if imp["values_nullified"]:
        parts.append(f"{imp['values_nullified']:,} value(s) set to missing")
    return ", ".join(parts) or "no changes"


@router.post("/{workspace_id}/cleaning/apply")
def apply(workspace_id: str, body: OpRequest):
    ws = get_ws(workspace_id)
    ds = ws.dataset(body.dataset_id)
    rec = _apply(ws, ds, body.op, body.params, body.confirm)
    return {"applied": rec, "dataset": ds.summary()}


class SafeRequest(BaseModel):
    dataset_id: Optional[str] = None
    suggestion_ids: Optional[List[str]] = None


@router.post("/{workspace_id}/cleaning/apply-safe", summary="Apply all non-destructive suggestions (or a chosen subset)")
def apply_safe(workspace_id: str, body: SafeRequest):
    ws = get_ws(workspace_id)
    ds = ws.dataset(body.dataset_id)
    applied, skipped = [], []
    for sug in A.issues(ws, ds):
        if body.suggestion_ids is not None and sug["id"] not in body.suggestion_ids:
            continue
        if not sug.get("recommended") or (body.suggestion_ids is None and not sug.get("safe")):
            continue
        r = sug["recommended"]
        try:
            # Safe suggestions never need confirmation; a chosen destructive one is skipped, not forced.
            applied.append(_apply(ws, ds, r["op"], r["params"], confirm=False))
        except Conflict as e:
            skipped.append({"suggestion": sug["title"], "reason": "needs confirmation", "impact": e.details.get("impact")})
        except ValidationFailed as e:
            skipped.append({"suggestion": sug["title"], "reason": e.message})
    return {"applied": applied, "skipped": skipped, "dataset": ds.summary()}


@router.post("/{workspace_id}/cleaning/undo")
def undo(workspace_id: str, dataset_id: Optional[str] = None):
    ws = get_ws(workspace_id)
    ds = ws.dataset(dataset_id)
    if not ds.ops:
        raise ValidationFailed("There is nothing to undo.")
    with ws.lock:
        last = ds.ops[-1]
        ops = ds.ops[:-1]
        ds.set_working(replay_ops(ds.original, ops), ops)
        ws.log("cleaning_undone", f"Undid '{last['label']}'", dataset_id=ds.id)
        ws.save()
    return {"undone": last, "dataset": ds.summary()}


class RevertRequest(BaseModel):
    op_id: Optional[str] = None  # keep operations up to and including this one; None = reset everything
    confirm: bool = False
    dataset_id: Optional[str] = None


@router.post("/{workspace_id}/cleaning/revert")
def revert(workspace_id: str, body: RevertRequest):
    ws = get_ws(workspace_id)
    ds = ws.dataset(body.dataset_id)
    if body.op_id is None:
        keep = []
    else:
        idx = next((i for i, o in enumerate(ds.ops) if o["id"] == body.op_id), None)
        if idx is None:
            raise NotFound("Cleaning step not found")
        keep = ds.ops[: idx + 1]
    removed = len(ds.ops) - len(keep)
    if removed > 1 and not body.confirm:
        raise Conflict(f"This reverts {removed} cleaning steps. Confirm to continue.", code="confirmation_required",
                       details={"steps_removed": removed})
    with ws.lock:
        ds.set_working(replay_ops(ds.original, keep), keep)
        ws.log("cleaning_reverted", f"Reverted {removed} cleaning step(s)", dataset_id=ds.id)
        ws.save()
    return {"removed": removed, "dataset": ds.summary()}


@router.get("/{workspace_id}/cleaning/history")
def history(workspace_id: str, dataset_id: Optional[str] = None):
    ws = get_ws(workspace_id)
    ds = ws.dataset(dataset_id)
    m = ds.meta
    lineage = [{"step": 0, "label": "Ingested", "detail": f"{m.get('original_rows'):,} rows × {m.get('original_columns')} columns "
                                                       f"from {m.get('filename')}", "source": m.get("source"),
                "lineage": m.get("lineage")}]
    for i, o in enumerate(ds.ops, 1):
        lineage.append({"step": i, "id": o["id"], "label": o["label"], "op": o["op"], "params": o["params"],
                        "fitted": o.get("fitted"), "detail": _impact_text(o["impact"]), "impact": o["impact"],
                        "notes": o.get("notes", []), "applies_to_new_data": o.get("applies_to_new_data"),
                        "at": o.get("created_at")})
    return to_jsonable({"dataset_id": ds.id, "version": ds.version, "steps": lineage,
                        "current": {"rows": m.get("rows"), "columns": m.get("columns")}})


# ── Data-quality rules ───────────────────────────────────────────────────────────

class RuleBody(BaseModel):
    rule: Dict[str, Any]


@router.get("/{workspace_id}/rules")
def list_rules(workspace_id: str):
    ws = get_ws(workspace_id)
    return {"rules": ws.meta.get("rules", []), "types": RULE_TYPES, "presets": list(PRESET_PATTERNS)}


@router.post("/{workspace_id}/rules", status_code=201)
def add_rule(workspace_id: str, body: RuleBody):
    ws = get_ws(workspace_id)
    rule = validate_rule(body.rule, ws.active.working)
    with ws.lock:
        rules = ws.meta.setdefault("rules", [])
        if len(rules) >= 100:
            raise ValidationFailed("A workspace can have at most 100 rules.")
        rules.append({**rule, "created_at": time.time()})
        ws.log("rule_added", f"Added rule '{rule['name']}'")
        ws.save()
    return {"rule": rule, "rules": ws.meta["rules"]}


@router.delete("/{workspace_id}/rules/{rule_id}", status_code=204)
def delete_rule(workspace_id: str, rule_id: str):
    ws = get_ws(workspace_id)
    with ws.lock:
        before = len(ws.meta.get("rules", []))
        ws.meta["rules"] = [r for r in ws.meta.get("rules", []) if r["id"] != rule_id]
        if len(ws.meta["rules"]) == before:
            raise NotFound("Rule not found")
        ws.save()


@router.post("/{workspace_id}/rules/evaluate")
def evaluate(workspace_id: str, dataset_id: Optional[str] = None):
    ws = get_ws(workspace_id)
    ds = ws.dataset(dataset_id)
    res = evaluate_rules(ws.meta.get("rules", []), ds.working)
    A.save_analysis(ws, "rules", {}, res, ds)
    return res
