"""Auditable cleaning: impact measurement, confirmation, undo, lineage, inference replay."""
import numpy as np
import pandas as pd
import pytest

from agents.cleaner import apply_inference_ops, apply_op, detect_issues, preview_op, replay_ops
from core.errors import ValidationFailed

from .conftest import csv_bytes, make_ws

MESSY = pd.DataFrame({
    "id": [1, 2, 3, 3, 4, 5, 6, 7],
    "name": [" Alice", "Bob ", "Carol", "Carol", "Dan", "Eve", "Frank", "Gina"],
    "amount": ["$1,200.50", "N/A", "$300", "$300", "45", "-", "$12", "$7"],
    "city": ["NY", "ny", "N.Y.", "N.Y.", "Boston", "boston", "Boston", "NY"],
    "age": [34, -2, 51, 51, None, 40, 29, 33],
})


def test_detects_expected_issues():
    kinds = {i["issue"] for i in detect_issues(MESSY)}
    assert {"duplicates", "numeric_as_text", "inconsistent_labels", "whitespace", "null_tokens", "negative_values",
            "missing_values"} <= kinds


def test_convert_numeric_and_impact():
    p = preview_op(MESSY, "convert_type", {"column": "amount", "to": "numeric"})
    imp = p["impact"]
    # "N/A" and "-" are placeholders → become missing, which counts as nullified (lost) values
    assert imp["values_nullified"] == 2 and imp["requires_confirmation"]
    df, rec = apply_op(MESSY, "convert_type", {"column": "amount", "to": "numeric"})
    assert df["amount"].dtype == np.float64
    assert df["amount"].iloc[0] == 1200.5 and np.isnan(df["amount"].iloc[1]) and df["amount"].iloc[2] == 300.0
    assert rec["applies_to_new_data"]


def test_imputation_stores_fitted_value_and_reuses_it():
    df, rec = apply_op(MESSY, "impute", {"columns": ["age"], "strategy": "median"})
    med = float(pd.Series([34, -2, 51, 51, 40, 29, 33]).median())
    assert rec["fitted"]["age"] == med and df["age"].isna().sum() == 0
    new = pd.DataFrame({"age": [None, 10.0]})
    out, notes = apply_inference_ops(new, [rec])
    assert out["age"].tolist() == [med, 10.0]  # same fitted median, not recomputed on new data


def test_drop_duplicates_requires_confirmation_via_api(client):
    wid = make_ws(client, MESSY)
    body = {"op": "drop_duplicates", "params": {}}
    r = client.post(f"/api/workspaces/{wid}/cleaning/apply", json=body)
    assert r.status_code == 409 and r.json()["error"]["code"] == "confirmation_required"
    assert r.json()["error"]["details"]["impact"]["rows_removed"] == 1
    r = client.post(f"/api/workspaces/{wid}/cleaning/apply", json={**body, "confirm": True})
    assert r.status_code == 200 and r.json()["dataset"]["rows"] == 7
    # original is untouched
    data = client.get(f"/api/workspaces/{wid}/data", params={"version": "original"}).json()
    assert data["total_rows"] == 8


def test_undo_revert_and_history(client):
    wid = make_ws(client, MESSY)
    base = f"/api/workspaces/{wid}/cleaning"
    assert client.post(f"{base}/apply", json={"op": "trim_whitespace", "params": {}}).status_code == 200
    assert client.post(f"{base}/apply", json={"op": "drop_duplicates", "params": {}, "confirm": True}).status_code == 200
    assert client.post(f"{base}/apply", json={"op": "standardize_case", "params": {"columns": ["city"], "case": "upper"}}).status_code == 200
    h = client.get(f"{base}/history").json()
    assert [s["label"] for s in h["steps"]][1:] == ["Trim whitespace", "Remove duplicate rows", "Standardize letter case"]
    assert h["current"]["rows"] == 7
    r = client.post(f"{base}/undo")
    assert r.status_code == 200 and r.json()["undone"]["op"] == "standardize_case"
    first_id = h["steps"][1]["id"]
    r = client.post(f"{base}/revert", json={"op_id": first_id})
    assert r.status_code == 200 and r.json()["dataset"]["rows"] == 8  # duplicates are back
    r = client.post(f"{base}/apply", json={"op": "drop_duplicates", "params": {}, "confirm": True})
    r = client.post(f"{base}/apply", json={"op": "drop_empty_rows", "params": {}})
    r = client.post(f"{base}/revert", json={"op_id": None})
    assert r.status_code == 409  # resetting several steps needs confirmation
    r = client.post(f"{base}/revert", json={"op_id": None, "confirm": True})
    assert r.json()["dataset"]["n_ops"] == 0


def test_replay_is_deterministic():
    ops = []
    df = MESSY
    for op, params in [("trim_whitespace", {}), ("convert_type", {"column": "amount", "to": "numeric"}),
                       ("impute", {"columns": ["amount"], "strategy": "mean"}),
                       ("map_values", {"column": "city", "mapping": {"ny": "NY", "N.Y.": "NY", "boston": "Boston"}})]:
        df, rec = apply_op(df, op, params)
        ops.append(rec)
    again = replay_ops(MESSY, ops)
    pd.testing.assert_frame_equal(df, again)


def test_apply_safe_only_runs_non_destructive_steps(client):
    wid = make_ws(client, MESSY)
    r = client.post(f"/api/workspaces/{wid}/cleaning/apply-safe", json={})
    assert r.status_code == 200
    ops = [a["op"] for a in r.json()["applied"]]
    assert "drop_duplicates" not in ops and "trim_whitespace" in ops
    for a in r.json()["applied"]:
        assert not a["impact"]["requires_confirmation"]


def test_operations_validate_parameters():
    with pytest.raises(ValidationFailed):
        apply_op(MESSY, "impute", {"columns": ["name"], "strategy": "mean"})
    with pytest.raises(ValidationFailed):
        apply_op(MESSY, "drop_columns", {"columns": ["nope"]})
    with pytest.raises(ValidationFailed):
        apply_op(MESSY, "not_an_op", {})
    with pytest.raises(ValidationFailed):
        apply_op(MESSY, "remove_rows_where", {"column": "name", "operator": "matches", "value": "(a+)+$"})


def test_outlier_ops_and_conditions():
    df = pd.DataFrame({"v": list(range(1, 51)) + [1000]})
    out, rec = apply_op(df, "clip_outliers", {"column": "v", "method": "iqr", "k": 1.5})
    assert out["v"].max() < 1000 and rec["impact"]["rows_removed"] == 0
    out, rec = apply_op(df, "remove_outliers", {"column": "v", "method": "iqr", "k": 3})
    assert len(out) == 50 and rec["impact"]["requires_confirmation"]
    out, rec = apply_op(df, "nullify_where", {"column": "v", "operator": ">", "value": 100})
    assert out["v"].isna().sum() == 1 and rec["impact"]["values_nullified"] == 1


def test_split_number_unit_reports_mixed_units():
    df = pd.DataFrame({"w": ["10 kg", "2 lb", "3.5 kg"]})
    out, rec = apply_op(df, "split_number_unit", {"column": "w"})
    assert out["w"].tolist() == [10.0, 2.0, 3.5] and out["w_unit"].tolist() == ["kg", "lb", "kg"]
    assert any("not converted" in n for n in rec["notes"])


def test_cleaning_invalidates_cached_analysis(client):
    wid = make_ws(client, MESSY)
    p1 = client.get(f"/api/workspaces/{wid}/profile").json()
    assert p1["duplicates"]["rows"] == 1
    client.post(f"/api/workspaces/{wid}/cleaning/apply", json={"op": "drop_duplicates", "params": {}, "confirm": True})
    p2 = client.get(f"/api/workspaces/{wid}/profile").json()
    assert p2["duplicates"]["rows"] == 0 and p2["shape"]["rows"] == 7


def test_quality_rules(client):
    wid = make_ws(client, MESSY)
    base = f"/api/workspaces/{wid}/rules"
    for rule in [{"type": "range", "column": "age", "min": 0, "max": 120},
                 {"type": "unique", "column": "id"},
                 {"type": "allowed_values", "column": "city", "values": ["NY", "Boston"]},
                 {"type": "not_null", "column": "age"}]:
        assert client.post(base, json={"rule": rule}).status_code == 201
    assert client.post(base, json={"rule": {"type": "pattern", "column": "name", "pattern": "(x+)+y"}}).status_code == 422
    res = client.post(f"{base}/evaluate").json()
    by = {r["rule"]["type"]: r for r in res["results"]}
    assert by["range"]["violations"] == 1 and by["unique"]["violations"] == 2
    assert by["allowed_values"]["violations"] == 4 and by["not_null"]["violations"] == 1
    assert res["rules_passed"] == 0
