"""AutoML correctness (selection, leakage prevention, evaluation) and prediction with schema validation."""
import io

import numpy as np
import pandas as pd
import pytest

from agents.automl import TrainConfig, train
from core.errors import ValidationFailed

from .conftest import csv_bytes, make_ws, wait_job

FAST = dict(models=["linear", "random_forest"], tune=False, cv_folds=3)


def test_regression_learns_known_relationship(regression_df):
    out = train(regression_df, TrainConfig(target="target", **FAST))
    r = out["result"]
    assert r["problem_type"] == "regression"
    assert r["selected"]["key"] != "baseline"
    assert r["test_metrics"]["r2"] > 0.9
    assert r["test_metrics"]["rmse"] < r["baseline_test_metrics"]["rmse"] / 3
    excluded = {f["column"]: f["reason"] for f in r["features"] if f["status"] == "excluded"}
    assert "row_id" in excluded and "identifier" in excluded["row_id"]
    imp = [i["feature"] for i in r["feature_importance"]["items"][:2]]
    assert "x1" in imp
    assert "not used for model selection" in r["rationale"]
    # CV results come from the training split only
    assert r["rows"]["train"] + r["rows"]["test"] == len(regression_df)


def test_imbalanced_binary_classification_reports_balanced_metrics(binary_df):
    out = train(binary_df, TrainConfig(target="churned", **FAST))
    r = out["result"]
    assert r["problem_type"] == "binary" and r["primary_metric"] == "roc_auc"
    tm = r["test_metrics"]
    for k in ("accuracy", "balanced_accuracy", "precision", "recall", "f1", "roc_auc", "average_precision", "brier"):
        assert k in tm
    assert tm["roc_auc"] > 0.6
    assert any("Imbalanced" in w for w in r["warnings"])
    assert r["class_weight"] == "balanced"
    assert r["diagnostics"]["confusion_matrix"]["labels"] == ["no", "yes"]
    assert sum(sum(row) for row in r["diagnostics"]["confusion_matrix"]["matrix"]) == r["rows"]["test"]
    assert any("calibrated" in lim for lim in r["limitations"])


def test_leakage_detection(regression_df):
    df = regression_df.copy()
    df["target_copy"] = df["target"]
    out = train(df, TrainConfig(target="target", **FAST))
    leak = {x["column"]: x for x in out["result"]["leakage"]}
    assert leak["target_copy"]["action"] == "removed"
    used = [f["column"] for f in out["result"]["features"] if f["status"] == "used"]
    assert "target_copy" not in used


def test_classification_relabel_leak_removed(binary_df):
    df = binary_df.copy()
    df["status_code"] = df["churned"].map({"yes": "S1", "no": "S0"})
    out = train(df, TrainConfig(target="churned", **FAST))
    leak = {x["column"]: x for x in out["result"]["leakage"]}
    assert leak["status_code"]["action"] == "removed"


def test_time_split_is_chronological():
    rng = np.random.default_rng(5)
    n = 300
    df = pd.DataFrame({"when": pd.date_range("2022-01-01", periods=n, freq="D").to_series().sample(frac=1, random_state=1).values,
                       "x": rng.normal(size=n)})
    df["y"] = df["x"] * 2 + rng.normal(size=n)
    r = train(df, TrainConfig(target="y", time_column="when", **FAST))["result"]
    assert r["split"]["method"] == "chronological"
    assert pd.Timestamp(r["split"]["test_from"]) > pd.Timestamp(r["split"]["train_until"])
    assert "time-series" in r["cv_description"]
    assert any(f["column"] == "when" and f["status"] == "excluded" for f in r["features"])


def test_duplicate_rows_across_split_are_flagged(regression_df):
    df = pd.concat([regression_df.drop(columns=["row_id"])] * 2, ignore_index=True)
    r = train(df, TrainConfig(target="target", **FAST))["result"]
    assert any("appear in the training data" in w for w in r["warnings"])


def test_invalid_training_requests():
    small = pd.DataFrame({"x": range(10), "y": range(10)})
    with pytest.raises(ValidationFailed):
        train(small, TrainConfig(target="y"))
    const = pd.DataFrame({"x": range(100), "y": [1] * 100})
    with pytest.raises(ValidationFailed):
        train(const, TrainConfig(target="y"))
    with pytest.raises(ValidationFailed):
        train(const, TrainConfig(target="missing"))
    with pytest.raises(ValidationFailed):
        TrainConfig(target="y", test_size=0.9).validate()


def test_rare_classes_are_removed_with_warning():
    rng = np.random.default_rng(0)
    df = pd.DataFrame({"x": rng.normal(size=200), "y": ["a"] * 100 + ["b"] * 98 + ["c"] * 2})
    r = train(df, TrainConfig(target="y", **FAST))["result"]
    assert r["classes"] == ["a", "b"] and any("fewer than" in w for w in r["warnings"])


@pytest.fixture(scope="module")
def trained(client, binary_df):
    wid = make_ws(client, binary_df, "churn.csv")
    r = client.post(f"/api/workspaces/{wid}/ml/train", json={"target": "churned", **FAST})
    assert r.status_code == 202
    job = wait_job(client, r.json()["job_id"])
    assert job["status"] == "succeeded", job
    return wid, job["result"]["model_id"], job["result"]["experiment_id"]


def test_training_job_and_registry(client, trained):
    wid, mid, eid = trained
    models = client.get(f"/api/workspaces/{wid}/models").json()["models"]
    assert models[0]["id"] == mid and models[0]["target"] == "churned"
    exp = client.get(f"/api/workspaces/{wid}/experiments/{eid}").json()
    assert exp["result"]["selected"]["model"]
    csv = client.get(f"/api/workspaces/{wid}/experiments/{eid}/export")
    assert csv.status_code == 200 and "cv_roc_auc_mean" in csv.text
    meta = client.get(f"/api/workspaces/{wid}/models/{mid}").json()
    assert meta["artifact_sha256"] and meta["data_fingerprint"]
    assert {f["name"] for f in meta["input_schema"]["features"]} == {"usage", "tenure", "plan"}


def test_single_prediction_with_validation_and_explanation(client, trained):
    wid, mid, _ = trained
    url = f"/api/workspaces/{wid}/models/{mid}/predict"
    r = client.post(url, json={"rows": [{"usage": 3, "tenure": 2, "plan": "basic"},
                                        {"usage": 80, "tenure": 50, "plan": "team"}], "explain": True})
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["predictions"][0] in ("yes", "no") and len(body["probabilities"]) == 2
    assert abs(sum(body["probabilities"][0].values()) - 1) < 1e-6
    assert body["probabilities"][0]["yes"] > body["probabilities"][1]["yes"]
    assert body["explanations"][0]["contributions"]
    assert any("calibrated" in n for n in body["notes"])
    # missing feature
    r = client.post(url, json={"rows": [{"usage": 3, "plan": "basic"}]})
    assert r.status_code == 422 and r.json()["error"]["code"] == "missing_features"
    # wrong type
    r = client.post(url, json={"rows": [{"usage": "lots", "tenure": 2, "plan": "basic"}]})
    assert r.status_code == 422 and r.json()["error"]["code"] == "invalid_values"
    # unseen category and out-of-range values are accepted with warnings
    r = client.post(url, json={"rows": [{"usage": 5000, "tenure": 2, "plan": "enterprise"}]})
    assert r.status_code == 200
    v = r.json()["validation"]
    assert v["unseen_categories"][0]["feature"] == "plan" and v["out_of_range"][0]["feature"] == "usage"


def test_batch_prediction_and_formula_safe_download(client, trained, binary_df):
    wid, mid, _ = trained
    new = binary_df.drop(columns=["churned"]).head(25).copy()
    new.loc[0, "customer_id"] = "=HYPERLINK(\"http://evil\")"
    r = client.post(f"/api/workspaces/{wid}/models/{mid}/predict/batch", files={"file": ("new.csv", csv_bytes(new))})
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["summary"]["rows"] == 25 and "predicted_churned" in body["columns"]
    dl = client.get(f"/api/workspaces/{wid}/predictions/{body['prediction_id']}/download")
    assert dl.status_code == 200
    out = pd.read_csv(io.StringIO(dl.text.lstrip("\ufeff")))
    assert out.loc[0, "customer_id"].startswith("'=HYPERLINK")  # neutralised formula
    assert len(out) == 25 and "probability_yes" in out.columns


def test_tampered_model_is_refused(client, trained):
    wid, mid, _ = trained
    from services import models as registry
    from services.workspace import store
    ws = store.get(wid)
    path = ws.dir / "models" / mid / "model.joblib"
    original = path.read_bytes()
    try:
        path.write_bytes(original + b"tamper")
        registry.clear_cache()
        r = client.post(f"/api/workspaces/{wid}/models/{mid}/predict",
                        json={"rows": [{"usage": 3, "tenure": 2, "plan": "basic"}]})
        assert r.status_code == 409 and r.json()["error"]["code"] == "integrity_check_failed"
    finally:
        path.write_bytes(original)
        registry.clear_cache()


def test_cleaning_steps_are_replayed_at_prediction_time(client):
    rng = np.random.default_rng(3)
    n = 200
    df = pd.DataFrame({"price": [f"${v:,.2f}" for v in rng.uniform(10, 500, n)], "qty": rng.integers(1, 10, n)})
    df["total"] = df["price"].str.replace(r"[$,]", "", regex=True).astype(float) * df["qty"] + rng.normal(0, 5, n)
    wid = make_ws(client, df)
    r = client.post(f"/api/workspaces/{wid}/cleaning/apply", json={"op": "convert_type", "params": {"column": "price", "to": "numeric"}})
    assert r.status_code == 200
    r = client.post(f"/api/workspaces/{wid}/ml/train", json={"target": "total", **FAST})
    job = wait_job(client, r.json()["job_id"])
    mid = job["result"]["model_id"]
    # New data arrives in the original messy format: "$" and thousands separators
    r = client.post(f"/api/workspaces/{wid}/models/{mid}/predict", json={"rows": [{"price": "$1,00.00", "qty": 3},
                                                                                 {"price": "$250.00", "qty": 2}]})
    assert r.status_code == 200, r.text
    preds = r.json()["predictions"]
    assert 250 < preds[0] < 350 and 400 < preds[1] < 600


def test_job_cancellation(client, regression_df):
    wid = make_ws(client, regression_df)
    r = client.post(f"/api/workspaces/{wid}/ml/train", json={"target": "target"})
    jid = r.json()["job_id"]
    client.post(f"/api/jobs/{jid}/cancel")
    j = wait_job(client, jid)
    assert j["status"] in ("cancelled", "succeeded")  # may finish before the cancel is observed
    assert client.get("/api/jobs/" + "f" * 16).status_code == 404


def test_clustering_and_anomalies(client, sample_ws):
    r = client.post(f"/api/workspaces/{sample_ws}/ml/cluster", json={"n_clusters": 3})
    assert r.status_code == 200
    c = r.json()
    assert c["n_clusters"] == 3 and sum(s["size"] for s in c["cluster_summaries"]) == 100
    assert "order_id" not in c["features_used"] and c["silhouette_score"] is not None
    r = client.post(f"/api/workspaces/{sample_ws}/ml/anomaly", json={"contamination": 0.05})
    a = r.json()
    assert a["n_anomalies"] == 5 and a["top_anomalies"][0]["reasons"]
    assert client.get(f"/api/workspaces/{sample_ws}/analyses/anomaly").status_code == 200


@pytest.mark.parametrize("problem", ["regression", "classification"])
def test_row_sampling_keeps_target_aligned(regression_df, binary_df, problem):
    """Regression test: sampling above TRAINING_MAX_ROWS used to misalign X and y (benchmark at 500k rows)."""
    df, target = (regression_df, "target") if problem == "regression" else (binary_df, "churned")
    out = train(df, TrainConfig(target=target, max_rows=150, **FAST))
    r = out["result"]
    assert r["rows"]["sampled_from"] == len(df[df[target].notna()])
    assert r["rows"]["used"] <= 150
    assert any("sample" in w for w in r["warnings"])
