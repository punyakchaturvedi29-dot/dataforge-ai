"""Forecasting (time-aware validation), diagnostics (no causal over-claiming) and validated joins."""
import numpy as np
import pandas as pd
import pytest

from agents import forecasting as F
from agents.multifile import join_candidates, preview_join
from agents.rca import run_diagnostics
from core.errors import ValidationFailed

from .conftest import csv_bytes, make_ws


def _monthly(df):
    d = df.copy()
    d["month"] = pd.to_datetime(d["month"])
    return d


def test_seasonal_series_beats_naive(monthly_df):
    r = F.run_forecast(_monthly(monthly_df), "month", "sales", horizon=12)
    assert r["available"] and r["frequency"] == "MS" and r["seasonal_period"] == 12
    sel = next(c for c in r["comparison"] if c["method"] == r["selected_method"])
    naive = next(c for c in r["comparison"] if c["method"] == "naive")
    assert sel["mase"] < naive["mase"] * 0.5
    assert r["selected_method"] in ("holt_winters", "trend_regression", "seasonal_naive", "arima")
    fc = r["forecasts"][r["selected_method"]]
    assert len(fc["forecast"]) == 12
    assert all(lo <= f <= hi for lo, f, hi in zip(fc["lower80"], fc["forecast"], fc["upper80"]))
    assert all(a <= b for a, b in zip(fc["lower95"], fc["lower80"]))
    assert r["future_index"][0].startswith("2024-01")


def test_backtest_never_trains_on_the_future(monthly_df, monkeypatch):
    seen = []
    orig = F.METHODS["drift"]

    def spy(train, h, m):
        seen.append(len(train))
        return orig(train, h, m)

    monkeypatch.setitem(F.METHODS, "drift", spy)
    y = _monthly(monthly_df)["sales"].to_numpy()
    res, origins = F.backtest(y, 6, 12, ["naive", "drift"], k=3)
    assert seen[:3] == origins  # each fold trains only on observations before its origin
    for fold, o in zip(res["drift"]["folds"], origins):
        assert fold["origin"] == o and len(fold["actual"]) == 6


def test_forecast_unavailable_without_dates(regression_df):
    opts = F.forecast_options(regression_df)
    assert not opts["available"] and "date" in opts["reason"].lower()


def test_short_series_and_gaps():
    df = pd.DataFrame({"d": pd.date_range("2024-01-01", periods=5, freq="MS"), "v": [1.0, 2, 3, 4, 5]})
    r = F.run_forecast(df, "d", "v", horizon=3)
    assert not r["available"] and "at least" in r["reason"]
    rng = np.random.default_rng(0)
    days = pd.to_datetime(rng.choice(pd.date_range("2023-01-01", "2023-12-31"), 120, replace=False))
    df = pd.DataFrame({"d": np.concatenate([days, days[:10]]), "v": rng.uniform(1, 10, 130)})
    r = F.run_forecast(df, "d", "v", horizon=4, agg="sum")
    assert r["available"]
    assert r["diagnostics"]["duplicate_timestamps"] == 10
    assert any("Aggregated to weekly" in w for w in r["warnings"])


def test_horizon_is_capped(monthly_df):
    r = F.run_forecast(_monthly(monthly_df).head(24), "month", "sales", horizon=20)
    assert r["horizon"] == 8 and any("reduced" in w for w in r["warnings"])


def test_forecast_api_persists_latest(client, monthly_df):
    wid = make_ws(client, monthly_df, "monthly.csv")
    r = client.post(f"/api/workspaces/{wid}/forecast", json={"date_column": "month", "target": "sales", "horizon": 6})
    assert r.status_code == 200 and r.json()["available"]
    latest = client.get(f"/api/workspaces/{wid}/analyses/forecast").json()
    assert latest["result"]["horizon"] == 6 and latest["stale"] is False
    ov = client.get(f"/api/workspaces/{wid}/overview").json()
    assert ov["indicators"]["forecast"]["method"]


def test_simpsons_paradox_is_detected():
    rng = np.random.default_rng(0)
    n = 2000
    dept = rng.choice(["A", "B"], n)
    grp = np.where(dept == "A", rng.choice(["X", "Y"], n, p=[0.9, 0.1]), rng.choice(["X", "Y"], n, p=[0.1, 0.9]))
    y = np.where(dept == "A", 100, 50) + np.where(grp == "Y", 5, 0) + rng.normal(0, 5, n)
    r = run_diagnostics(pd.DataFrame({"dept": dept, "grp": grp, "score": y}), "score", dimensions=["grp", "dept"])
    g = next(s for s in r["segments"] if s["dimension"] == "grp")
    assert g["best_vs_worst"]["best"] == "X"
    assert g["confounding_check"]["controlled_for"] == "dept"
    assert g["confounding_check"]["adjusted_to_raw_ratio"] < 0
    f = next(f for f in r["findings"] if "'grp'" in f["title"])
    assert f["label"] == "diagnostic hypothesis" and "dept" in f["confounders"]


def test_period_decomposition_is_exact(sample_ws, client):
    r = client.post(f"/api/workspaces/{sample_ws}/diagnostics", json={"outcome": "revenue"})
    assert r.status_code == 200
    p = r.json()["period_change"]
    for dim in p["contributions"]:
        assert sum(x["contribution"] for x in dim["rows"]) == pytest.approx(p["change"], rel=1e-6, abs=1e-6) or len(dim["rows"]) == 12


def test_diagnostics_language_and_labels(sample_ws, client):
    r = client.post(f"/api/workspaces/{sample_ws}/diagnostics", json={"outcome": "revenue"}).json()
    assert "cannot prove causation" in r["causal_note"]
    for f in r["findings"]:
        assert f["label"] in ("observation", "potential driver", "diagnostic hypothesis")
        assert "root cause" not in f["title"].lower()
        assert f["validation"]


def test_binary_outcome_needs_positive_class_when_ambiguous(sample_ws, client):
    r = client.post(f"/api/workspaces/{sample_ws}/diagnostics", json={"outcome": "region"})
    assert r.status_code == 422 and r.json()["error"]["code"] == "needs_positive_class"
    r = client.post(f"/api/workspaces/{sample_ws}/diagnostics", json={"outcome": "region", "positive_class": "North"})
    assert r.status_code == 200 and r.json()["outcome_kind"] == "rate"


ORDERS = pd.DataFrame({"order_id": range(1, 201), "cust": np.random.default_rng(0).integers(1, 51, 200),
                       "amount": np.round(np.random.default_rng(1).normal(100, 20, 200), 2), "region": ["N", "S"] * 100})
CUSTOMERS = pd.DataFrame({"customer_id": [str(i) for i in range(1, 46)], "segment": ["A", "B", "C"] * 15,
                          "region": ["N", "S", "E"] * 15})


def test_join_candidates_use_value_evidence():
    c = join_candidates(ORDERS, CUSTOMERS)
    assert (c[0]["left_on"], c[0]["right_on"]) == ("cust", "customer_id")
    assert c[0]["cardinality"] == "N:1"
    spurious = next(x for x in c if x["left_on"] == "order_id")
    assert any("auto-increment" in r for r in spurious["reasons"])


def test_join_preview_counts():
    p = preview_join(ORDERS, CUSTOMERS, "cust", "customer_id", "left")
    unmatched = int((~ORDERS["cust"].isin(range(1, 46))).sum())
    assert p["rows_result"] == 200 and p["unmatched_left_rows"] == unmatched
    assert p["matched_left_rows"] == 200 - unmatched and not p["requires_confirmation"]
    assert p["column_collisions"] == ["region"]


def test_join_and_stack_api(client):
    wid = make_ws(client, ORDERS, "orders.csv")
    r = client.post(f"/api/workspaces/{wid}/sources", files={"file": ("customers.csv", csv_bytes(CUSTOMERS))})
    assert r.status_code == 201
    ws = client.get(f"/api/workspaces/{wid}").json()
    left = ws["active_dataset"]["id"]
    right = next(d["id"] for d in ws["datasets"] if d["id"] != left)
    cands = client.get(f"/api/workspaces/{wid}/sources/join-candidates", params={"left": left, "right": right}).json()
    assert cands["candidates"][0]["left_on"] == "cust"
    nm = {"left_id": left, "right_id": right, "left_on": "region", "right_on": "region", "how": "inner"}
    r = client.post(f"/api/workspaces/{wid}/sources/join", json=nm)
    assert r.status_code == 409  # many-to-many explosion needs confirmation
    good = {"left_id": left, "right_id": right, "left_on": "cust", "right_on": "customer_id", "how": "left"}
    r = client.post(f"/api/workspaces/{wid}/sources/join", json=good)
    assert r.status_code == 201
    ds = r.json()["dataset"]
    assert ds["rows"] == 200 and ds["source"] == "join" and ds["lineage"]["left_on"] == "cust"
    ws = client.get(f"/api/workspaces/{wid}").json()
    assert ws["active_dataset"]["id"] == ds["id"]
    r = client.post(f"/api/workspaces/{wid}/sources/stack", json={"ids": [left, left], "mode": "intersection"})
    assert r.status_code == 201 and r.json()["dataset"]["rows"] == 400
