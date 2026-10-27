"""Insights must be evidence-based (and quiet on noise); charts must refuse misleading combinations."""
import numpy as np
import pandas as pd
import pytest

from agents.charts import build_chart, recommend_charts
from agents.dashboard import kpi_cards
from agents.insights import generate_findings
from core.errors import ValidationFailed

REQUIRED = {"title", "observation", "evidence", "evidence_text", "why_it_matters", "interpretations", "follow_up",
            "claim_type", "confidence", "score"}


def test_findings_structure_and_known_effect(regression_df):
    f = generate_findings(regression_df)
    assert f, "expected findings"
    for x in f:
        assert REQUIRED <= set(x)
        assert x["claim_type"] in ("observation", "association", "hypothesis")
        assert x["interpretations"] and x["follow_up"]
    seg = [x for x in f if x["category"] == "segment"]
    assert any(set(x["columns"]) == {"target", "region"} for x in seg)
    rel = [x for x in f if x["category"] == "relationship"]
    assert any(set(x["columns"]) == {"x1", "target"} for x in rel)
    # Associations are never phrased as causes
    for x in f:
        assert "causes" not in x["observation"].lower()


def test_no_spurious_findings_on_pure_noise():
    rng = np.random.default_rng(2024)
    n = 400
    df = pd.DataFrame({f"m{i}": rng.normal(0, 1, n) for i in range(6)})
    for i in range(3):
        df[f"g{i}"] = rng.choice(["a", "b", "c", "d"], n)
    f = generate_findings(df)
    assert not [x for x in f if x["category"] in ("relationship", "segment")], [x["title"] for x in f]


def test_identifiers_never_used_as_measures(binary_df):
    f = generate_findings(binary_df)
    for x in f:
        if x["category"] in ("relationship", "segment", "trend", "distribution"):
            assert "customer_id" not in x["columns"]


def test_bar_chart_groups_other_and_flags_small_groups():
    df = pd.DataFrame({"cat": [f"c{i}" for i in range(30)] * 2 + ["big"] * 100, "v": range(160)})
    c = build_chart(df, {"type": "bar", "x": "cat", "agg": "count", "limit": 10})
    labels = [d["label"] for d in c["data"]]
    assert labels[0] == "big" and labels[-1].startswith("Other") and sum(d["value"] for d in c["data"]) == 160
    c = build_chart(df, {"type": "bar", "x": "cat", "y": "v", "agg": "mean", "limit": 10})
    assert any("fewer than 5" in n for n in c["notes"])


def test_chart_validation_rejects_misleading_charts():
    df = pd.DataFrame({"x": np.arange(100) - 50.0, "cat": ["a", "b"] * 50, "d": pd.date_range("2024", periods=100),
                       "id": np.arange(100)})
    with pytest.raises(ValidationFailed):
        build_chart(df, {"type": "histogram", "x": "cat"})
    with pytest.raises(ValidationFailed):
        build_chart(df, {"type": "line", "x": "cat", "y": "x"})
    with pytest.raises(ValidationFailed):
        build_chart(df, {"type": "pie", "x": "cat", "y": "x"})  # sums are negative
    with pytest.raises(ValidationFailed):
        build_chart(df, {"type": "bar", "x": "x"})  # continuous variable as categories
    with pytest.raises(ValidationFailed):
        build_chart(df, {"type": "unknown"})


def test_scatter_sampling_is_disclosed(monkeypatch):
    from core.config import settings
    monkeypatch.setattr(settings, "chart_max_points", 100)
    rng = np.random.default_rng(0)
    df = pd.DataFrame({"a": rng.normal(size=1000), "b": rng.normal(size=1000)})
    c = build_chart(df, {"type": "scatter", "x": "a", "y": "b"})
    assert c["sampled"] and len(c["points"]) == 100 and c["n_total"] == 1000
    assert any("sample" in n for n in c["notes"])


def test_line_chart_marks_gaps_not_zero():
    df = pd.DataFrame({"d": pd.to_datetime(["2024-01-05", "2024-01-20", "2024-03-03", "2024-03-15", "2024-04-02",
                                            "2024-05-09", "2024-06-01"]), "v": [1.0, 2, 3, 4, 5, 6, 7]})
    c = build_chart(df, {"type": "line", "x": "d", "y": "v", "agg": "mean", "freq": "MS"})
    feb = [p for p in c["series"][0]["points"] if p["t"].startswith("2024-02")][0]
    assert feb["value"] is None and feb["n"] == 0


def test_recommended_charts_are_valid(regression_df):
    charts = recommend_charts(regression_df)
    assert charts and all("reason" in c for c in charts)
    assert not any(c.get("x") == "row_id" or c.get("y") == "row_id" for c in charts)


def test_kpis_label_aggregation(client):
    df = pd.DataFrame({"revenue": [10.0, 20.0, 30.0], "unit_price": [1.0, 2.0, 3.0], "order_id": [1, 2, 3]})
    cards = {c["label"]: c for c in kpi_cards(df)}
    assert cards["Total revenue"]["value"] == 60 and cards["Average unit_price"]["value"] == 2
    assert not any("order_id" in k for k in cards)


def test_charts_api(client, sample_ws):
    r = client.post(f"/api/workspaces/{sample_ws}/charts", json={"type": "box", "y": "revenue", "x": "region"})
    assert r.status_code == 200 and r.json()["groups"]
    r = client.post(f"/api/workspaces/{sample_ws}/charts", json={"type": "histogram", "x": "order_id"})
    assert r.status_code == 422  # identifier
    r = client.get(f"/api/workspaces/{sample_ws}/insights/export")
    assert r.status_code == 200 and "priority" in r.text
