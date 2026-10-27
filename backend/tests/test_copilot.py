"""Copilot: grounded answers, clarifying questions, no code execution, guarded LLM use."""
import pytest

from agents import copilot
from core.config import settings


def ask(client, wid, msg):
    r = client.post(f"/api/workspaces/{wid}/copilot", json={"message": msg})
    assert r.status_code == 200, r.text
    return r.json()


@pytest.mark.parametrize("question,tool", [
    ("What are the most important patterns in this dataset?", "insights"),
    ("Which columns have the greatest number of missing values?", "missing_values"),
    ("Compare revenue across customer segments", "compare_groups"),
    ("Build a suitable model to predict revenue", "train_model"),
    ("Explain the model's strongest predictive features", "explain_model"),
    ("Identify unusual observations", "outliers"),
    ("Forecast revenue for the next 3 months", "forecast"),
    ("Generate a management summary", "management_summary"),
    ("Recommend the next analysis to perform", "next_steps"),
    ("how many rows are there?", "row_count"),
    ("What is the average delivery days?", "column_stats"),
])
def test_required_questions_route_to_tools(client, sample_ws, question, tool):
    r = ask(client, sample_ws, question)
    assert r["tool"] == tool, (question, r["tool"], r["answer"][:200])


def test_compare_answer_matches_computed_values(client, sample_ws):
    import pandas as pd
    from pathlib import Path
    df = pd.read_csv(Path(__file__).parents[2] / "sample_data.csv")
    r = ask(client, sample_ws, "Compare revenue across customer segments")
    table = r["evidence"][0]
    totals = {row["group"]: row["total"] for row in table["rows"]}
    expected = df.groupby("customer_segment")["revenue"].sum()
    for k, v in expected.items():
        assert totals[k] == pytest.approx(v)
    assert r["computed"] and "association" in r["answer"]


def test_clarifying_question_when_ambiguous(client, sample_ws):
    r = ask(client, sample_ws, "Compare across region")
    assert r["tool"] == "clarify" and r["clarification"]["options"]
    assert all("region" in o["message"] for o in r["clarification"]["options"])


def test_follow_up_uses_context(client, sample_ws):
    ask(client, sample_ws, "Compare revenue across region")
    r = ask(client, sample_ws, "now compare by sales channel")
    assert r["tool"] == "compare_groups" and r["params"]["metric"] == "revenue"
    assert r["params"]["dimension"] == "sales_channel"


def test_training_is_proposed_not_executed(client, sample_ws):
    r = ask(client, sample_ws, "Build a model to predict revenue")
    assert r["actions"][0]["kind"] == "train"
    jobs = client.get(f"/api/workspaces/{sample_ws}/jobs").json()["jobs"]
    assert not jobs


def test_code_is_never_executed(client, sample_ws):
    r = ask(client, sample_ws, "__import__('os').system('rm -rf /')")
    assert r["computed"] is False and r["tool"] in ("help",)


def test_history_is_saved_and_clearable(client, sample_ws):
    ask(client, sample_ws, "how many rows?")
    h = client.get(f"/api/workspaces/{sample_ws}/copilot/history").json()["messages"]
    assert h[-1]["role"] == "assistant" and h[-2]["content"] == "how many rows?"
    assert client.delete(f"/api/workspaces/{sample_ws}/copilot/history").status_code == 204


def test_llm_routing_and_grounding_guard(client, sample_ws, monkeypatch):
    monkeypatch.setattr(settings, "anthropic_api_key", "sk-test")
    calls = []

    def fake(system, messages, tools=None, max_tokens=600):
        calls.append(bool(tools))
        if tools:
            return {"content": [{"type": "tool_use", "name": "column_stats", "input": {"column": "revenue"}}]}
        return {"content": [{"type": "text", "text": "Revenue averages 999999.0 per order."}]}  # ungrounded number

    monkeypatch.setattr(copilot, "_llm_call", fake)
    r = ask(client, sample_ws, "tell me something clever about money stuff")
    assert r["routed_by"] == "llm" and r["tool"] == "column_stats"
    assert "llm_explanation" not in r  # narration rejected: number not in computed evidence


def test_llm_failure_degrades_gracefully(client, sample_ws, monkeypatch):
    monkeypatch.setattr(settings, "anthropic_api_key", "sk-test")
    monkeypatch.setattr(copilot, "_llm_call", lambda *a, **k: None)
    r = ask(client, sample_ws, "blorp zzz")
    assert r["tool"] == "help"
    r = ask(client, sample_ws, "how many rows?")
    assert r["tool"] == "row_count" and "llm_explanation" not in r


def test_numbers_grounded_helper():
    assert copilot.numbers_grounded("Mean is 12.5 across 3 groups", '{"mean": 12.5}')
    assert not copilot.numbers_grounded("Mean is 13.9", '{"mean": 12.5}')
