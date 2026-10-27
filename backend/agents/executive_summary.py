"""
Executive summary and next-step recommendations.

The summary is assembled deterministically from computed artifacts (profile,
findings, the latest model, forecast and diagnostics). If an LLM is configured
it may rewrite the prose, but the rewrite is only accepted when every number it
contains can be traced back to the computed facts; otherwise the template text
is kept.
"""
from __future__ import annotations

import json
import logging
from typing import Any, Dict, List, Optional

import pandas as pd

from core.config import settings
from core.jsonsafe import to_jsonable

log = logging.getLogger("dataforge.summary")


def _f(v, d=2):
    if v is None:
        return "—"
    v = float(v)
    return f"{v:,.0f}" if abs(v) >= 1000 else f"{v:,.{d}f}"


def recommendations(state: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Rule-based next steps from the current workspace state."""
    recs: List[Dict[str, Any]] = []
    issues = state.get("issues", [])
    high = [i for i in issues if i["severity"] == "high" and i.get("recommended")]
    if high and not state.get("n_ops"):
        recs.append({"title": "Review data-quality issues", "page": "clean", "priority": 1,
                     "reason": f"{len(high)} high-severity issue(s) found, e.g. {high[0]['title']}."})
    if not state.get("profile_viewed"):
        recs.append({"title": "Profile the data", "page": "profile", "priority": 2,
                     "reason": "Check distributions, outliers and correlations before drawing conclusions."})
    if state.get("findings"):
        top = state["findings"][0]
        if top.get("category") in ("segment", "trend") and not state.get("has_diagnostics"):
            measure = top["columns"][0] if top.get("columns") else None
            recs.append({"title": f"Diagnose drivers of {measure}" if measure else "Run diagnostics", "page": "diagnose",
                         "priority": 3, "reason": f"{top['title']} — check confounders before acting on it."})
    if state.get("has_datetime") and not state.get("has_forecast"):
        recs.append({"title": "Forecast a key metric", "page": "forecast", "priority": 4,
                     "reason": "The dataset has a date column; backtested forecasts quantify where the metric is heading."})
    if not state.get("has_model") and state.get("rows", 0) >= 30:
        recs.append({"title": "Train a predictive model", "page": "model", "priority": 5,
                     "reason": "Compare models with cross-validation to see which columns predict your target."})
    if state.get("has_model") and not state.get("has_predictions"):
        recs.append({"title": "Score new data", "page": "predict", "priority": 6,
                     "reason": "Use the trained model on new records with schema validation."})
    if not state.get("n_rules"):
        recs.append({"title": "Add data-quality rules", "page": "clean", "priority": 7,
                     "reason": "Rules (ranges, allowed values, uniqueness) catch problems in future data loads."})
    recs.append({"title": "Export a report", "page": "report", "priority": 9,
                 "reason": "Share findings, methodology and limitations as PDF, Markdown or JSON."})
    recs.sort(key=lambda r: r["priority"])
    return recs


def build_summary(dataset_name: str, profile: Dict[str, Any], findings: List[Dict[str, Any]], ops: List[Dict],
                  model: Optional[Dict] = None, forecast: Optional[Dict] = None, diagnostics: Optional[Dict] = None,
                  recs: Optional[List[Dict]] = None) -> Dict[str, Any]:
    q = profile["quality"]
    rows, cols = profile["shape"]["rows"], profile["shape"]["columns"]
    sections: List[Dict[str, Any]] = []
    sections.append({"id": "overview", "title": "Dataset overview", "icon": "database",
                     "content": f"'{dataset_name}' contains {rows:,} rows and {cols} columns "
                                f"({', '.join(f'{v} {k}' for k, v in profile['type_counts'].items())}).",
                     "evidence": {"rows": rows, "columns": cols}})
    qc = q["components"]
    issues = profile.get("issues_count", {})
    sections.append({"id": "quality", "title": "Data quality", "icon": "shield",
                     "content": f"Quality score {q['score']:.0f}/100 ({q['label']}): completeness "
                                f"{qc['completeness']['value']:.1f}%, row uniqueness {qc['uniqueness']['value']:.1f}%, "
                                f"validity {qc['validity']['value']:.1f}%. {issues.get('high', 0)} high- and "
                                f"{issues.get('medium', 0)} medium-severity issue(s) detected; "
                                f"{len(ops)} cleaning step(s) applied.",
                     "evidence": {"score": q["score"], "components": qc}})
    key = [f for f in findings if f.get("category") != "data_quality"][:3]
    if key:
        sections.append({"id": "findings", "title": "Key findings", "icon": "zap",
                         "content": " ".join(f"({i}) {f['title']}: {f['observation']}" for i, f in enumerate(key, 1)),
                         "evidence": [{"title": f["title"], "evidence": f["evidence_text"], "claim_type": f["claim_type"]}
                                      for f in key]})
    if model:
        r = model["result"]
        tm, bm = r["test_metrics"], r.get("baseline_test_metrics") or {}
        k = {"regression": ("r2", "R²"), "binary": ("roc_auc", "ROC-AUC"), "multiclass": ("f1_macro", "macro F1")}[r["problem_type"]]
        k2 = ("rmse", "RMSE") if r["problem_type"] == "regression" else ("balanced_accuracy", "balanced accuracy")
        top_f = [i["feature"] for i in r["feature_importance"]["items"][:3] if i["importance"] > 0]
        sections.append({"id": "model", "title": "Predictive model", "icon": "cpu",
                         "content": f"A {r['selected']['model']} model predicts '{r['target']}' with test {k[1]} "
                                    f"{_f(tm.get(k[0]), 3)} and {k2[1]} {_f(tm.get(k2[0]), 3)} "
                                    f"(baseline {k2[1]} {_f(bm.get(k2[0]), 3)}), evaluated on {r['rows']['test']:,} held-out rows."
                                    + (f" Most influential features: {', '.join(top_f)}." if top_f else ""),
                         "evidence": {"test_metrics": tm, "baseline": bm}})
    if forecast and forecast.get("available"):
        fc = forecast["forecasts"][forecast["selected_method"]]
        sections.append({"id": "forecast", "title": "Forecast", "icon": "trending-up",
                         "content": f"{forecast['selected_label']} forecasts the {forecast['aggregation']} of "
                                    f"'{forecast.get('target') or 'records'}' at {_f(fc['forecast'][0])} for the next "
                                    f"{forecast['frequency_label']} (80% interval {_f(fc['lower80'][0])}–{_f(fc['upper80'][0])}). "
                                    + forecast["explanation"][1],
                         "evidence": {"method": forecast["selected_method"], "horizon": forecast["horizon"]}})
    if diagnostics and diagnostics.get("findings"):
        d0 = diagnostics["findings"][0]
        sections.append({"id": "diagnostics", "title": f"Drivers of {diagnostics['outcome']}", "icon": "search",
                         "content": f"{d0['title']} ({d0['label']}): {d0['evidence']}. " + diagnostics["causal_note"].split(".")[0] + ".",
                         "evidence": {"label": d0["label"]}})
    if recs:
        sections.append({"id": "recommendations", "title": "Recommended next steps", "icon": "target",
                         "content": " ".join(f"({i}) {r['title']} — {r['reason']}" for i, r in enumerate(recs[:4], 1)),
                         "evidence": None})
    sections.append({"id": "limitations", "title": "Limitations", "icon": "alert-triangle",
                     "content": "Findings are statistical associations in observational data and do not establish cause and "
                                "effect. Results depend on the completeness and accuracy of the uploaded data"
                                + (" and on the cleaning steps applied." if ops else "."),
                     "evidence": None})
    if key:
        headline = key[0]["title"]
    elif model:
        headline = f"Model for '{model['result']['target']}' ready ({model['result']['selected']['model']})"
    else:
        headline = f"{rows:,} records analysed — data quality {q['label'].lower()} ({q['score']:.0f}/100)"
    out = {"headline": headline, "sections": sections, "quality_score": round(q["score"]), "quality_label": q["label"],
           "generated_by": "template"}
    if settings.llm_enabled:
        polished = _polish(out)
        if polished:
            return polished
    return to_jsonable(out)


def _polish(summary: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    from .copilot import _llm_call, numbers_grounded
    facts = json.dumps({"headline": summary["headline"],
                        "sections": [{"id": s["id"], "title": s["title"], "content": s["content"]} for s in summary["sections"]]},
                       ensure_ascii=False)
    system = ("Rewrite the summary sections for an executive audience: clearer, concise prose. Keep the same section ids. "
              "Use only numbers present in the input; do not add facts, causes or recommendations. "
              'Return JSON: {"headline": str, "sections": [{"id": str, "content": str}]}')
    data = _llm_call(system, [{"role": "user", "content": facts}], max_tokens=1200)
    if not data:
        return None
    try:
        text = "".join(b.get("text", "") for b in data.get("content", []) if b.get("type") == "text").strip()
        text = text.strip("`").removeprefix("json").strip()
        new = json.loads(text)
        if not numbers_grounded(json.dumps(new), facts):
            log.info("Discarded LLM summary: contained numbers not present in the computed facts")
            return None
        by_id = {s["id"]: s for s in new.get("sections", [])}
        out = json.loads(json.dumps(summary, default=str))
        out["headline"] = str(new.get("headline") or out["headline"])[:300]
        for s in out["sections"]:
            if s["id"] in by_id and isinstance(by_id[s["id"]].get("content"), str):
                s["content"] = by_id[s["id"]]["content"][:2000]
        out["generated_by"] = "template + AI rewrite (numbers verified)"
        return out
    except Exception:
        return None


# v4 compatibility ------------------------------------------------------------------

async def generate_executive_summary(df: pd.DataFrame, eda: Dict, insights: List) -> Dict:
    from .cleaner import detect_issues
    from .insights import generate_findings
    from .profiling import profile_dataset
    from .schema import infer_schema
    schema = infer_schema(df)
    issues = detect_issues(df, schema)
    prof = profile_dataset(df, schema, issues)
    f = generate_findings(df, schema, issues)
    return build_summary("dataset", prof, f, [])
