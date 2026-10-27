"""
Analytics copilot.

Design principles
-----------------
* Numbers come only from deterministic, validated analysis functions ("tools").
  Every numeric answer carries the evidence (table/metric) it was computed from
  and is marked ``computed: true``.
* The copilot never executes model-generated code or shell commands.
* If a question is ambiguous (which metric? which dimension?) it asks a
  clarifying question with clickable options instead of guessing.
* Heavy or state-changing operations (training a model) are *proposed* as an
  action the user confirms, not run silently.
* An LLM (optional, Anthropic API) may be used to (a) map a free-form question
  to one of the tools and (b) narrate computed results. LLM text is labelled,
  and narrated numbers are checked against the computed evidence.
"""
from __future__ import annotations

import difflib
import json
import logging
import re
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from core.config import settings
from core.errors import ValidationFailed
from core.jsonsafe import records, to_jsonable

from . import stats as st
from .schema import analysis_columns, name_tokens, rank_measures

log = logging.getLogger("dataforge.copilot")


@dataclass
class Ctx:
    df: pd.DataFrame
    schema: List[Dict]
    dataset_name: str
    findings: Callable[[], List[Dict]]
    profile: Callable[[], Dict]
    latest_model: Optional[Dict] = None
    latest_forecast: Optional[Dict] = None
    latest_diagnostics: Optional[Dict] = None
    recommendations: Callable[[], List[Dict]] = lambda: []
    memory: Dict[str, Any] = field(default_factory=dict)
    forecast_fn: Optional[Callable[..., Dict]] = None
    exec_summary: Optional[Callable[[], Dict]] = None

    @property
    def smap(self) -> Dict[str, Dict]:
        return {c["name"]: c for c in self.schema}

    @property
    def cols(self) -> Dict[str, List[str]]:
        return analysis_columns(self.df, self.schema)


def _fmt(v) -> str:
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return "—"
    if isinstance(v, (int, np.integer)):
        return f"{int(v):,}"
    v = float(v)
    if abs(v) >= 1e6:
        return f"{v:,.0f}"
    if abs(v) >= 100:
        return f"{v:,.1f}"
    return f"{v:,.3g}" if abs(v) < 1 else f"{v:,.2f}"


def _reply(answer: str, tool: str, params: Optional[Dict] = None, evidence: Optional[List[Dict]] = None,
           suggestions: Optional[List[str]] = None, clarification: Optional[Dict] = None,
           actions: Optional[List[Dict]] = None, computed: bool = True) -> Dict[str, Any]:
    return to_jsonable({"answer": answer, "tool": tool, "params": params or {}, "evidence": evidence or [],
                        "computed": computed, "suggestions": suggestions or [], "clarification": clarification,
                        "actions": actions or [], "generated_by": "analysis engine"})


# ── Column resolution ───────────────────────────────────────────────────────────

def find_columns(text: str, columns: List[str]) -> List[str]:
    """Columns mentioned in the text, in order of appearance."""
    t = " " + re.sub(r"[^a-z0-9]+", " ", text.lower()) + " "
    hits: List[Tuple[int, str]] = []
    for c in sorted(columns, key=lambda c: -len(str(c))):
        variants = {str(c).lower(), " ".join(name_tokens(c))}
        for v in variants:
            v2 = re.sub(r"[^a-z0-9]+", " ", v).strip()
            if not v2:
                continue
            m = re.search(r" " + re.escape(v2) + r"(s|es)? ", t)  # tolerate plurals: "customer segments"
            pos = m.start() if m else -1
            if pos >= 0 and not any(p <= pos < p + len(h) + 2 for p, h in hits):
                hits.append((pos, str(c)))
                break
    if not hits:
        words = [w for w in re.findall(r"[a-z0-9]+", text.lower()) if len(w) > 3]
        tokmap: Dict[str, List[str]] = {}
        for c in columns:
            for tk in name_tokens(c):
                if len(tk) > 3:
                    tokmap.setdefault(tk, []).append(str(c))
        for w in words:
            m = difflib.get_close_matches(w, list(tokmap), n=1, cutoff=0.85)
            if m and len(tokmap[m[0]]) == 1:
                hits.append((text.lower().find(w), tokmap[m[0]][0]))
    seen, out = set(), []
    for _, c in sorted(hits):
        if c not in seen:
            seen.add(c)
            out.append(c)
    return out


def _metrics_dims(ctx: Ctx, mentioned: List[str]) -> Tuple[List[str], List[str]]:
    cols = ctx.cols
    metrics = [c for c in mentioned if c in cols["numeric"]]
    dims = [c for c in mentioned if c in cols["all_categorical"] or c in cols["datetime"]]
    return metrics, dims


# ── Tools ────────────────────────────────────────────────────────────────────

def tool_overview(ctx: Ctx, **_) -> Dict:
    p = ctx.profile()
    q = p["quality"]
    tc = p["type_counts"]
    ans = (f"**{ctx.dataset_name}** has **{p['shape']['rows']:,} rows** and **{p['shape']['columns']} columns** "
           f"({', '.join(f'{v} {k}' for k, v in tc.items())}).\n\n"
           f"- Data quality score: **{q['score']:.0f}/100 ({q['label']})**\n"
           f"- Missing cells: **{p['missing']['pct']:.1f}%** · duplicate rows: **{p['duplicates']['rows']:,}**")
    ev = [{"type": "metrics", "title": "Dataset metrics", "items": [
        {"label": "Rows", "value": p["shape"]["rows"]}, {"label": "Columns", "value": p["shape"]["columns"]},
        {"label": "Quality score", "value": q["score"]}, {"label": "Missing cells %", "value": p["missing"]["pct"]},
        {"label": "Duplicate rows", "value": p["duplicates"]["rows"]}]}]
    return _reply(ans, "overview", evidence=ev, suggestions=["What are the most important patterns?",
                                                             "Which columns have the most missing values?"])


def tool_missing(ctx: Ctx, **_) -> Dict:
    miss = ctx.df.isna().sum()
    miss = miss[miss > 0].sort_values(ascending=False)
    if miss.empty:
        return _reply("There are **no missing values** in this dataset.", "missing_values",
                      evidence=[{"type": "metrics", "items": [{"label": "Missing cells", "value": 0}]}])
    n = len(ctx.df)
    rows = [{"column": c, "missing": int(v), "missing_pct": round(v / n * 100, 2)} for c, v in miss.head(15).items()]
    top = rows[0]
    ans = (f"**{len(miss)} column(s)** have missing values. The most affected is **{top['column']}** with "
           f"{top['missing']:,} missing ({top['missing_pct']}%).")
    return _reply(ans, "missing_values", evidence=[{"type": "table", "title": "Missing values by column",
                                                    "columns": ["column", "missing", "missing_pct"], "rows": rows}],
                  suggestions=["Show me the cleaning suggestions", "Give me a dataset summary"])


def tool_duplicates(ctx: Ctx, **_) -> Dict:
    d = int(ctx.df.duplicated().sum())
    n = len(ctx.df)
    ans = (f"There are **{d:,} exact duplicate rows** ({d / n:.1%} of {n:,})." if d else
           "There are **no exact duplicate rows**.")
    return _reply(ans, "duplicates", evidence=[{"type": "metrics", "items": [{"label": "Duplicate rows", "value": d},
                                                                             {"label": "Rows", "value": n}]}])


def tool_columns(ctx: Ctx, **_) -> Dict:
    rows = [{"column": c["name"], "type": c["logical_type"], "missing_pct": c["missing_pct"], "unique": c["n_unique"]}
            for c in ctx.schema]
    return _reply(f"The dataset has **{len(rows)} columns**.", "columns",
                  evidence=[{"type": "table", "title": "Columns", "columns": ["column", "type", "missing_pct", "unique"],
                             "rows": rows}])


def tool_patterns(ctx: Ctx, **_) -> Dict:
    f = ctx.findings()[:6]
    if not f:
        return _reply("No statistically supported patterns were found.", "insights")
    lines = []
    for i, x in enumerate(f, 1):
        tag = {"association": "association", "observation": "observation", "hypothesis": "hypothesis"}[x["claim_type"]]
        lines.append(f"{i}. **{x['title']}** — {x['observation']} _({tag}; {x['evidence_text']})_")
    ans = "Here are the most important findings, ranked by evidence and impact:\n\n" + "\n".join(lines) + \
          "\n\nAssociations describe the data; they do not establish cause and effect."
    return _reply(ans, "insights", evidence=[{"type": "findings", "items": f}],
                  suggestions=["Recommend the next analysis", "Identify unusual observations"])


def tool_column_stats(ctx: Ctx, column: str, **_) -> Dict:
    s = ctx.df[column]
    if pd.api.types.is_numeric_dtype(s) and not pd.api.types.is_bool_dtype(s):
        v = s.dropna().astype(float)
        items = [("Count", int(v.count())), ("Missing", int(s.isna().sum())), ("Mean", v.mean()), ("Median", v.median()),
                 ("Std", v.std()), ("Min", v.min()), ("25%", v.quantile(0.25)), ("75%", v.quantile(0.75)),
                 ("Max", v.max()), ("Sum", v.sum())]
        ans = (f"**{column}**: mean {_fmt(v.mean())}, median {_fmt(v.median())}, range {_fmt(v.min())} – {_fmt(v.max())} "
               f"(n = {v.count():,}).")
        if abs(v.skew()) > 1:
            ans += f" The distribution is skewed (skewness {v.skew():.2f}), so the median is the better typical value."
    else:
        vc = s.astype("string").value_counts()
        items = [("Count", int(s.notna().sum())), ("Missing", int(s.isna().sum())), ("Distinct", int(len(vc))),
                 ("Most frequent", str(vc.index[0]) if len(vc) else "—"),
                 ("Most frequent share %", float(vc.iloc[0] / vc.sum() * 100) if len(vc) else None)]
        ans = f"**{column}** has {len(vc):,} distinct values; the most frequent is **{vc.index[0]}** ({vc.iloc[0] / vc.sum():.1%})." if len(vc) else f"**{column}** is empty."
    return _reply(ans, "column_stats", {"column": column},
                  evidence=[{"type": "metrics", "title": f"Statistics for {column}",
                             "items": [{"label": k, "value": to_jsonable(v)} for k, v in items]}],
                  suggestions=[f"Compare {column} across segments"] if column in ctx.cols["numeric"] else [])


def tool_compare(ctx: Ctx, metric: str, dimension: str, agg: Optional[str] = None, **_) -> Dict:
    df = ctx.df
    agg = agg or st.default_agg(metric, ctx.smap[metric]["semantic_type"])
    g = df.groupby(df[dimension].astype("string").fillna("(missing)"), observed=True)[metric]
    tab = pd.DataFrame({"n": g.count(), "sum": g.sum(), "mean": g.mean(), "median": g.median()})
    tab = tab.sort_values(agg if agg in tab.columns else "mean", ascending=False)
    test = st.group_test(df[metric], df[dimension])
    rows = [{"group": str(k), "n": int(r["n"]), "total": r["sum"], "mean": r["mean"], "median": r["median"],
             "share_of_total_pct": r["sum"] / tab["sum"].sum() * 100 if agg == "sum" and tab["sum"].sum() else None}
            for k, r in tab.head(20).iterrows()]
    top, bottom = rows[0], rows[-1]
    val = "total" if agg == "sum" else "mean"
    ans = (f"By **{dimension}**, the highest {val} {metric} is **{top['group']}** ({_fmt(top[val])}, n={top['n']}) and the "
           f"lowest is **{bottom['group']}** ({_fmt(bottom[val])}, n={bottom['n']}).")
    if test["kruskal_p"] is not None and test["eta_squared"] is not None:
        sig = test["kruskal_p"] < 0.05
        ans += (f"\n\nThe difference between groups is {'statistically significant' if sig else 'not statistically significant'} "
                f"(Kruskal–Wallis p = {test['kruskal_p']:.3g}; η² = {test['eta_squared']:.3f}, "
                f"{st.effect_label_eta(test['eta_squared'])} effect). This is an association, not proof that "
                f"{dimension} causes the difference.")
    small = [r["group"] for r in rows if r["n"] < 5]
    if small:
        ans += f"\n\n⚠ Groups with fewer than 5 rows ({', '.join(small[:3])}) give unreliable averages."
    ctx.memory.update({"metric": metric, "dimension": dimension})
    return _reply(ans, "compare_groups", {"metric": metric, "dimension": dimension, "agg": agg},
                  evidence=[{"type": "table", "title": f"{metric} by {dimension}",
                             "columns": ["group", "n", "total", "mean", "median"] + (["share_of_total_pct"] if agg == "sum" else []),
                             "rows": rows},
                            {"type": "chart", "chart": {"type": "bar", "x": dimension, "y": metric, "agg": agg}}],
                  suggestions=[f"Run diagnostics on {metric}", f"Show the distribution of {metric}"])


def tool_top(ctx: Ctx, column: str, n: int = 5, ascending: bool = False, **_) -> Dict:
    df = ctx.df
    s = df[column]
    if pd.api.types.is_numeric_dtype(s) and not pd.api.types.is_bool_dtype(s):
        sub = df.nsmallest(n, column) if ascending else df.nlargest(n, column)
        show = [column] + [c for c in df.columns if c != column][:6]
        word = "lowest" if ascending else "highest"
        return _reply(f"The {n} rows with the {word} **{column}**:", "top_rows", {"column": column, "n": n},
                      evidence=[{"type": "table", "title": f"Top {n} by {column}", "columns": show,
                                 "rows": records(sub[show])}])
    vc = s.astype("string").value_counts().head(n)
    rows = [{"value": str(k), "count": int(v), "pct": v / s.notna().sum() * 100} for k, v in vc.items()]
    return _reply(f"The most frequent values of **{column}**:", "top_values", {"column": column},
                  evidence=[{"type": "table", "columns": ["value", "count", "pct"], "rows": rows}])


def tool_correlations(ctx: Ctx, column: Optional[str] = None, **_) -> Dict:
    num = [c for c in ctx.cols["numeric"] if ctx.df[c].nunique() > 2]
    if len(num) < 2:
        return _reply("At least two numeric measure columns are needed for correlation analysis.", "correlations")
    pairs = []
    targets = [column] if column else num
    for a in targets:
        for b in num:
            if a == b or (not column and b <= a):
                continue
            t = st.corr_test(ctx.df[a], ctx.df[b], "spearman")
            if t["r"] is not None:
                pairs.append({"a": a, "b": b, "spearman": t["r"], "p_value": t["p_value"], "n": t["n"]})
    adj = st.benjamini_hochberg([p["p_value"] for p in pairs])
    for p, q in zip(pairs, adj):
        p["p_adjusted"] = q
    pairs.sort(key=lambda p: -abs(p["spearman"]))
    sig = [p for p in pairs if p["p_adjusted"] is not None and p["p_adjusted"] < 0.05 and abs(p["spearman"]) >= 0.3]
    if not sig:
        ans = "No moderate or strong correlations (|ρ| ≥ 0.3, adjusted p < 0.05) were found."
    else:
        ans = "Strongest associations (Spearman ρ, p adjusted for multiple tests):\n\n" + "\n".join(
            f"- **{p['a']} ↔ {p['b']}**: ρ = {p['spearman']:.2f} ({st.strength_label(p['spearman'])})" for p in sig[:6]) + \
            "\n\nCorrelation does not imply causation."
    return _reply(ans, "correlations", {"column": column},
                  evidence=[{"type": "table", "title": "Correlations", "columns": ["a", "b", "spearman", "p_adjusted", "n"],
                             "rows": pairs[:15]}])


def tool_outliers(ctx: Ctx, column: Optional[str] = None, **_) -> Dict:
    num = [column] if column else ctx.cols["numeric"]
    rows = []
    for c in num:
        o = st.outlier_summary(ctx.df[c])
        if o.get("iqr_mild") or o.get("mad"):
            rows.append({"column": c, "beyond_1.5_iqr": o["iqr_mild"], "beyond_3_iqr": o["iqr_extreme"],
                         "robust_z_gt_3.5": o["mad"], "fences": o.get("fences_mild")})
    rows.sort(key=lambda r: -r["beyond_3_iqr"])
    if not rows:
        return _reply("No outliers were found with the IQR or robust z-score rules.", "outliers")
    t = rows[0]
    ans = (f"Unusual values were found in **{len(rows)} column(s)**. The most affected is **{t['column']}** "
           f"({t['beyond_3_iqr']} extreme values beyond 3×IQR, {t['beyond_1.5_iqr']} beyond 1.5×IQR). "
           "For records that are unusual across several columns at once, run Anomaly detection in the Model step.")
    return _reply(ans, "outliers", {"column": column},
                  evidence=[{"type": "table", "title": "Outliers by column",
                             "columns": ["column", "beyond_1.5_iqr", "beyond_3_iqr", "robust_z_gt_3.5"], "rows": rows}],
                  actions=[{"label": "Open anomaly detection", "kind": "navigate", "payload": {"page": "model", "tab": "anomaly"}}])


def tool_train(ctx: Ctx, target: str, **_) -> Dict:
    lt = ctx.smap[target]["logical_type"]
    if lt in ("identifier", "text", "empty", "constant"):
        return _reply(f"**{target}** is a {lt} column and is not a sensible prediction target. Choose a measure or a "
                      "category to predict.", "train_model", {"target": target})
    kind = "regression" if lt == "numeric" and ctx.df[target].nunique() > 10 else "classification"
    ans = (f"I can train and compare models to predict **{target}** ({kind}). Models are selected by cross-validation "
           f"on a training split and evaluated once on a held-out test set, next to a baseline. "
           "Training runs in the background; start it when ready.")
    return _reply(ans, "train_model", {"target": target},
                  actions=[{"label": f"Train models for {target}", "kind": "train", "payload": {"target": target}},
                           {"label": "Open the Model step", "kind": "navigate", "payload": {"page": "model", "target": target}}])


def tool_explain_model(ctx: Ctx, **_) -> Dict:
    m = ctx.latest_model
    if not m:
        return _reply("No model has been trained in this workspace yet. Ask me to build a model for a target first.",
                      "explain_model")
    r = m["result"]
    items = r["feature_importance"]["items"][:8]
    pm = r["primary_metric_label"]
    tm = r["test_metrics"]
    key_metric = {"regression": ("r2", "R²"), "binary": ("roc_auc", "ROC-AUC"), "multiclass": ("f1_macro", "F1 macro")}[r["problem_type"]]
    lines = "\n".join(f"- **{i['feature']}** — importance {i['importance']:.3g}" + (f"; {i['direction']}" if i.get("direction") else "")
                      for i in items if i["importance"] > 0) or "- No feature had a measurable effect on the score."
    ans = (f"The latest model predicts **{r['target']}** using **{r['selected']['model']}** "
           f"(test {key_metric[1]} = {_fmt(tm.get(key_metric[0]))}). Its strongest predictive features "
           f"(permutation importance on the test set, metric {pm}):\n\n{lines}\n\n"
           "Importance shows what the model relies on, not what causes the outcome.")
    return _reply(ans, "explain_model", {"model_id": m.get("id")},
                  evidence=[{"type": "table", "title": "Feature importance", "columns": ["feature", "importance", "std", "direction"],
                             "rows": items}])


def tool_forecast(ctx: Ctx, metric: Optional[str], date_col: Optional[str] = None, horizon: int = 6, **_) -> Dict:
    if ctx.forecast_fn is None:
        return _reply("Forecasting is not available.", "forecast", computed=False)
    dates = ctx.cols["datetime"]
    if not dates:
        return _reply("Forecasting needs a date/time column, and this dataset has none. If a column holds dates as text, "
                      "convert it in the Clean step.", "forecast")
    date_col = date_col or dates[0]
    r = ctx.forecast_fn(date_col=date_col, target=metric, horizon=horizon)
    if not r.get("available"):
        return _reply(f"A forecast is not possible: {r.get('reason')}", "forecast", {"metric": metric})
    fc = r["forecasts"][r["selected_method"]]
    rows = [{"period": t[:10], "forecast": f, "lower80": lo, "upper80": hi}
            for t, f, lo, hi in zip(r["future_index"], fc["forecast"], fc["lower80"], fc["upper80"])]
    ans = (f"Forecast of **{r['aggregation']} {metric or 'record count'}** per {r['frequency_label']} for the next "
           f"{r['horizon']} periods using **{r['selected_label']}** (selected by backtesting).\n\n" + " ".join(r["explanation"][:2])
           + "\n\nThe ranges are approximate 80% prediction intervals — the future value is uncertain.")
    ctx.memory["metric"] = metric
    return _reply(ans, "forecast", {"metric": metric, "date_col": date_col, "horizon": r["horizon"]},
                  evidence=[{"type": "table", "title": "Forecast", "columns": ["period", "forecast", "lower80", "upper80"],
                             "rows": rows}],
                  actions=[{"label": "Open Forecasting", "kind": "navigate", "payload": {"page": "forecast", "target": metric}}])


def tool_summary(ctx: Ctx, **_) -> Dict:
    if ctx.exec_summary is None:
        return tool_overview(ctx)
    s = ctx.exec_summary()
    body = "\n\n".join(f"**{sec['title']}** — {sec['content']}" for sec in s["sections"])
    return _reply(f"### {s['headline']}\n\n{body}", "management_summary",
                  evidence=[{"type": "metrics", "items": [{"label": "Quality score", "value": s.get("quality_score")}]}],
                  actions=[{"label": "Build a report", "kind": "navigate", "payload": {"page": "report"}}])


def tool_next(ctx: Ctx, **_) -> Dict:
    recs = ctx.recommendations()[:4]
    if not recs:
        return _reply("Everything looks covered. You could export a report.", "next_steps")
    ans = "Recommended next steps:\n\n" + "\n".join(f"{i}. **{r['title']}** — {r['reason']}" for i, r in enumerate(recs, 1))
    return _reply(ans, "next_steps", actions=[{"label": r["title"], "kind": "navigate", "payload": {"page": r["page"]}}
                                              for r in recs if r.get("page")])


def tool_rowcount(ctx: Ctx, **_) -> Dict:
    return _reply(f"The dataset has **{len(ctx.df):,} rows** and **{len(ctx.df.columns)} columns**.", "row_count",
                  evidence=[{"type": "metrics", "items": [{"label": "Rows", "value": len(ctx.df)},
                                                          {"label": "Columns", "value": len(ctx.df.columns)}]}])


def tool_help(ctx: Ctx, **_) -> Dict:
    return _reply(
        "I answer questions with computed results from your current dataset. Try:\n\n"
        "- *What are the most important patterns in this dataset?*\n- *Which columns have the most missing values?*\n"
        "- *Compare revenue across customer segments*\n- *What is the average delivery days?*\n"
        "- *Identify unusual observations*\n- *Forecast revenue*\n- *Build a model to predict churn*\n"
        "- *Explain the model's strongest features*\n- *Generate a management summary*\n- *What should I analyse next?*",
        "help", computed=False)


TOOLS: Dict[str, Callable] = {
    "overview": tool_overview, "missing_values": tool_missing, "duplicates": tool_duplicates, "columns": tool_columns,
    "insights": tool_patterns, "column_stats": tool_column_stats, "compare_groups": tool_compare, "top": tool_top,
    "correlations": tool_correlations, "outliers": tool_outliers, "train_model": tool_train,
    "explain_model": tool_explain_model, "forecast": tool_forecast, "management_summary": tool_summary,
    "next_steps": tool_next, "row_count": tool_rowcount, "help": tool_help,
}

TOOL_SPECS = [
    {"name": "overview", "description": "Dataset size, column types and quality score.", "params": {}},
    {"name": "missing_values", "description": "Missing values per column.", "params": {}},
    {"name": "duplicates", "description": "Count of duplicate rows.", "params": {}},
    {"name": "columns", "description": "List columns with their types.", "params": {}},
    {"name": "insights", "description": "Most important statistically supported patterns.", "params": {}},
    {"name": "column_stats", "description": "Descriptive statistics of one column.", "params": {"column": "column name"}},
    {"name": "compare_groups", "description": "Compare a numeric metric across groups of a categorical column.",
     "params": {"metric": "numeric column", "dimension": "categorical column", "agg": "sum|mean (optional)"}},
    {"name": "top", "description": "Top rows by a numeric column, or most frequent values of a category.",
     "params": {"column": "column", "n": "integer", "ascending": "boolean"}},
    {"name": "correlations", "description": "Correlations between numeric columns.", "params": {"column": "optional column"}},
    {"name": "outliers", "description": "Unusual values per column.", "params": {"column": "optional column"}},
    {"name": "train_model", "description": "Propose training a model for a target column.", "params": {"target": "column"}},
    {"name": "explain_model", "description": "Explain the latest model's important features.", "params": {}},
    {"name": "forecast", "description": "Forecast a numeric metric over time.", "params": {"metric": "numeric column", "horizon": "integer"}},
    {"name": "management_summary", "description": "Executive/management summary.", "params": {}},
    {"name": "next_steps", "description": "Recommend the next analysis.", "params": {}},
    {"name": "row_count", "description": "Number of rows and columns.", "params": {}},
]


# ── Intent routing ──────────────────────────────────────────────────────────────

def _has(msg: str, *words: str) -> bool:
    return any(re.search(r"\b" + w, msg) for w in words)


def route(ctx: Ctx, message: str) -> Tuple[Optional[str], Dict[str, Any], Optional[Dict]]:
    """Return (tool, params, clarification)."""
    msg = message.lower().strip()
    mentioned = find_columns(message, [c["name"] for c in ctx.schema])
    metrics, dims = _metrics_dims(ctx, mentioned)
    cols = ctx.cols
    measures = rank_measures(cols["numeric"], ctx.smap)

    def ask(question: str, options: List[str], template: str, kind: str) -> Tuple[None, Dict, Dict]:
        return None, {}, {"question": question, "options": [{"label": o, "message": template.format(o)} for o in options[:8]],
                          "kind": kind}

    if re.fullmatch(r"(hi|hello|hey|help|\?|what can you do\??)", msg) or _has(msg, "what can you", "help me", "capabilit"):
        return "help", {}, None
    if _has(msg, "management summary", "executive summary", "summary for management", "summarize for", "board"):
        return "management_summary", {}, None
    if _has(msg, "next step", "what should i", "recommend", "what next", "next analys"):
        return "next_steps", {}, None
    if _has(msg, "explain the model", "model's", "feature importance", "important feature", "strongest predict",
            "why does the model", "explain model"):
        return "explain_model", {}, None
    if _has(msg, "forecast", "project", "next month", "next quarter", "future"):
        metric = metrics[0] if metrics else ctx.memory.get("metric")
        if not metric and len(measures) > 1:
            return ask("Which metric should I forecast?", measures, "Forecast {}", "metric")
        hz = re.search(r"(\d+)\s*(period|month|week|day|quarter)", msg)
        return "forecast", {"metric": metric or (measures[0] if measures else None),
                            "horizon": int(hz.group(1)) if hz else 6}, None
    if _has(msg, "build a model", "train", "predict", "model to", "classifier", "regression"):
        target = mentioned[0] if mentioned else ctx.memory.get("target")
        if not target:
            opts = measures[:4] + [c for c in cols["categorical"] if ctx.smap[c]["n_unique"] <= 10][:4]
            return ask("Which column should the model predict?", opts, "Build a model to predict {}", "target")
        return "train_model", {"target": target}, None
    if _has(msg, "missing", "null", "nan", "empty value", "blank"):
        return "missing_values", {}, None
    if _has(msg, "duplicate"):
        return "duplicates", {}, None
    if _has(msg, "outlier", "anomal", "unusual", "weird", "strange", "extreme"):
        return "outliers", {"column": metrics[0] if metrics else None}, None
    if _has(msg, "correlat", "relationship", "related", "associat"):
        return "correlations", {"column": metrics[0] if metrics else None}, None
    if _has(msg, "pattern", "insight", "interesting", "important", "finding", "key takeaway"):
        return "insights", {}, None
    if _has(msg, "compare", "across", "by ", "per ", "breakdown", "split", "segment", "differ"):
        metric = metrics[0] if metrics else ctx.memory.get("metric")
        dim = dims[0] if dims else None
        if dim is None:
            for c in cols["categorical"]:
                if any(tk in msg for tk in name_tokens(c) if len(tk) > 3) and c not in metrics:
                    dim = c
                    break
        if metric and not dim:
            opts = [c for c in cols["categorical"] if 2 <= ctx.smap[c]["n_unique"] <= 30]
            return ask(f"Which dimension should I compare {metric} across?", opts, f"Compare {metric} across {{}}", "dimension")
        if dim and not metric:
            return ask(f"Which metric should I compare across {dim}?", measures, f"Compare {{}} across {dim}", "metric")
        if metric and dim:
            if dim in cols["datetime"]:
                return "forecast", {"metric": metric, "date_col": dim}, None
            agg = "sum" if _has(msg, "total", "sum") else ("mean" if _has(msg, "average", "mean", "avg") else None)
            return "compare_groups", {"metric": metric, "dimension": dim, "agg": agg}, None
    if _has(msg, "top", "highest", "largest", "biggest", "best", "lowest", "smallest", "worst", "bottom", "most common", "most frequent"):
        col = mentioned[0] if mentioned else (measures[0] if measures else None)
        if col:
            n = re.search(r"\b(\d{1,3})\b", msg)
            return "top", {"column": col, "n": min(int(n.group(1)), 100) if n else 5,
                           "ascending": _has(msg, "lowest", "smallest", "worst", "bottom")}, None
    if _has(msg, "how many rows", "row count", "number of rows", "how big", "size of"):
        return "row_count", {}, None
    if _has(msg, "column", "schema", "field", "variables", "data types"):
        return "columns", {}, None
    if mentioned and _has(msg, "average", "mean", "median", "sum", "total", "max", "min", "statistic", "distribution",
                          "describe", "tell me about", "what is", "how"):
        return "column_stats", {"column": mentioned[0]}, None
    if _has(msg, "summary", "overview", "describe", "tell me about", "dataset"):
        return "overview", {}, None
    if mentioned:
        return "column_stats", {"column": mentioned[0]}, None
    return None, {}, None


def _validate_params(ctx: Ctx, tool: str, params: Dict[str, Any]) -> Dict[str, Any]:
    cols = [c["name"] for c in ctx.schema]
    clean: Dict[str, Any] = {}
    for k, v in (params or {}).items():
        if k in ("column", "metric", "dimension", "target", "date_col"):
            if v is None:
                clean[k] = None
                continue
            if v not in cols:
                m = difflib.get_close_matches(str(v), cols, n=1, cutoff=0.8)
                if not m:
                    raise ValidationFailed(f"Column '{v}' does not exist in this dataset.")
                v = m[0]
            clean[k] = v
        elif k in ("n", "horizon"):
            clean[k] = max(1, min(int(v), 100))
        elif k == "ascending":
            clean[k] = bool(v)
        elif k == "agg":
            clean[k] = v if v in ("sum", "mean", "median") else None
    if tool == "compare_groups":
        if clean.get("metric") not in ctx.cols["numeric"]:
            raise ValidationFailed(f"'{clean.get('metric')}' is not a numeric measure.")
        if clean.get("dimension") not in ctx.cols["all_categorical"]:
            raise ValidationFailed(f"'{clean.get('dimension')}' is not a categorical column.")
    if tool in ("column_stats", "top") and not clean.get("column"):
        raise ValidationFailed("Which column do you mean?")
    if tool == "train_model" and not clean.get("target"):
        raise ValidationFailed("Which column should be predicted?")
    return clean


# ── Optional LLM ────────────────────────────────────────────────────────────────

def _llm_call(system: str, messages: List[Dict], tools: Optional[List[Dict]] = None, max_tokens: int = 600) -> Optional[Dict]:
    if not settings.llm_enabled:
        return None
    import httpx
    body: Dict[str, Any] = {"model": settings.anthropic_model, "max_tokens": max_tokens, "system": system,
                            "messages": messages}
    if tools:
        body["tools"] = tools
    try:
        with httpx.Client(timeout=settings.llm_timeout_s) as client:
            r = client.post(f"{settings.anthropic_base_url.rstrip('/')}/v1/messages",
                            headers={"x-api-key": settings.anthropic_api_key, "anthropic-version": "2023-06-01",
                                     "content-type": "application/json"}, json=body)
        if r.status_code != 200:
            log.warning("LLM request failed with HTTP %s", r.status_code)
            return None
        return r.json()
    except Exception as e:  # network errors, timeouts
        log.warning("LLM request failed: %s", type(e).__name__)
        return None


def _schema_brief(ctx: Ctx) -> str:
    return "\n".join(f"- {c['name']} ({c['logical_type']}, {c['n_unique']} distinct, {c['missing_pct']}% missing)"
                     for c in ctx.schema[:80])


def llm_route(ctx: Ctx, message: str, history: List[Dict]) -> Optional[Tuple[str, Dict[str, Any]]]:
    tools = [{"name": t["name"], "description": t["description"],
              "input_schema": {"type": "object", "properties": {k: {"type": "string" if k not in ("n", "horizon") else "integer",
                                                                    "description": d} for k, d in t["params"].items()}}}
             for t in TOOL_SPECS]
    system = ("You route data-analysis questions to analysis tools. Choose exactly one tool and fill its parameters "
              "with exact column names from the schema. Never invent columns. Dataset columns:\n" + _schema_brief(ctx))
    msgs = [{"role": h["role"], "content": str(h["content"])[:2000]} for h in history[-6:] if h.get("role") in ("user", "assistant")]
    msgs.append({"role": "user", "content": message[:4000]})
    data = _llm_call(system, msgs, tools)
    if not data:
        return None
    for block in data.get("content", []):
        if block.get("type") == "tool_use" and block.get("name") in TOOLS:
            return block["name"], block.get("input") or {}
    return None


_NUM = re.compile(r"-?\d[\d,]*\.?\d*")


def _numbers(text: str) -> List[float]:
    out = []
    for m in _NUM.findall(text):
        try:
            out.append(float(m.replace(",", "")))
        except ValueError:
            pass
    return out


def numbers_grounded(narration: str, evidence_text: str) -> bool:
    """Every number in the narration must appear (within rounding) in the computed evidence."""
    allowed = _numbers(evidence_text)
    for v in _numbers(narration):
        if abs(v) < 10 and float(v).is_integer():
            continue  # small counts such as "2 groups", list numbering
        if not any(abs(v - a) <= max(0.01 * abs(a), 0.051) for a in allowed):
            return False
    return True


def llm_narrate(reply: Dict[str, Any], message: str) -> Optional[str]:
    facts = json.dumps({"answer": reply["answer"], "evidence": reply["evidence"]}, default=str)[:12000]
    system = ("You explain computed data-analysis results to a business user in 2-4 sentences. Use ONLY the numbers in "
              "the provided results; do not compute new numbers, do not speculate about causes beyond saying what "
              "would need to be checked.")
    data = _llm_call(system, [{"role": "user", "content": f"Question: {message}\n\nComputed results (JSON): {facts}"}],
                     max_tokens=300)
    if not data:
        return None
    text = "".join(b.get("text", "") for b in data.get("content", []) if b.get("type") == "text").strip()
    if not text or not numbers_grounded(text, facts):
        return None
    return text


# ── Entry point ─────────────────────────────────────────────────────────────────

def answer(ctx: Ctx, message: str, history: Optional[List[Dict]] = None) -> Dict[str, Any]:
    message = (message or "").strip()
    if not message:
        raise ValidationFailed("Please type a question.")
    if len(message) > 2000:
        raise ValidationFailed("Questions are limited to 2,000 characters.")
    history = history or []
    tool, params, clarification = route(ctx, message)
    routed_by = "rules"
    if clarification:
        r = _reply(clarification["question"], "clarify", computed=False, clarification=clarification)
        r["routed_by"] = routed_by
        return r
    if tool is None and settings.llm_enabled:
        got = llm_route(ctx, message, history)
        if got:
            tool, params = got
            routed_by = "llm"
    if tool is None:
        r = tool_help(ctx)
        r["answer"] = ("I couldn't match that question to a supported analysis. " + r["answer"]
                       + ("" if settings.llm_enabled else "\n\n_Free-form questions can be routed by an AI model if "
                                                           "ANTHROPIC_API_KEY is configured on the server._"))
        r["routed_by"] = "none"
        return r
    try:
        params = _validate_params(ctx, tool, params)
        reply = TOOLS[tool](ctx, **params)
    except ValidationFailed as e:
        r = _reply(f"I couldn't run that analysis: {e.message}", tool, params, computed=False)
        r["routed_by"] = routed_by
        return r
    reply["routed_by"] = routed_by
    if settings.llm_enabled and reply.get("computed") and reply["tool"] not in ("help", "clarify"):
        narr = llm_narrate(reply, message)
        if narr:
            reply["llm_explanation"] = narr
            reply["generated_by"] = "analysis engine + AI explanation"
    return reply
