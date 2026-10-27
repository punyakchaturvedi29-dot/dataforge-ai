"""
Report generation.

A report is first built as a structured document (sections of typed blocks:
paragraph, bullets, metrics, table, chart) from computed artifacts only, and
then rendered to PDF, Markdown or JSON. Users choose which sections, findings,
model and charts to include.

All user-derived text (column names, category values, file names) is escaped
before it reaches ReportLab's mini-markup parser.
"""
from __future__ import annotations

import io
import logging
import os
import time
from typing import Any, Dict, List, Optional
from xml.sax.saxutils import escape

from core.jsonsafe import to_jsonable

log = logging.getLogger("dataforge.report")

ALL_SECTIONS = ["summary", "overview", "quality", "methodology", "findings", "charts", "model", "forecast",
                "diagnostics", "rules", "recommendations", "limitations", "reproducibility"]


def _num(v, d=3):
    if v is None:
        return "—"
    if isinstance(v, bool):
        return str(v)
    if isinstance(v, int):
        return f"{v:,}"
    try:
        v = float(v)
    except (TypeError, ValueError):
        return str(v)
    if abs(v) >= 1000:
        return f"{v:,.0f}"
    return f"{v:.{d}g}"


def build_report(*, title: str, dataset: Dict[str, Any], profile: Dict, findings: List[Dict], summary: Dict,
                 ops: List[Dict], ingestion: Dict, charts: List[Dict], model: Optional[Dict], forecast: Optional[Dict],
                 diagnostics: Optional[Dict], rules: Optional[Dict], recs: List[Dict], versions: Dict,
                 sections: Optional[List[str]] = None, workspace_id: str = "", app_version: str = "") -> Dict[str, Any]:
    want = [s for s in (sections or ALL_SECTIONS) if s in ALL_SECTIONS]
    S: List[Dict[str, Any]] = []

    def sec(sid, ttl, blocks):
        if sid in want and blocks:
            S.append({"id": sid, "title": ttl, "blocks": blocks})

    sec("summary", "Executive summary", [{"type": "paragraph", "text": summary["headline"], "style": "lead"}] +
        [{"type": "paragraph", "text": f"{s['title']}: {s['content']}"} for s in summary["sections"]
         if s["id"] not in ("limitations",)][:6])

    q = profile["quality"]
    sec("overview", "Dataset overview", [
        {"type": "metrics", "items": [
            {"label": "Rows", "value": profile["shape"]["rows"]}, {"label": "Columns", "value": profile["shape"]["columns"]},
            {"label": "Original rows", "value": dataset.get("original_rows")},
            {"label": "Missing cells", "value": f"{profile['missing']['pct']:.2f}%"},
            {"label": "Duplicate rows", "value": profile["duplicates"]["rows"]},
            {"label": "Quality score", "value": f"{q['score']:.0f}/100 ({q['label']})"}]},
        {"type": "table", "title": "Columns", "columns": ["Column", "Type", "Missing %", "Distinct", "Summary"],
         "rows": [[c["name"], c["logical_type"], f"{c['missing_pct']:.1f}", c["n_unique"], _col_summary(c)]
                  for c in profile["columns"][:60]]},
    ])

    qc = q["components"]
    qblocks = [{"type": "paragraph", "text": q["formula"]},
               {"type": "table", "title": "Quality components", "columns": ["Component", "Value", "Weight"],
                "rows": [[k.title(), f"{v['value']:.1f}%", v["weight"]] for k, v in qc.items()]}]
    if profile["missing"]["by_column"]:
        qblocks.append({"type": "table", "title": "Missing values", "columns": ["Column", "Missing", "%"],
                        "rows": [[m["column"], m["missing"], f"{m['pct']:.1f}"] for m in profile["missing"]["by_column"][:20]]})
    sec("quality", "Data quality assessment", qblocks)

    meth = [f"Source file: {ingestion.get('filename')} ({ingestion.get('format', '').upper()}, "
            f"{_num(ingestion.get('size_bytes'))} bytes, SHA-256 {str(ingestion.get('sha256', ''))[:16]}…).",
            f"Ingestion: encoding {ingestion.get('encoding') or 'n/a'}, delimiter {ingestion.get('delimiter') or 'n/a'}, "
            f"{ingestion.get('malformed_rows', {}).get('count', 0)} malformed row(s) skipped, "
            f"{len(ingestion.get('column_changes', []))} column name(s) repaired."]
    if ops:
        meth.append(f"{len(ops)} cleaning step(s) were applied to a copy of the original data, in this order:")
    mblocks = [{"type": "bullets", "items": meth}]
    if ops:
        mblocks.append({"type": "table", "columns": ["#", "Step", "Parameters", "Rows removed", "Values changed"],
                        "rows": [[i, o["label"], _params(o.get("params", {})), o["impact"]["rows_removed"],
                                  o["impact"]["cells_modified"] + o["impact"]["values_filled"] + o["impact"]["values_nullified"]]
                                 for i, o in enumerate(ops, 1)]})
    mblocks.append({"type": "bullets", "items": [
        "Statistics: Spearman/Pearson correlations, Kruskal–Wallis and ANOVA group tests, effect sizes (η², Cohen's d); "
        "p-values are adjusted for multiple comparisons with the Benjamini–Hochberg procedure.",
        "Insights are ranked by a priority score combining evidence strength and practical impact."]})
    sec("methodology", "Methodology", mblocks)

    if findings:
        fb = []
        for f in findings:
            fb.append({"type": "finding", "title": f["title"], "claim_type": f["claim_type"], "confidence": f["confidence"],
                       "observation": f["observation"], "evidence": f["evidence_text"], "why": f["why_it_matters"],
                       "interpretations": f["interpretations"], "follow_up": f["follow_up"]})
        sec("findings", "Key findings", fb)

    if charts:
        sec("charts", "Charts", [{"type": "chart", "chart": c} for c in charts])

    if model:
        r = model["result"]
        tm, bm = r["test_metrics"], r.get("baseline_test_metrics") or {}
        keys = [k for k in tm if isinstance(tm[k], (int, float)) and k != "n"]
        cand = [c for c in r["candidates"] if c["status"] == "ok"]
        pm = r["primary_metric"]
        sec("model", f"Machine learning: predicting '{r['target']}'", [
            {"type": "paragraph", "text": r["rationale"]},
            {"type": "metrics", "items": [{"label": "Problem", "value": r["problem_type"]},
                                          {"label": "Selected model", "value": r["selected"]["model"]},
                                          {"label": "Train / test rows", "value": f"{r['rows']['train']:,} / {r['rows']['test']:,}"},
                                          {"label": "Split", "value": r["split"]["method"]}]},
            {"type": "table", "title": "Test-set metrics (model vs baseline)", "columns": ["Metric", "Model", "Baseline"],
             "rows": [[k, _num(tm.get(k), 4), _num(bm.get(k), 4)] for k in keys]},
            {"type": "table", "title": f"Model comparison ({r['cv_description']}, {r['primary_metric_label']})",
             "columns": ["Model", "CV mean", "CV std", "Train score", "Seconds"],
             "rows": [[c["model"], _num(c["cv"][pm]["mean"], 4), _num(c["cv"][pm]["std"], 3),
                       _num(c["train"].get(pm), 4), _num(c["cv_seconds"], 3)] for c in cand]},
            {"type": "table", "title": "Feature importance (permutation, test set)", "columns": ["Feature", "Importance", "Std", "Direction"],
             "rows": [[i["feature"], _num(i["importance"], 3), _num(i["std"], 2), i.get("direction") or "—"]
                      for i in r["feature_importance"]["items"][:15]]},
            {"type": "bullets", "title": "Warnings and limitations", "items": r["warnings"] + r["limitations"]},
        ])

    if forecast and forecast.get("available"):
        fc = forecast["forecasts"][forecast["selected_method"]]
        sec("forecast", f"Forecast: {forecast['aggregation']} of {forecast.get('target') or 'records'}", [
            {"type": "paragraph", "text": " ".join(forecast["explanation"])},
            {"type": "chart", "chart": {"type": "forecast", "forecast": forecast,
                                        "title": f"{forecast['selected_label']} forecast with 80% / 95% intervals"}},
            {"type": "table", "title": "Forecast values", "columns": ["Period", "Forecast", "80% low", "80% high", "95% low", "95% high"],
             "rows": [[t[:10], _num(a), _num(b), _num(c), _num(d), _num(e)] for t, a, b, c, d, e in
                      zip(forecast["future_index"], fc["forecast"], fc["lower80"], fc["upper80"], fc["lower95"], fc["upper95"])]},
            {"type": "table", "title": "Backtest comparison", "columns": ["Method", "MASE", "RMSE", "80% coverage", "Skill vs naive"],
             "rows": [[c["label"], _num(c.get("mase")), _num(c.get("rmse")),
                       f"{c['coverage80']:.0%}" if c.get("coverage80") is not None else "—",
                       f"{c['skill_vs_naive_pct']:.0f}%" if c.get("skill_vs_naive_pct") is not None else "—"]
                      for c in forecast["comparison"] if c["status"] == "ok"]},
            {"type": "bullets", "title": "Limitations", "items": forecast["warnings"] + forecast["limitations"]},
        ])

    if diagnostics and diagnostics.get("findings"):
        sec("diagnostics", f"Diagnostics: drivers of {diagnostics['outcome']}", [
            {"type": "paragraph", "text": diagnostics["causal_note"]}] + [
            {"type": "finding", "title": f["title"], "claim_type": f["label"], "confidence": None,
             "observation": f"Metric: {f['metric']}. Comparison: {f['comparison']}.", "evidence": f["evidence"],
             "why": None, "interpretations": f["alternatives"] + ([f"Potential confounders: {', '.join(f['confounders'])}"]
                                                                  if f.get("confounders") else []),
             "follow_up": f["validation"]} for f in diagnostics["findings"][:6]])

    if rules and rules.get("results"):
        sec("rules", "Data-quality rules", [
            {"type": "paragraph", "text": f"{rules['rules_passed']} of {rules['rules_total']} rule(s) passed; "
                                          f"{rules['rows_with_violations']:,} row(s) violate at least one rule."},
            {"type": "table", "columns": ["Rule", "Status", "Violations", "%"],
             "rows": [[r["rule"]["name"], r["status"], r.get("violations", "—"),
                       f"{r['violation_pct']:.2f}" if r.get("violation_pct") is not None else "—"] for r in rules["results"]]}])

    sec("recommendations", "Recommendations", [{"type": "bullets", "items": [f"{r['title']}: {r['reason']}" for r in recs[:6]]}])
    sec("limitations", "Limitations and uncertainty", [{"type": "bullets", "items": [
        "Results describe associations in the uploaded data and do not establish causation.",
        "Statistical tests assume the rows are independent observations; repeated measurements of the same entity violate this.",
        "Model and forecast accuracy were measured on held-out data from the same source and may degrade on new data.",
        "Any data issue not detected by the quality checks propagates into every result."]}])
    sec("reproducibility", "Reproducibility", [{"type": "table", "columns": ["Item", "Value"], "rows": [
        ["Workspace", workspace_id], ["Dataset", dataset.get("name")], ["Dataset version", dataset.get("version")],
        ["Source SHA-256", ingestion.get("sha256")], ["Cleaning steps", len(ops)],
        ["Generated", time.strftime("%Y-%m-%d %H:%M:%S %Z")], ["DataForge", app_version]] +
        [[k, v or "not installed"] for k, v in versions.items()]}])

    return to_jsonable({"meta": {"title": title, "generated_at": time.time(), "dataset": dataset,
                                 "workspace_id": workspace_id, "app_version": app_version, "sections": want},
                        "sections": S})


def _params(p: Dict) -> str:
    s = ", ".join(f"{k}={v}" for k, v in p.items() if k != "mapping")
    if "mapping" in p:
        s += (", " if s else "") + f"mapping of {len(p['mapping'])} value(s)"
    return s[:120]


def _col_summary(c: Dict) -> str:
    if c.get("numeric"):
        n = c["numeric"]
        return f"mean {_num(n.get('mean'))}, median {_num(n.get('median'))}, range {_num(n.get('min'))}–{_num(n.get('max'))}"
    if c.get("datetime"):
        d = c["datetime"]
        return f"{d.get('min', '')[:10]} → {d.get('max', '')[:10]}"
    if c.get("categorical") and c["categorical"].get("top_values"):
        t = c["categorical"]["top_values"][0]
        return f"top: {str(t['value'])[:30]} ({t['pct']:.0f}%)"
    return ""


# ── Markdown ────────────────────────────────────────────────────────────────────

def _md_escape(s: Any) -> str:
    return str(s).replace("|", "\\|").replace("\n", " ")


def render_markdown(rep: Dict[str, Any]) -> str:
    m = rep["meta"]
    out = [f"# {m['title']}", "", f"_Dataset: {m['dataset'].get('name')} · generated "
           f"{time.strftime('%Y-%m-%d %H:%M', time.localtime(m['generated_at']))} by DataForge AI {m['app_version']}_", ""]
    for s in rep["sections"]:
        out += [f"## {s['title']}", ""]
        for b in s["blocks"]:
            t = b["type"]
            if t == "paragraph":
                out += [f"**{b['text']}**" if b.get("style") == "lead" else b["text"], ""]
            elif t == "bullets":
                if b.get("title"):
                    out += [f"**{b['title']}**", ""]
                out += [f"- {i}" for i in b["items"]] + [""]
            elif t == "metrics":
                out += [" · ".join(f"**{i['label']}:** {_num(i['value']) if not isinstance(i['value'], str) else i['value']}"
                                   for i in b["items"]), ""]
            elif t == "table":
                if b.get("title"):
                    out += [f"**{b['title']}**", ""]
                out += ["| " + " | ".join(_md_escape(c) for c in b["columns"]) + " |",
                        "|" + "|".join("---" for _ in b["columns"]) + "|"]
                out += ["| " + " | ".join(_md_escape(_num(v) if isinstance(v, float) else v) for v in r) + " |" for r in b["rows"]]
                out += [""]
            elif t == "finding":
                out += [f"### {b['title']}", f"_{b['claim_type']}"
                        + (f", {b['confidence']} confidence" if b.get("confidence") else "") + "_", "",
                        f"- **Observed:** {b['observation']}", f"- **Evidence:** {b['evidence']}"]
                if b.get("why"):
                    out.append(f"- **Why it matters:** {b['why']}")
                if b.get("interpretations"):
                    out.append(f"- **Possible interpretations:** {'; '.join(b['interpretations'])}")
                if b.get("follow_up"):
                    out.append(f"- **Follow-up:** {'; '.join(b['follow_up'])}")
                out.append("")
            elif t == "chart":
                c = b["chart"]
                out += [f"_Chart: {c.get('title', c.get('type'))}_ (see PDF export)", ""]
    return "\n".join(out)


# ── PDF ─────────────────────────────────────────────────────────────────────────

_FONT = None


def _fonts():
    global _FONT
    if _FONT:
        return _FONT
    from reportlab.pdfbase import pdfmetrics
    from reportlab.pdfbase.ttfonts import TTFont
    try:
        import matplotlib
        base = os.path.join(os.path.dirname(matplotlib.__file__), "mpl-data", "fonts", "ttf")
        pdfmetrics.registerFont(TTFont("DFSans", os.path.join(base, "DejaVuSans.ttf")))
        pdfmetrics.registerFont(TTFont("DFSans-Bold", os.path.join(base, "DejaVuSans-Bold.ttf")))
        from reportlab.pdfbase.pdfmetrics import registerFontFamily
        registerFontFamily("DFSans", normal="DFSans", bold="DFSans-Bold", italic="DFSans", boldItalic="DFSans-Bold")
        _FONT = ("DFSans", "DFSans-Bold")
    except Exception:  # fall back to the built-in Helvetica (Latin-1 only)
        _FONT = ("Helvetica", "Helvetica-Bold")
    return _FONT


def _chart_png(chart: Dict[str, Any]) -> Optional[bytes]:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:
        return None
    ink, muted, accent, accent2 = "#1f2937", "#6b7280", "#2563eb", "#93c5fd"
    fig, ax = plt.subplots(figsize=(6.4, 2.8), dpi=150)
    try:
        t = chart.get("type")
        if t == "histogram":
            d = chart["data"]
            if chart.get("discrete"):
                ax.bar([x["label"] for x in d], [x["count"] for x in d], color=accent)
            else:
                ax.bar([x["x0"] for x in d], [x["count"] for x in d], width=[x["x1"] - x["x0"] for x in d], align="edge",
                       color=accent, edgecolor="white", linewidth=0.5)
            ax.set_xlabel(chart.get("x", ""))
            ax.set_ylabel("Count")
        elif t in ("bar", "pie"):
            d = chart["data"][:15]
            ax.barh([str(x["label"])[:24] for x in d][::-1], [x["value"] or 0 for x in d][::-1], color=accent)
            ax.set_xlabel(chart.get("y_label", ""))
        elif t == "line":
            for i, s in enumerate(chart["series"][:6]):
                xs = [p["t"][:10] for p in s["points"]]
                ys = [p["value"] if p["value"] is not None else float("nan") for p in s["points"]]
                ax.plot(range(len(xs)), ys, color=accent if i == 0 else None, linewidth=1.6, label=s["name"])
            n = len(chart["series"][0]["points"])
            step = max(1, n // 6)
            ax.set_xticks(range(0, n, step))
            ax.set_xticklabels([chart["series"][0]["points"][i]["t"][:10] for i in range(0, n, step)], fontsize=6)
            if len(chart["series"]) > 1:
                ax.legend(fontsize=6, frameon=False)
        elif t == "scatter":
            pts = chart["points"]
            ax.scatter([p["x"] for p in pts], [p["y"] for p in pts], s=6, alpha=0.5, color=accent)
            tr = chart.get("trend")
            if tr:
                ax.plot([tr["x_min"], tr["x_max"]], [tr["intercept"] + tr["slope"] * tr["x_min"],
                                                     tr["intercept"] + tr["slope"] * tr["x_max"]], color=ink, linewidth=1)
            ax.set_xlabel(chart["x"])
            ax.set_ylabel(chart["y"])
        elif t == "box":
            g = chart["groups"][:12]
            stats = [{"med": x["median"], "q1": x["q1"], "q3": x["q3"], "whislo": x["whisker_low"],
                      "whishi": x["whisker_high"], "fliers": x.get("outliers", [])[:50], "label": str(x["label"])[:14]} for x in g]
            ax.bxp(stats, showfliers=True, patch_artist=True, boxprops={"facecolor": accent2, "edgecolor": accent},
                   medianprops={"color": ink}, flierprops={"markersize": 2})
            ax.tick_params(axis="x", labelsize=6)
            ax.set_ylabel(chart["y"])
        elif t == "heatmap":
            import numpy as np
            mat = np.array([[v if v is not None else np.nan for v in r] for r in chart["matrix"]], dtype=float)
            im = ax.imshow(mat, cmap="RdBu_r", vmin=-1, vmax=1)
            ax.set_xticks(range(len(chart["columns"])))
            ax.set_xticklabels([c[:12] for c in chart["columns"]], rotation=60, fontsize=5, ha="right")
            ax.set_yticks(range(len(chart["columns"])))
            ax.set_yticklabels([c[:12] for c in chart["columns"]], fontsize=5)
            fig.colorbar(im, ax=ax, shrink=0.8)
            fig.set_size_inches(5, 4)
        elif t == "missing":
            d = chart["data"][:20]
            ax.barh([x["label"][:24] for x in d][::-1], [x["pct"] for x in d][::-1], color="#d97706")
            ax.set_xlabel("% missing")
        elif t == "forecast":
            f = chart["forecast"]
            hist = f["history"][-80:]
            fc = f["forecasts"][f["selected_method"]]
            n = len(hist)
            ax.plot(range(n), [h["value"] for h in hist], color=ink, linewidth=1.4, label="History")
            xs = list(range(n, n + len(fc["forecast"])))
            ax.fill_between(xs, fc["lower95"], fc["upper95"], color=accent2, alpha=0.35, label="95% interval", linewidth=0)
            ax.fill_between(xs, fc["lower80"], fc["upper80"], color=accent2, alpha=0.7, label="80% interval", linewidth=0)
            ax.plot(xs, fc["forecast"], color=accent, linewidth=1.8, label="Forecast")
            ax.legend(fontsize=6, frameon=False)
            labels = [h["t"][:10] for h in hist] + [t[:10] for t in f["future_index"]]
            step = max(1, len(labels) // 6)
            ax.set_xticks(range(0, len(labels), step))
            ax.set_xticklabels(labels[::step], fontsize=6)
        else:
            return None
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
        ax.tick_params(labelsize=7, colors=muted)
        ax.set_title(chart.get("title", ""), fontsize=8, color=ink, loc="left")
        fig.tight_layout()
        buf = io.BytesIO()
        fig.savefig(buf, format="png")
        return buf.getvalue()
    except Exception as e:
        log.warning("Chart rendering failed: %s", e)
        return None
    finally:
        plt.close(fig)


def render_pdf(rep: Dict[str, Any]) -> bytes:
    from reportlab.lib import colors
    from reportlab.lib.enums import TA_CENTER
    from reportlab.lib.pagesizes import A4
    from reportlab.lib.styles import ParagraphStyle
    from reportlab.lib.units import cm
    from reportlab.platypus import (HRFlowable, Image, KeepTogether, PageBreak, Paragraph, SimpleDocTemplate,
                                    Spacer, Table, TableStyle)

    reg, bold = _fonts()
    ink = colors.HexColor("#111827")
    muted = colors.HexColor("#6b7280")
    accent = colors.HexColor("#1d4ed8")
    rule = colors.HexColor("#e5e7eb")
    zebra = colors.HexColor("#f8fafc")
    st = {
        "brand": ParagraphStyle("brand", fontName=bold, fontSize=9, textColor=accent, spaceAfter=4),
        "title": ParagraphStyle("title", fontName=bold, fontSize=22, leading=27, textColor=ink, spaceAfter=6),
        "sub": ParagraphStyle("sub", fontName=reg, fontSize=10, textColor=muted, spaceAfter=14),
        "h2": ParagraphStyle("h2", fontName=bold, fontSize=14, leading=18, textColor=ink, spaceBefore=14, spaceAfter=6),
        "h3": ParagraphStyle("h3", fontName=bold, fontSize=10.5, leading=14, textColor=ink, spaceBefore=8, spaceAfter=3),
        "body": ParagraphStyle("body", fontName=reg, fontSize=9.2, leading=13.5, textColor=ink, spaceAfter=5),
        "lead": ParagraphStyle("lead", fontName=bold, fontSize=11, leading=15, textColor=ink, spaceAfter=8),
        "small": ParagraphStyle("small", fontName=reg, fontSize=7.8, leading=10.5, textColor=muted, spaceAfter=3),
        "cell": ParagraphStyle("cell", fontName=reg, fontSize=7.6, leading=9.6, textColor=ink),
        "cellb": ParagraphStyle("cellb", fontName=bold, fontSize=7.6, leading=9.6, textColor=ink),
        "foot": ParagraphStyle("foot", fontName=reg, fontSize=7, textColor=muted, alignment=TA_CENTER),
    }

    def P(text, style="body"):
        return Paragraph(escape(str(text)), st[style])

    m = rep["meta"]
    story: List[Any] = [P("DataForge AI · Analytics report", "brand"), P(m["title"], "title"),
                        P(f"Dataset: {m['dataset'].get('name')} · {m['dataset'].get('rows', '—')} rows · generated "
                          f"{time.strftime('%d %B %Y, %H:%M', time.localtime(m['generated_at']))}", "sub"),
                        HRFlowable(width="100%", thickness=1, color=accent, spaceAfter=10)]
    width = A4[0] - 4 * cm

    def table(block):
        cols = block["columns"]
        data = [[Paragraph(escape(str(c)), st["cellb"]) for c in cols]]
        for r in block["rows"][:200]:
            data.append([Paragraph(escape(_num(v) if isinstance(v, float) else str(v if v is not None else "—")), st["cell"])
                         for v in r])
        t = Table(data, repeatRows=1, colWidths=[width / len(cols)] * len(cols))
        t.setStyle(TableStyle([("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#eef2ff")),
                               ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, zebra]),
                               ("LINEBELOW", (0, 0), (-1, 0), 0.6, accent), ("GRID", (0, 0), (-1, -1), 0.25, rule),
                               ("VALIGN", (0, 0), (-1, -1), "TOP"), ("TOPPADDING", (0, 0), (-1, -1), 3),
                               ("BOTTOMPADDING", (0, 0), (-1, -1), 3)]))
        return t

    for s in rep["sections"]:
        if s["id"] == "reproducibility":
            story.append(PageBreak())
        story.append(P(s["title"], "h2"))
        for b in s["blocks"]:
            t = b["type"]
            if t == "paragraph":
                story.append(P(b["text"], "lead" if b.get("style") == "lead" else "body"))
            elif t == "bullets":
                if b.get("title"):
                    story.append(P(b["title"], "h3"))
                for i in b["items"]:
                    story.append(Paragraph("• " + escape(str(i)), st["body"]))
            elif t == "metrics":
                cells = [[Paragraph(f"<font size=7 color='#6b7280'>{escape(str(i['label']))}</font><br/>"
                                    f"<b>{escape(i['value'] if isinstance(i['value'], str) else _num(i['value']))}</b>", st["cell"])
                          for i in b["items"]]]
                tt = Table(cells, colWidths=[width / len(b["items"])] * len(b["items"]))
                tt.setStyle(TableStyle([("BOX", (0, 0), (-1, -1), 0.4, rule), ("INNERGRID", (0, 0), (-1, -1), 0.4, rule),
                                        ("TOPPADDING", (0, 0), (-1, -1), 5), ("BOTTOMPADDING", (0, 0), (-1, -1), 5)]))
                story += [tt, Spacer(1, 6)]
            elif t == "table":
                if b.get("title"):
                    story.append(P(b["title"], "h3"))
                story += [table(b), Spacer(1, 6)]
            elif t == "finding":
                parts = [P(b["title"], "h3"),
                         P(f"{b['claim_type']}" + (f" · {b['confidence']} confidence" if b.get("confidence") else ""), "small"),
                         P(f"Observed: {b['observation']}"), P(f"Evidence: {b['evidence']}")]
                if b.get("why"):
                    parts.append(P(f"Why it matters: {b['why']}"))
                if b.get("interpretations"):
                    parts.append(P("Possible interpretations: " + "; ".join(b["interpretations"])))
                if b.get("follow_up"):
                    parts.append(P("Follow-up: " + "; ".join(b["follow_up"])))
                story.append(KeepTogether(parts))
            elif t == "chart":
                png = _chart_png(b["chart"])
                if png:
                    img = Image(io.BytesIO(png))
                    ratio = img.imageHeight / img.imageWidth
                    img.drawWidth = width
                    img.drawHeight = width * ratio
                    story.append(img)
                    for n in (b["chart"].get("notes") or [])[:2]:
                        story.append(P(n, "small"))
                    story.append(Spacer(1, 6))

    def footer(canvas, doc):
        canvas.saveState()
        canvas.setFont(reg, 7)
        canvas.setFillColor(muted)
        canvas.drawCentredString(A4[0] / 2, 1.2 * cm, f"DataForge AI report · page {doc.page}")
        canvas.restoreState()

    buf = io.BytesIO()
    doc = SimpleDocTemplate(buf, pagesize=A4, leftMargin=2 * cm, rightMargin=2 * cm, topMargin=1.8 * cm,
                            bottomMargin=1.8 * cm, title=m["title"], author="DataForge AI")
    doc.build(story, onFirstPage=footer, onLaterPages=footer)
    return buf.getvalue()
