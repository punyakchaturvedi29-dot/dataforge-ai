"""Reports, security-sensitive behaviour and the v4-compatible API."""
import json
import os
import time

import numpy as np
import pandas as pd
import pytest

from core.config import settings
from core.security import check_user_regex, content_disposition, escape_formulas, sanitize_filename
from core.errors import ValidationFailed

from .conftest import csv_bytes, make_ws, upload, wait_job


# ── Reports ───────────────────────────────────────────────────────────────────

def test_pdf_markdown_json_reports(client, sample_ws):
    base = f"/api/workspaces/{sample_ws}/report"
    r = client.post(base, json={"format": "pdf"})
    assert r.status_code == 200 and r.content[:4] == b"%PDF" and len(r.content) > 20_000
    assert "attachment" in r.headers["content-disposition"]
    md = client.post(base, json={"format": "markdown"}).text
    for h in ("## Executive summary", "## Data quality assessment", "## Methodology", "## Key findings",
              "## Limitations and uncertainty", "## Reproducibility"):
        assert h in md
    js = client.post(base, json={"format": "json", "sections": ["overview", "quality"]}).json()
    assert [s["id"] for s in js["sections"]] == ["overview", "quality"]
    assert client.post(base, json={"format": "json", "sections": ["bogus"]}).status_code == 422


def test_report_reflects_real_metrics(client, sample_ws):
    js = client.post(f"/api/workspaces/{sample_ws}/report", json={"format": "json"}).json()
    ov = next(s for s in js["sections"] if s["id"] == "overview")
    metrics = {m["label"]: m["value"] for m in ov["blocks"][0]["items"]}
    assert metrics["Rows"] == 100 and metrics["Columns"] == 15


def test_report_with_markup_characters_in_names(client):
    df = pd.DataFrame({"R&D <cost>": np.arange(40.0), "segment</b>": ["a&b", "<i>c"] * 20,
                       "y": np.random.default_rng(0).normal(size=40)})
    wid = make_ws(client, df, "we<ird>&name.csv")
    r = client.post(f"/api/workspaces/{wid}/report", json={"format": "pdf"})
    assert r.status_code == 200 and r.content[:4] == b"%PDF"


def test_report_includes_model_forecast_and_diagnostics(client, sample_ws):
    r = client.post(f"/api/workspaces/{sample_ws}/ml/train", json={"target": "revenue", "models": ["linear"], "tune": False})
    assert wait_job(client, r.json()["job_id"])["status"] == "succeeded"
    client.post(f"/api/workspaces/{sample_ws}/forecast", json={"date_column": "order_date", "target": "revenue", "horizon": 3})
    client.post(f"/api/workspaces/{sample_ws}/diagnostics", json={"outcome": "revenue"})
    js = client.post(f"/api/workspaces/{sample_ws}/report", json={"format": "json"}).json()
    ids = [s["id"] for s in js["sections"]]
    assert {"model", "forecast", "diagnostics"} <= set(ids)
    pdf = client.post(f"/api/workspaces/{sample_ws}/report", json={"format": "pdf"})
    assert pdf.status_code == 200 and len(pdf.content) > 50_000  # charts rendered


# ── Security ──────────────────────────────────────────────────────────────────

@pytest.mark.parametrize("bad", ["..%2F..%2Fetc", "....", "abc", "0" * 100, "zzzzzzzzzzzzzzzz", "%2e%2e"])
def test_invalid_ids_do_not_reach_the_filesystem(client, bad):
    r = client.get(f"/api/workspaces/{bad}")
    assert r.status_code == 404


def test_traversal_in_dataset_ids(client, sample_ws):
    r = client.get(f"/api/workspaces/{sample_ws}/data", params={"dataset_id": "../../secret"})
    assert r.status_code == 404


def test_filename_sanitisation_and_headers():
    assert sanitize_filename("../../etc/passwd") == "passwd"
    assert sanitize_filename("C:\\temp\\evil.csv") == "evil.csv"
    assert sanitize_filename("a\x00b\r\nc.csv") == "abc.csv"
    h = content_disposition('evil"\r\nSet-Cookie: x=1.csv')
    assert "\r" not in h and "\n" not in h


def test_formula_escaping():
    df = pd.DataFrame({"t": ["=1+1", "+cmd", "-2", "@SUM(A1)", "safe"], "n": [-5, 1, 2, 3, 4]})
    out = escape_formulas(df)
    assert out["t"].tolist() == ["'=1+1", "'+cmd", "'-2", "'@SUM(A1)", "safe"]
    assert out["n"].tolist() == [-5, 1, 2, 3, 4]


def test_download_is_formula_safe(client):
    wid = make_ws(client, pd.DataFrame({"a": ["=cmd|' /C calc'!A0", "ok"], "b": [1, 2]}))
    r = client.get(f"/api/workspaces/{wid}/download")
    assert r.status_code == 200 and "'=cmd" in r.text
    x = client.get(f"/api/workspaces/{wid}/download", params={"format": "xlsx"})
    assert x.status_code == 200 and x.content[:2] == b"PK"


def test_regex_screening():
    for bad in ["(a+)+$", "(x*)*", "(\\w+)+@", "a" * 201, "(a)\\1"]:
        with pytest.raises(ValidationFailed):
            check_user_regex(bad)
    assert check_user_regex(r"^[A-Z]{3}\d{2}$")


def test_large_json_body_rejected(client, sample_ws):
    big = {"message": "x" * (settings.max_json_body_kb * 1024 + 10)}
    r = client.post(f"/api/workspaces/{sample_ws}/copilot", content=json.dumps(big),
                    headers={"content-type": "application/json"})
    assert r.status_code == 413


def test_errors_do_not_leak_internals(client, sample_ws, monkeypatch):
    from services import analysis
    monkeypatch.setattr(analysis, "profile", lambda *a, **k: (_ for _ in ()).throw(RuntimeError("secret path /etc/x")))
    from fastapi.testclient import TestClient
    import main
    with TestClient(main.app, raise_server_exceptions=False) as c:
        r = c.get(f"/api/workspaces/{sample_ws}/overview")
    assert r.status_code == 500
    body = r.json()
    assert body["error"]["code"] == "internal_error" and "secret" not in r.text and body["error"]["request_id"]


def test_validation_errors_are_structured(client, sample_ws):
    r = client.post(f"/api/workspaces/{sample_ws}/ml/train", json={"target": "revenue", "test_size": 5})
    assert r.status_code == 422 and r.json()["error"]["code"] == "invalid_request"


def test_cors_allows_only_configured_origins(client):
    ok = client.get("/api/health", headers={"Origin": "http://localhost:5173"})
    assert ok.headers.get("access-control-allow-origin") == "http://localhost:5173"
    bad = client.get("/api/health", headers={"Origin": "http://evil.example"})
    assert "access-control-allow-origin" not in bad.headers


def test_security_headers(client):
    r = client.get("/api/health")
    assert r.headers["x-content-type-options"] == "nosniff" and r.headers["cache-control"] == "no-store"


def test_delete_workspace_removes_files(client):
    wid = make_ws(client, pd.DataFrame({"a": [1, 2, 3]}))
    path = settings.data_dir / "workspaces" / wid
    assert path.exists()
    assert client.delete(f"/api/workspaces/{wid}").status_code == 204
    assert not path.exists() and client.get(f"/api/workspaces/{wid}").status_code == 404


def test_retention_purge(client):
    from services.workspace import store
    wid = make_ws(client, pd.DataFrame({"a": [1, 2, 3]}))
    ws = store.get(wid)
    ws.meta["updated_at"] = time.time() - (settings.workspace_ttl_days + 1) * 86400
    from services.workspace import _atomic_write_json
    _atomic_write_json(ws.dir / "workspace.json", ws.meta)
    store.clear_cache()
    assert store.maybe_purge(force=True) >= 1
    assert client.get(f"/api/workspaces/{wid}").status_code == 404


def test_workspace_listing_can_be_disabled(client, monkeypatch):
    assert client.get("/api/workspaces").status_code == 200
    monkeypatch.setattr(settings, "workspace_listing", False)
    assert client.get("/api/workspaces").status_code == 403


def test_secret_key_file_permissions():
    p = settings.data_dir / ".secret_key"
    assert p.exists() and oct(p.stat().st_mode)[-3:] == "600"


def test_health_and_config_do_not_expose_secrets(client, monkeypatch):
    monkeypatch.setattr(settings, "anthropic_api_key", "sk-ant-super-secret")
    for url in ("/api/health", "/api/config"):
        assert "super-secret" not in client.get(url).text


# ── v4 compatibility ──────────────────────────────────────────────────────────

def test_v4_endpoints(client):
    from pathlib import Path
    content = (Path(__file__).parents[2] / "sample_data.csv").read_bytes()
    r = client.post("/api/upload/", files={"file": ("sample_data.csv", content)})
    assert r.status_code == 200
    up = r.json()
    sid = up["session_id"]
    assert up["rows"] == 100 and up["cleaning_report"] == [] and "health_score" in up and len(up["preview"]) == 10
    a = client.get(f"/api/analyze/{sid}").json()
    assert "order_id" not in a["eda"]["numeric_cols"] and a["insights"]
    assert "revenue" in a["eda"]["time_series"] or a["eda"]["time_series"]
    chat = client.post("/api/chat/", json={"session_id": sid, "message": "which columns have missing values?"})
    lines = [json.loads(x) for x in chat.text.strip().splitlines()]
    assert "text" in lines[0] and lines[1]["reply"]["tool"] == "missing_values"
    assert client.get(f"/api/report/{sid}").content[:4] == b"%PDF"
    p = client.post("/api/predict/", json={"session_id": sid, "task": "regression", "target_col": "revenue"}).json()
    assert p["metrics"]["r2"] is not None and "forecast" not in p
    p = client.post("/api/predict/", json={"session_id": sid, "task": "cluster", "n_clusters": 3}).json()
    assert p["n_clusters"] == 3
    assert client.get(f"/api/features/executive-summary/{sid}").json()["headline"]
    assert client.get(f"/api/features/dashboard/{sid}").json()["kpis"]
    fc = client.get(f"/api/features/forecast/{sid}?horizon=3").json()
    assert fc["available"] and "order_id" not in fc["columns"]
    assert client.post("/api/features/rca", json={"session_id": sid}).json()["target"] == "revenue"
    adv = client.post("/api/advanced/predict", json={"session_id": sid, "task": "anomaly"}).json()
    assert adv["n_anomalies"] > 0
    assert client.get(f"/api/advanced/profile/{sid}").json()["shape"]["rows"] == 100
    assert client.get(f"/api/advanced/correlations/{sid}").json()["pairs"]
    # v4 dashed session ids still resolve
    dashed = f"{sid[:8]}-{sid[8:12]}-{sid[12:16]}-{sid[16:20]}-{sid[20:]}"
    assert client.get(f"/api/analyze/{dashed}").status_code == 200


def test_v4_multifile(client):
    r = client.post("/api/features/multifile/upload", files={"file": ("o.csv", csv_bytes(pd.DataFrame({"k": [1, 2, 3], "v": [1, 2, 3]})))})
    sid = r.json()["session_id"]
    r = client.post("/api/features/multifile/upload", data={"session_id": sid},
                    files={"file": ("c.csv", csv_bytes(pd.DataFrame({"k": [1, 2, 3], "w": ["a", "b", "c"]})))})
    assert r.json()["total_files"] == 2
    a = client.get(f"/api/features/multifile/{sid}").json()
    assert a["available"] and a["join_options"]
    files = a["files"]
    m = client.post("/api/features/multifile/merge", json={"session_id": sid, "fid_a": files[0]["file_id"],
                                                          "fid_b": files[1]["file_id"], "col_a": "k"}).json()
    assert m["rows"] == 3 and "w" in m["columns"]
