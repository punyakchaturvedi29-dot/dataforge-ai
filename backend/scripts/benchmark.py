"""
Measure end-to-end API timings on synthetic data.

    python scripts/benchmark.py --rows 100000 500000

Uses an isolated temporary data directory and the in-process ASGI client, so
numbers include request handling, JSON serialisation and parquet I/O but not
network latency. Results are printed as a Markdown table; nothing is cached
between sizes.
"""
from __future__ import annotations

import argparse
import io
import os
import sys
import tempfile
import time
from pathlib import Path

import numpy as np
import pandas as pd

BACKEND = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BACKEND))
os.environ.setdefault("DATAFORGE_DATA_DIR", tempfile.mkdtemp(prefix="dataforge-bench-"))
os.environ["ANTHROPIC_API_KEY"] = ""
os.environ.setdefault("LOG_LEVEL", "WARNING")
os.environ.setdefault("MAX_UPLOAD_MB", "500")

from core.config import reload_settings  # noqa: E402

reload_settings()
from fastapi.testclient import TestClient  # noqa: E402

import main  # noqa: E402


def synth(n: int, seed: int = 7) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    seg = rng.choice(["Regular", "Premium", "Senior"], n, p=[0.6, 0.3, 0.1])
    region = rng.choice(["North", "South", "East", "West"], n)
    cat = rng.choice(["Electronics", "Clothing", "Home", "Beauty", "Sports", "Toys"], n)
    qty = rng.integers(1, 6, n)
    price = np.round(rng.lognormal(3.5, 0.6, n), 2)
    disc = np.round(rng.uniform(0, 0.3, n), 2)
    revenue = np.round(qty * price * (1 - disc) * np.where(seg == "Premium", 1.3, 1.0) + rng.normal(0, 5, n), 2)
    df = pd.DataFrame({
        "order_id": np.arange(1, n + 1),
        "order_date": pd.Timestamp("2022-01-01") + pd.to_timedelta(rng.integers(0, 900, n), unit="D"),
        "customer_segment": seg, "region": region, "product_category": cat,
        "quantity": qty, "unit_price": price, "discount_pct": disc,
        "shipping_days": rng.integers(1, 10, n).astype(float),
        "rating": rng.integers(1, 6, n).astype(float),
        "returned": rng.random(n) < 0.08, "revenue": revenue,
    })
    df.loc[rng.random(n) < 0.03, "shipping_days"] = np.nan
    df.loc[rng.random(n) < 0.02, "rating"] = np.nan
    return df


def timed(label, fn, out):
    t = time.perf_counter()
    r = fn()
    dt = time.perf_counter() - t
    ok = getattr(r, "status_code", 200) < 400
    out.append((label, dt, ok, "" if ok else r.text[:160]))
    return r


def run(n: int, client: TestClient):
    df = synth(n)
    buf = io.StringIO()
    df.to_csv(buf, index=False)
    raw = buf.getvalue().encode()
    res = []
    r = timed("upload + ingest + type inference", lambda: client.post("/api/workspaces/upload", files={"file": ("bench.csv", raw, "text/csv")}), res)
    wid = r.json()["id"]
    base = f"/api/workspaces/{wid}"
    timed("data page (sorted, page 1)", lambda: client.get(f"{base}/data", params={"sort": "revenue", "desc": True, "page_size": 50}), res)
    timed("profile (all columns)", lambda: client.get(f"{base}/profile"), res)
    timed("insights (ranked findings)", lambda: client.get(f"{base}/insights"), res)
    timed("auto dashboard", lambda: client.get(f"{base}/dashboard"), res)
    timed("cleaning suggestions", lambda: client.get(f"{base}/cleaning/suggestions"), res)
    timed("forecast (revenue, auto)", lambda: client.post(f"{base}/forecast", json={"date_column": "order_date", "target": "revenue", "horizon": 6}), res)

    def train():
        jr = client.post(f"{base}/ml/train", json={"target": "revenue", "quick": True})
        if jr.status_code >= 400:
            return jr
        jid = jr.json()["job_id"]
        while True:
            j = client.get(f"/api/jobs/{jid}").json()
            if j["status"] in ("succeeded", "failed", "cancelled"):
                class R:  # noqa: D401 - tiny status holder
                    status_code = 200 if j["status"] == "succeeded" else 500
                    text = str(j.get("error"))
                return R()
            time.sleep(0.25)

    timed("AutoML quick (revenue, regression)", train, res)
    return raw, res


def main_():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rows", type=int, nargs="+", default=[100_000])
    a = ap.parse_args()
    with TestClient(main.app) as client:
        for n in a.rows:
            raw, res = run(n, client)
            print(f"\n### {n:,} rows × 12 columns ({len(raw) / 1e6:.1f} MB CSV)\n")
            print("| Step | Seconds | OK |\n|---|---:|:-:|")
            for label, dt, ok, err in res:
                print(f"| {label} | {dt:.2f} | {'yes' if ok else 'NO: ' + err} |")


if __name__ == "__main__":
    main_()
