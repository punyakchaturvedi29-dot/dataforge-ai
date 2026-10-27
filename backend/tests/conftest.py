"""
Shared fixtures. Every test session uses a fresh temporary data directory, and
datasets are generated with fixed seeds so statistical assertions are stable.
"""
from __future__ import annotations

import io
import os
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

BACKEND = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BACKEND))

_TMP = tempfile.mkdtemp(prefix="dataforge-test-")
os.environ["DATAFORGE_DATA_DIR"] = _TMP
os.environ["ANTHROPIC_API_KEY"] = ""
os.environ["TRAINING_TIME_BUDGET_S"] = "90"
os.environ["LOG_LEVEL"] = "WARNING"
os.environ["ML_N_JOBS"] = "1"

from core.config import reload_settings  # noqa: E402

reload_settings()

from fastapi.testclient import TestClient  # noqa: E402

import main  # noqa: E402

SAMPLE = BACKEND.parent / "sample_data.csv"


@pytest.fixture(scope="session")
def client():
    with TestClient(main.app) as c:
        yield c


def csv_bytes(df: pd.DataFrame, **kw) -> bytes:
    buf = io.StringIO()
    df.to_csv(buf, index=False, **kw)
    return buf.getvalue().encode("utf-8")


def upload(client, content: bytes, name: str = "data.csv", **form):
    return client.post("/api/workspaces/upload", files={"file": (name, content, "application/octet-stream")}, data=form)


def make_ws(client, df: pd.DataFrame, name: str = "data.csv") -> str:
    r = upload(client, csv_bytes(df), name)
    assert r.status_code == 201, r.text
    return r.json()["id"]


@pytest.fixture()
def sample_ws(client) -> str:
    r = client.post("/api/workspaces/sample")
    assert r.status_code == 201, r.text
    return r.json()["id"]


def wait_job(client, job_id: str, timeout: float = 240):
    import time
    t0 = time.time()
    while time.time() - t0 < timeout:
        j = client.get(f"/api/jobs/{job_id}").json()
        if j["status"] not in ("queued", "running"):
            return j
        time.sleep(0.2)
    raise AssertionError("job did not finish in time")


# ── Deterministic synthetic datasets ───────────────────────────────────────────

@pytest.fixture(scope="session")
def regression_df() -> pd.DataFrame:
    rng = np.random.default_rng(7)
    n = 400
    x1 = rng.normal(50, 10, n)
    x2 = rng.uniform(0, 1, n)
    cat = rng.choice(["north", "south", "east"], n)
    y = 3 * x1 + 40 * x2 + np.where(cat == "north", 25, 0) + rng.normal(0, 5, n)
    return pd.DataFrame({"row_id": np.arange(1, n + 1), "x1": x1, "x2": x2, "region": cat, "noise": rng.normal(0, 1, n),
                         "target": y})


@pytest.fixture(scope="session")
def binary_df() -> pd.DataFrame:
    rng = np.random.default_rng(11)
    n = 600
    usage = rng.gamma(2, 10, n)
    tenure = rng.integers(1, 60, n)
    plan = rng.choice(["basic", "pro", "team"], n, p=[0.5, 0.35, 0.15])
    logit = -3.2 + 0.06 * (30 - usage) - 0.03 * tenure + np.where(plan == "basic", 0.8, 0)
    churn = rng.uniform(0, 1, n) < 1 / (1 + np.exp(-logit))
    return pd.DataFrame({"customer_id": [f"C{i:05d}" for i in range(n)], "usage": usage, "tenure": tenure, "plan": plan,
                         "churned": np.where(churn, "yes", "no")})


@pytest.fixture(scope="session")
def monthly_df() -> pd.DataFrame:
    rng = np.random.default_rng(3)
    t = pd.date_range("2017-01-01", periods=84, freq="MS")
    y = 200 + np.arange(84) * 3 + 40 * np.sin(2 * np.pi * np.arange(84) / 12) + rng.normal(0, 6, 84)
    return pd.DataFrame({"month": t.strftime("%Y-%m-%d"), "sales": y.round(2)})
