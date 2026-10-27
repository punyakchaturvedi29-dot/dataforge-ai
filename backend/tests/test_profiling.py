"""Profiling statistics are checked against independent numpy/scipy computations."""
import numpy as np
import pandas as pd
import pytest
from scipy import stats as sps

from agents.profiling import correlations, profile_dataset, quality_score, target_analysis
from agents.schema import infer_schema
from agents.stats import benjamini_hochberg, cramers_v, two_group


@pytest.fixture(scope="module")
def df():
    rng = np.random.default_rng(42)
    n = 500
    a = rng.normal(100, 15, n)
    b = a * 0.5 + rng.normal(0, 5, n)
    c = rng.exponential(10, n)
    g = rng.choice(["x", "y", "z"], n)
    d = pd.DataFrame({"a": a, "b": b, "c": c, "g": g})
    d.loc[:9, "c"] = np.nan
    d = pd.concat([d, d.iloc[:5]], ignore_index=True)  # 5 duplicates
    return d


def test_numeric_statistics_match_numpy(df):
    p = profile_dataset(df)
    col = next(c for c in p["columns"] if c["name"] == "c")
    v = df["c"].dropna().to_numpy()
    n = col["numeric"]
    assert n["mean"] == pytest.approx(v.mean())
    assert n["median"] == pytest.approx(np.median(v))
    assert n["std"] == pytest.approx(v.std(ddof=1))
    assert n["variance"] == pytest.approx(v.var(ddof=1))
    assert n["q1"] == pytest.approx(np.percentile(v, 25))
    assert n["q3"] == pytest.approx(np.percentile(v, 75))
    assert n["iqr"] == pytest.approx(np.percentile(v, 75) - np.percentile(v, 25))
    assert n["skewness"] == pytest.approx(pd.Series(v).skew())
    assert n["range"] == pytest.approx(v.max() - v.min())
    assert sum(n["histogram"]["counts"]) == len(v)
    nmiss = int(df["c"].isna().sum())
    assert col["n_missing"] == nmiss and col["missing_pct"] == pytest.approx(nmiss / len(df) * 100, abs=0.01)
    assert n["shape"].startswith("strongly right-skewed")


def test_dataset_metrics(df):
    p = profile_dataset(df)
    assert p["duplicates"]["rows"] == 5
    nmiss = int(df.isna().sum().sum())
    assert p["missing"]["cells"] == nmiss
    q = p["quality"]
    expected = 100 * (0.4 * (1 - nmiss / (len(df) * 4)) + 0.2 * (1 - 5 / len(df)) + 0.25 * 1 + 0.15 * 1)
    assert q["score"] == pytest.approx(expected, abs=0.1)
    assert set(q["components"]) == {"completeness", "uniqueness", "validity", "usefulness"}


def test_correlation_p_values_match_scipy(df):
    c = correlations(df)
    pair = next(p for p in c["pairs"] if {p["a"], p["b"]} == {"a", "b"})
    r, pv = sps.pearsonr(df["a"], df["b"])
    rs, _ = sps.spearmanr(df["a"], df["b"])
    assert pair["pearson"] == pytest.approx(r) and pair["spearman"] == pytest.approx(rs)
    assert pair["p_value"] == pytest.approx(pv, rel=1e-6)
    assert pair["ci95"][0] < r < pair["ci95"][1]
    assert "p_adjusted" in pair


def test_benjamini_hochberg_matches_reference():
    p = [0.01, 0.04, 0.03, 0.2, None]
    adj = benjamini_hochberg(p)
    # reference: statsmodels multipletests(fdr_bh)
    from statsmodels.stats.multitest import multipletests
    ref = multipletests([0.01, 0.04, 0.03, 0.2], method="fdr_bh")[1]
    assert adj[:4] == pytest.approx(list(ref)) and adj[4] is None


def test_two_group_effect_sizes():
    rng = np.random.default_rng(0)
    a, b = rng.normal(1, 1, 300), rng.normal(0, 1, 300)
    r = two_group(a, b)
    assert 0.8 < r["cohens_d"] < 1.2 and r["mann_whitney_p"] < 1e-10
    lo, hi = r["diff_ci95"]
    assert lo < a.mean() - b.mean() < hi


def test_cramers_v_independent_vs_dependent():
    rng = np.random.default_rng(1)
    x = rng.choice(["a", "b"], 2000)
    assert cramers_v(pd.Series(x), pd.Series(rng.choice(["u", "v"], 2000)))["v"] < 0.05
    assert cramers_v(pd.Series(x), pd.Series(x))["v"] > 0.95


def test_target_analysis_classification(binary_df):
    t = target_analysis(binary_df, "churned")
    assert t["task_hint"] == "classification"
    assert sum(c["count"] for c in t["class_balance"]) == len(binary_df)
    feats = [f["feature"] for f in t["features"]]
    assert "customer_id" not in feats and "usage" in feats


def test_profile_endpoint_marks_stage(client, sample_ws):
    r = client.get(f"/api/workspaces/{sample_ws}/profile")
    assert r.status_code == 200
    stages = {s["id"]: s["completed"] for s in client.get(f"/api/workspaces/{sample_ws}").json()["stages"]}
    assert stages["profile"]
    csv = client.get(f"/api/workspaces/{sample_ws}/profile/export")
    assert csv.status_code == 200 and "column,type" in csv.text.splitlines()[0].lstrip("\ufeff")
