"""
Unsupervised analysis (clustering, anomaly detection) and v4-compatible
wrappers around the AutoML engine.

Fixes over v4: identifiers and dates are excluded from features, missing values
are imputed with medians (not zeros), features are standardised, the 2-D view
uses a PCA projection instead of two arbitrary columns, and every anomaly comes
with the features that make it unusual.
"""
from __future__ import annotations

import warnings
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
from sklearn.cluster import DBSCAN, KMeans
from sklearn.decomposition import PCA
from sklearn.ensemble import IsolationForest
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

from core.config import settings
from core.errors import ValidationFailed
from core.jsonsafe import records, to_jsonable

from .schema import analysis_columns, infer_schema

MAX_UNSUPERVISED_ROWS = 100_000


def _feature_matrix(df: pd.DataFrame, columns: Optional[List[str]], schema=None, include_categorical: bool = True):
    schema = schema or infer_schema(df)
    cols = analysis_columns(df, schema)
    if columns:
        bad = [c for c in columns if c not in df.columns]
        if bad:
            raise ValidationFailed(f"Unknown column(s): {', '.join(bad)}")
        num = [c for c in columns if c in cols["numeric"] or (pd.api.types.is_numeric_dtype(df[c]) and not pd.api.types.is_bool_dtype(df[c]))]
        cat = [c for c in columns if c not in num and c in cols["all_categorical"]]
    else:
        num = list(cols["numeric"])
        cat = [c for c in cols["categorical"] if df[c].nunique() <= 12] if include_categorical else []
    if not num and not cat:
        raise ValidationFailed("No numeric or low-cardinality categorical columns are available.")
    parts = []
    if num:
        X = df[num].astype(float)
        X = X.fillna(X.median())
        parts.append(X)
    if cat:
        parts.append(pd.get_dummies(df[cat].astype("string").fillna("(missing)"), prefix=cat, dtype=float))
    X = pd.concat(parts, axis=1)
    X = X.loc[:, X.std(ddof=0) > 0]
    if X.shape[1] < 1:
        raise ValidationFailed("All selected features are constant.")
    return X, num, cat


def run_clustering(df: pd.DataFrame, n_clusters: int = 3, method: str = "kmeans", columns: Optional[List[str]] = None,
                   eps: Optional[float] = None, min_samples: Optional[int] = None, random_state: int = 42) -> Dict[str, Any]:
    if method not in ("kmeans", "dbscan"):
        raise ValidationFailed("method must be kmeans or dbscan")
    data = df
    note = []
    if len(df) > MAX_UNSUPERVISED_ROWS:
        data = df.sample(MAX_UNSUPERVISED_ROWS, random_state=random_state)
        note.append(f"Clustering used a random sample of {MAX_UNSUPERVISED_ROWS:,} rows.")
    if len(data) < 10:
        raise ValidationFailed("At least 10 rows are needed for clustering.")
    X, num, cat = _feature_matrix(data, columns)
    Xs = StandardScaler().fit_transform(X)
    elbow = []
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        if method == "kmeans":
            k = int(max(2, min(n_clusters, 12, len(data) - 1)))
            sil_sample = min(len(Xs), 5000)
            for ki in range(2, min(11, len(data))):
                km = KMeans(n_clusters=ki, random_state=random_state, n_init=5).fit(Xs)
                sil = silhouette_score(Xs, km.labels_, sample_size=sil_sample, random_state=random_state) if ki < len(Xs) else None
                elbow.append({"k": ki, "inertia": float(km.inertia_), "silhouette": to_jsonable(sil)})
            model = KMeans(n_clusters=k, random_state=random_state, n_init=10).fit(Xs)
            labels = model.labels_
        else:
            ms = int(min_samples or max(5, len(Xs) // 50))
            if eps is None:
                from sklearn.neighbors import NearestNeighbors
                d, _ = NearestNeighbors(n_neighbors=ms).fit(Xs).kneighbors(Xs)
                eps = float(np.quantile(d[:, -1], 0.9))
                note.append(f"eps chosen automatically as the 90th percentile of the {ms}-nearest-neighbour distance ({eps:.2f}).")
            labels = DBSCAN(eps=eps, min_samples=ms).fit_predict(Xs)
    uniq = sorted(set(labels))
    n_found = len([u for u in uniq if u != -1])
    sil = None
    valid = labels != -1
    if n_found >= 2 and valid.sum() > n_found:
        sil = float(silhouette_score(Xs[valid], labels[valid], sample_size=min(int(valid.sum()), 5000), random_state=random_state))
    prof_cols = num[:8]
    overall = data[prof_cols].astype(float).mean() if prof_cols else pd.Series(dtype=float)
    overall_sd = data[prof_cols].astype(float).std() if prof_cols else pd.Series(dtype=float)
    summaries = []
    for ci in uniq:
        mask = labels == ci
        sub = data.loc[mask]
        row: Dict[str, Any] = {"cluster": int(ci), "label": "Noise" if ci == -1 else f"Cluster {ci + 1}",
                               "size": int(mask.sum()), "pct": float(mask.mean() * 100), "means": {}, "distinctive": []}
        for c in prof_cols:
            m = float(sub[c].astype(float).mean())
            row["means"][c] = to_jsonable(m)
            if overall_sd.get(c, 0) > 0:
                z = (m - overall[c]) / overall_sd[c]
                if abs(z) >= 0.5:
                    row["distinctive"].append({"feature": c, "z": float(z), "direction": "higher" if z > 0 else "lower"})
        for c in cat[:4]:
            vc = sub[c].astype("string").value_counts(normalize=True)
            if len(vc):
                row.setdefault("top_categories", {})[c] = {"value": str(vc.index[0]), "share": float(vc.iloc[0])}
        row["distinctive"].sort(key=lambda d: -abs(d["z"]))
        summaries.append(row)
    pca = PCA(n_components=2, random_state=random_state).fit(Xs) if Xs.shape[1] >= 2 else None
    proj = pca.transform(Xs) if pca is not None else np.column_stack([Xs[:, 0], np.zeros(len(Xs))])
    n = min(len(proj), settings.chart_max_points)
    idx = np.random.default_rng(random_state).choice(len(proj), n, replace=False) if len(proj) > n else np.arange(len(proj))
    scatter = {"x_label": "Principal component 1", "y_label": "Principal component 2",
               "explained_variance": [float(v) for v in pca.explained_variance_ratio_] if pca is not None else None,
               "points": [{"x": float(proj[i, 0]), "y": float(proj[i, 1]), "cluster": int(labels[i])} for i in idx],
               "sampled": bool(len(proj) > n)}
    quality = ("well separated" if sil is not None and sil >= 0.5 else "moderately separated" if sil is not None and sil >= 0.25
               else "weakly separated" if sil is not None else "n/a")
    return to_jsonable({
        "task": "cluster", "method": method, "n_clusters": n_found, "features_used": list(X.columns),
        "numeric_features": num, "categorical_features": cat, "cluster_summaries": summaries,
        "silhouette_score": sil, "separation": quality, "elbow": elbow, "scatter": scatter,
        "inertia": float(model.inertia_) if method == "kmeans" else None,
        "noise_points": int((labels == -1).sum()),
        "notes": note + ["Features were standardised; categorical columns were one-hot encoded.",
                         "Clusters are descriptive groupings found by the algorithm, not natural categories. "
                         "Check that they are stable and meaningful before acting on them."],
    })


def run_anomaly_detection(df: pd.DataFrame, contamination: float = 0.05, columns: Optional[List[str]] = None,
                          random_state: int = 42) -> Dict[str, Any]:
    if not (0.001 <= float(contamination) <= 0.5):
        raise ValidationFailed("contamination must be between 0.001 and 0.5")
    data = df
    if len(df) > MAX_UNSUPERVISED_ROWS:
        data = df.sample(MAX_UNSUPERVISED_ROWS, random_state=random_state)
    if len(data) < 20:
        raise ValidationFailed("At least 20 rows are needed for anomaly detection.")
    X, num, cat = _feature_matrix(data, columns)
    scaler = StandardScaler().fit(X)
    Xs = scaler.transform(X)
    model = IsolationForest(contamination=contamination, random_state=random_state, n_estimators=200,
                            n_jobs=max(1, settings.ml_n_jobs)).fit(Xs)
    flags = model.predict(Xs) == -1
    score = -model.score_samples(Xs)
    # Robust per-feature deviation explains *why* a row is unusual
    med = np.median(Xs, axis=0)
    mad = np.median(np.abs(Xs - med), axis=0) * 1.4826
    mad[mad == 0] = 1
    rz = (Xs - med) / mad
    order = np.argsort(-score)
    top = order[: min(50, int(flags.sum()) or 10)]
    rows = []
    feats = list(X.columns)
    show_cols = [c for c in data.columns][:12]
    for i in top:
        contrib = sorted(((feats[j], float(rz[i, j])) for j in range(len(feats))), key=lambda t: -abs(t[1]))[:3]
        rows.append({"row": to_jsonable(data.index[i]), "score": float(score[i]), "flagged": bool(flags[i]),
                     "values": {c: to_jsonable(data.iloc[i][c]) for c in show_cols},
                     "reasons": [{"feature": f, "robust_z": z, "direction": "high" if z > 0 else "low"} for f, z in contrib]})
    counts, edges = np.histogram(score, bins=30)
    thr = float(np.min(score[flags])) if flags.any() else None
    signals = []
    for j, f in enumerate(feats):
        if flags.any():
            signals.append({"column": f, "mean_abs_z_anomalies": float(np.mean(np.abs(rz[flags, j]))),
                            "mean_abs_z_normal": float(np.mean(np.abs(rz[~flags, j])))})
    signals.sort(key=lambda d: -(d["mean_abs_z_anomalies"] - d["mean_abs_z_normal"]))
    return to_jsonable({
        "task": "anomaly", "n_anomalies": int(flags.sum()), "anomaly_rate_pct": float(flags.mean() * 100),
        "contamination": contamination, "rows_scored": int(len(data)), "features_used": feats,
        "top_anomalies": rows, "score_distribution": {"edges": edges.tolist(), "counts": counts.tolist(), "threshold": thr},
        "column_signals": signals[:8], "flagged_index": [to_jsonable(i) for i in data.index[flags][:5000]],
        "notes": ["Isolation Forest flags rows that are easy to isolate from the rest; 'contamination' sets the share "
                  "of rows flagged and is an assumption, not an estimate.",
                  "An anomaly is unusual, not necessarily wrong. Review flagged rows before removing them.",
                  "Reasons show the features with the largest robust z-scores for each row."],
    })


# ── v4 compatibility wrappers ───────────────────────────────────────────────────

def _legacy(df: pd.DataFrame, target: str, problem: str) -> Dict[str, Any]:
    from .automl import TrainConfig, train
    out = train(df, TrainConfig(target=target, problem_type=problem, quick=True, tune=False))["result"]
    pm = out["primary_metric"]
    return {
        "task": "regression" if out["problem_type"] == "regression" else "classification",
        "target": target, "best_model": out["selected"]["model"], "metrics": out["test_metrics"],
        "all_models": [{"model": c["model"], **({pm: c["cv"][pm]["mean"]} if c["status"] == "ok" else {})}
                       for c in out["candidates"]],
        "classes": out["classes"],
        "feature_importances": [{"feature": i["feature"], "importance": i["share"], "direction": i["direction"]}
                                for i in out["feature_importance"]["items"]],
        "confusion_matrix": out["diagnostics"].get("confusion_matrix", {}).get("matrix"),
        "per_class_metrics": out["diagnostics"].get("per_class"),
        "actual_sample": [p["actual"] for p in out["diagnostics"].get("actual_vs_predicted", [])][:40],
        "predicted_sample": [p["predicted"] for p in out["diagnostics"].get("actual_vs_predicted", [])][:40],
        "n_train": out["rows"]["train"], "n_test": out["rows"]["test"],
        "notes": ["Metrics are computed on a held-out test set; the model was selected by cross-validation.",
                  "The v4 'forecast' fields were removed: row-order extrapolation is not a valid forecast. "
                  "Use the Forecasting workspace instead."],
    }


def run_automl_regression(df: pd.DataFrame, target: str, time_budget: int = 20) -> Dict:
    return _legacy(df, target, "regression")


def run_automl_classification(df: pd.DataFrame, target: str) -> Dict:
    return _legacy(df, target, "auto")
