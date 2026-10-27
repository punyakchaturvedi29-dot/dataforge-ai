"""
AutoML engine: reproducible, leakage-aware supervised learning.

Pipeline
--------
1. Target validation and problem-type detection (binary / multiclass / regression).
2. Feature screening: identifiers, constants, free text and the time column are
   excluded with reasons; datetimes are expanded to calendar features.
3. Leakage checks: exact target re-codings are removed; near-perfect single
   feature predictors and train/test duplicates are flagged.
4. Hold-out split (stratified, or chronological when a time column is given).
5. Every candidate is a single sklearn ``Pipeline`` (imputation, encoding,
   scaling, model), so preprocessing is fitted inside each CV fold only.
6. Candidates are compared with K-fold (or time-series) cross-validation on the
   training split. The best few are tuned with a randomized search within the
   time budget. Selection uses the mean CV score of the primary metric.
7. The selected pipeline is refitted on the full training split and evaluated
   **once** on the untouched test split, next to a baseline.
8. Explanations: permutation importance on the test split (model-agnostic) and
   occlusion-based local explanations. Neither implies causality.
"""
from __future__ import annotations

import math
import platform
import time
import warnings
from dataclasses import asdict, dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
from sklearn.base import clone
import pandas as pd
import sklearn
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier, DummyRegressor
from sklearn.ensemble import (HistGradientBoostingClassifier, HistGradientBoostingRegressor,
                              RandomForestClassifier, RandomForestRegressor)
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import (accuracy_score, average_precision_score, balanced_accuracy_score, brier_score_loss,
                             confusion_matrix, f1_score, log_loss, matthews_corrcoef, mean_absolute_error,
                             mean_absolute_percentage_error, mean_squared_error, median_absolute_error,
                             precision_recall_curve, precision_recall_fscore_support, precision_score, r2_score,
                             recall_score, roc_auc_score, roc_curve)
from sklearn.model_selection import (KFold, RandomizedSearchCV, StratifiedKFold, TimeSeriesSplit, cross_validate,
                                     train_test_split)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor

from core.config import settings
from core.errors import ValidationFailed
from core.jsonsafe import to_jsonable

from .ml_transformers import BoolToFloat, DateFeatures, ToString
from .schema import infer_schema

try:  # optional accelerators
    import xgboost as xgb  # type: ignore
except Exception:  # pragma: no cover
    xgb = None
try:
    import lightgbm as lgb  # type: ignore
except Exception:  # pragma: no cover
    lgb = None

PROBLEM_TYPES = ("binary", "multiclass", "regression")
PRIMARY = {"regression": "neg_root_mean_squared_error", "binary": "roc_auc", "multiclass": "f1_macro"}
METRIC_LABELS = {
    "neg_root_mean_squared_error": "RMSE", "neg_mean_absolute_error": "MAE", "r2": "R²",
    "roc_auc": "ROC-AUC", "roc_auc_ovr": "ROC-AUC (one-vs-rest)", "f1_macro": "F1 (macro)", "accuracy": "Accuracy",
    "balanced_accuracy": "Balanced accuracy", "average_precision": "PR-AUC", "f1": "F1", "f1_weighted": "F1 (weighted)",
}
ALLOWED_PRIMARY = {
    "regression": ["neg_root_mean_squared_error", "neg_mean_absolute_error", "r2"],
    "binary": ["roc_auc", "average_precision", "f1", "balanced_accuracy", "accuracy"],
    "multiclass": ["f1_macro", "balanced_accuracy", "accuracy", "roc_auc_ovr", "f1_weighted"],
}


class Cancelled(Exception):
    pass


@dataclass
class TrainConfig:
    target: str
    problem_type: str = "auto"
    features: Optional[List[str]] = None
    exclude: List[str] = field(default_factory=list)
    test_size: float = 0.2
    cv_folds: int = 5
    time_column: Optional[str] = None
    models: Optional[List[str]] = None
    tune: bool = True
    n_iter: int = 8
    time_budget_s: Optional[int] = None
    primary_metric: Optional[str] = None
    class_weight: str = "auto"  # auto | balanced | none
    random_state: int = 42
    max_rows: Optional[int] = None
    quick: bool = False

    def validate(self) -> None:
        if not (0.05 <= float(self.test_size) <= 0.5):
            raise ValidationFailed("test_size must be between 0.05 and 0.5")
        if not (2 <= int(self.cv_folds) <= 10):
            raise ValidationFailed("cv_folds must be between 2 and 10")
        if self.problem_type not in ("auto",) + PROBLEM_TYPES:
            raise ValidationFailed("problem_type must be auto, binary, multiclass or regression")
        if self.class_weight not in ("auto", "balanced", "none"):
            raise ValidationFailed("class_weight must be auto, balanced or none")
        if not (1 <= int(self.n_iter) <= 50):
            raise ValidationFailed("n_iter must be between 1 and 50")


def library_versions() -> Dict[str, Optional[str]]:
    import scipy
    return {"python": platform.python_version(), "scikit-learn": sklearn.__version__, "pandas": pd.__version__,
            "numpy": np.__version__, "scipy": scipy.__version__,
            "xgboost": getattr(xgb, "__version__", None), "lightgbm": getattr(lgb, "__version__", None)}


def available_models() -> Dict[str, bool]:
    return {"baseline": True, "linear": True, "decision_tree": True, "random_forest": True,
            "hist_gradient_boosting": True, "xgboost": xgb is not None, "lightgbm": lgb is not None}


# ─────────────────────────────────────────────────────────────────────────────
# Problem type & features
# ─────────────────────────────────────────────────────────────────────────────

def detect_problem_type(y: pd.Series, logical: str, requested: str) -> Tuple[str, List[str]]:
    notes: List[str] = []
    nunique = int(y.nunique())
    numeric = pd.api.types.is_numeric_dtype(y) and not pd.api.types.is_bool_dtype(y)
    if requested != "auto":
        if requested == "regression" and not numeric:
            raise ValidationFailed("Regression needs a numeric target. Choose classification instead.")
        if requested == "binary" and nunique != 2:
            raise ValidationFailed(f"Binary classification needs exactly 2 classes; the target has {nunique}.")
        if requested == "multiclass" and nunique < 3:
            raise ValidationFailed("Multiclass classification needs at least 3 classes.")
        pt = requested
    elif nunique == 2:
        pt = "binary"
    elif not numeric or logical in ("categorical", "boolean"):
        pt = "multiclass"
    elif pd.api.types.is_integer_dtype(y) or np.allclose(y.dropna() % 1, 0):
        if nunique <= 10:
            pt = "multiclass"
            notes.append(f"The numeric target has only {nunique} distinct integer values, so it is treated as "
                         "classes. Choose regression explicitly if the order/magnitude matters.")
        else:
            pt = "regression"
    else:
        pt = "regression"
    if pt != "regression" and nunique > 50:
        raise ValidationFailed(f"The target has {nunique} classes; classification is limited to 50. "
                               "Group rare classes first or use regression for numeric targets.")
    if nunique < 2:
        raise ValidationFailed("The target has a single value; there is nothing to predict.")
    return pt, notes


def select_features(df: pd.DataFrame, cfg: TrainConfig, schema: List[Dict]) -> Tuple[Dict[str, List[str]], List[Dict]]:
    smap = {c["name"]: c for c in schema}
    report: List[Dict[str, Any]] = []
    explicit = cfg.features is not None
    candidates = list(cfg.features) if explicit else [c for c in df.columns if c != cfg.target]
    for c in candidates:
        if c not in df.columns:
            raise ValidationFailed(f"Unknown feature column '{c}'")
    kinds: Dict[str, List[str]] = {"numeric": [], "categorical": [], "boolean": [], "datetime": []}
    for c in candidates:
        if c == cfg.target:
            continue
        lt = smap[c]["logical_type"]
        reason = None
        if c in cfg.exclude:
            reason = "excluded by user"
        elif cfg.time_column and c == cfg.time_column:
            reason = "time column used for chronological splitting (expanded features would leak ordering)"
        elif lt in ("empty", "constant"):
            reason = f"{lt} column carries no information"
        elif lt == "identifier" and not explicit:
            reason = "identifier (unique per row) — would let the model memorise rows"
        elif lt == "text" and not explicit:
            reason = "free-text column (not supported without NLP features)"
        if reason:
            report.append({"column": c, "status": "excluded", "reason": reason})
            continue
        s = df[c]
        if lt == "datetime" or pd.api.types.is_datetime64_any_dtype(s):
            kinds["datetime"].append(c)
            kind = "datetime"
        elif pd.api.types.is_bool_dtype(s) or lt == "boolean":
            if pd.api.types.is_numeric_dtype(s) and not pd.api.types.is_bool_dtype(s):
                kinds["numeric"].append(c)
                kind = "numeric"
            else:
                kinds["boolean"].append(c)
                kind = "boolean"
        elif pd.api.types.is_numeric_dtype(s):
            kinds["numeric"].append(c)
            kind = "numeric"
        else:
            kinds["categorical"].append(c)
            kind = "categorical"
        note = None
        if lt == "identifier":
            note = "identifier included explicitly by the user"
        elif kind == "categorical" and smap[c]["n_unique"] > 20:
            note = f"{smap[c]['n_unique']} levels: the 20 most frequent are one-hot encoded, the rest grouped"
        report.append({"column": c, "status": "used", "kind": kind, "note": note})
    if not any(kinds.values()):
        raise ValidationFailed("No usable feature columns remain after screening. Include features explicitly.")
    return kinds, report


def leakage_checks(X: pd.DataFrame, y: pd.Series, problem: str, kinds: Dict[str, List[str]]) -> Tuple[List[str], List[Dict]]:
    """Return (columns to drop, suspects)."""
    drop: List[str] = []
    suspects: List[Dict[str, Any]] = []
    n = len(X)
    for c in kinds["numeric"]:
        s = X[c]
        d = pd.concat([s, y], axis=1).dropna()
        if len(d) < 10:
            continue
        if problem == "regression":
            r = d.iloc[:, 0].corr(d.iloc[:, 1], method="spearman")
            if pd.notna(r) and abs(r) >= 0.995:
                same = np.allclose(d.iloc[:, 0].to_numpy(dtype=float), d.iloc[:, 1].to_numpy(dtype=float))
                if same:
                    drop.append(c)
                    suspects.append({"column": c, "reason": "identical to the target", "action": "removed"})
                else:
                    suspects.append({"column": c, "reason": f"almost perfectly correlated with the target (ρ={r:.3f})",
                                     "action": "kept — verify it is known at prediction time"})
        else:
            # Perfect separation by thresholds on a single numeric feature
            g = d.groupby(d.columns[1])[d.columns[0]].agg(["min", "max"]).sort_values("min")
            if len(g) >= 2 and (g["max"].shift(1) < g["min"]).iloc[1:].all():
                suspects.append({"column": c, "reason": "separates the classes perfectly on its own",
                                 "action": "kept — verify it is known at prediction time"})
    for c in kinds["categorical"] + kinds["boolean"]:
        d = pd.concat([X[c].astype("string"), y.astype("string")], axis=1).dropna()
        if len(d) < 10:
            continue
        if problem != "regression":
            purity = d.groupby(d.columns[0])[d.columns[1]].nunique()
            n_levels = len(purity)
            if (purity == 1).all() and n_levels <= max(50, n // 2):
                inv = d.groupby(d.columns[1])[d.columns[0]].nunique()
                if (inv == 1).all():
                    drop.append(c)
                    suspects.append({"column": c, "reason": "a one-to-one relabelling of the target", "action": "removed"})
                else:
                    suspects.append({"column": c, "reason": "every value maps to exactly one target class",
                                     "action": "kept — verify it is known at prediction time"})
    return drop, suspects


def build_preprocessor(kinds: Dict[str, List[str]]) -> ColumnTransformer:
    transformers = []
    if kinds["numeric"]:
        transformers.append(("num", Pipeline([("impute", SimpleImputer(strategy="median", add_indicator=True)),
                                              ("scale", StandardScaler())]), kinds["numeric"]))
    if kinds["boolean"]:
        transformers.append(("bool", Pipeline([("cast", BoolToFloat()),
                                               ("impute", SimpleImputer(strategy="most_frequent"))]), kinds["boolean"]))
    if kinds["categorical"]:
        transformers.append(("cat", Pipeline([
            ("str", ToString()),
            ("impute", SimpleImputer(strategy="constant", fill_value="__missing__")),
            ("onehot", OneHotEncoder(handle_unknown="infrequent_if_exist", max_categories=20, min_frequency=3,
                                     sparse_output=False)),
        ]), kinds["categorical"]))
    if kinds["datetime"]:
        transformers.append(("date", Pipeline([("expand", DateFeatures()),
                                               ("impute", SimpleImputer(strategy="median")),
                                               ("scale", StandardScaler())]), kinds["datetime"]))
    return ColumnTransformer(transformers, remainder="drop", verbose_feature_names_out=True)


def _candidates(problem: str, balanced: bool, n_rows: int, rs: int, quick: bool, allowed: Optional[List[str]],
                n_classes: int) -> List[Tuple[str, str, Any, Dict[str, List[Any]]]]:
    """(key, display name, estimator, search space). Ordered roughly by cost."""
    nj = max(1, settings.ml_n_jobs)
    cw = "balanced" if balanced else None
    out: List[Tuple[str, str, Any, Dict]] = []
    if problem == "regression":
        out.append(("baseline", "Baseline (mean)", DummyRegressor(strategy="mean"), {}))
        out.append(("linear", "Ridge regression", Ridge(alpha=1.0, random_state=rs),
                    {"model__alpha": [0.01, 0.1, 1.0, 10.0, 100.0]}))
        out.append(("decision_tree", "Decision tree", DecisionTreeRegressor(max_depth=6, min_samples_leaf=5, random_state=rs),
                    {"model__max_depth": [3, 4, 6, 8, 12, None], "model__min_samples_leaf": [1, 3, 5, 10, 20]}))
        out.append(("random_forest", "Random forest",
                    RandomForestRegressor(n_estimators=200, min_samples_leaf=2, random_state=rs, n_jobs=nj),
                    {"model__max_depth": [None, 6, 10, 16], "model__min_samples_leaf": [1, 2, 5, 10],
                     "model__max_features": [1.0, 0.6, "sqrt"]}))
        out.append(("hist_gradient_boosting", "Histogram gradient boosting",
                    HistGradientBoostingRegressor(random_state=rs, max_iter=300, early_stopping="auto"),
                    {"model__learning_rate": [0.03, 0.06, 0.1, 0.2], "model__max_leaf_nodes": [15, 31, 63],
                     "model__min_samples_leaf": [5, 20, 40], "model__l2_regularization": [0.0, 0.1, 1.0]}))
        if xgb is not None:
            out.append(("xgboost", "XGBoost", xgb.XGBRegressor(n_estimators=300, learning_rate=0.06, max_depth=5,
                                                               subsample=0.9, colsample_bytree=0.9, random_state=rs,
                                                               n_jobs=nj, tree_method="hist", verbosity=0),
                        {"model__max_depth": [3, 4, 5, 6, 8], "model__learning_rate": [0.03, 0.06, 0.1, 0.2],
                         "model__subsample": [0.7, 0.85, 1.0], "model__n_estimators": [150, 300, 500]}))
        if lgb is not None:
            out.append(("lightgbm", "LightGBM", lgb.LGBMRegressor(n_estimators=300, learning_rate=0.06, random_state=rs,
                                                                  n_jobs=nj, verbose=-1),
                        {"model__num_leaves": [15, 31, 63], "model__learning_rate": [0.03, 0.06, 0.1],
                         "model__min_child_samples": [5, 20, 40], "model__n_estimators": [150, 300, 500]}))
    else:
        out.append(("baseline", "Baseline (class prior)", DummyClassifier(strategy="prior"), {}))
        out.append(("linear", "Logistic regression",
                    LogisticRegression(max_iter=2000, C=1.0, class_weight=cw, random_state=rs),
                    {"model__C": [0.01, 0.1, 1.0, 10.0, 100.0]}))
        out.append(("decision_tree", "Decision tree",
                    DecisionTreeClassifier(max_depth=6, min_samples_leaf=5, class_weight=cw, random_state=rs),
                    {"model__max_depth": [3, 4, 6, 8, 12, None], "model__min_samples_leaf": [1, 3, 5, 10, 20]}))
        out.append(("random_forest", "Random forest",
                    RandomForestClassifier(n_estimators=200, min_samples_leaf=2, class_weight=cw, random_state=rs, n_jobs=nj),
                    {"model__max_depth": [None, 6, 10, 16], "model__min_samples_leaf": [1, 2, 5, 10],
                     "model__max_features": ["sqrt", 0.5, 1.0]}))
        out.append(("hist_gradient_boosting", "Histogram gradient boosting",
                    HistGradientBoostingClassifier(random_state=rs, max_iter=300, early_stopping="auto", class_weight=cw),
                    {"model__learning_rate": [0.03, 0.06, 0.1, 0.2], "model__max_leaf_nodes": [15, 31, 63],
                     "model__min_samples_leaf": [5, 20, 40], "model__l2_regularization": [0.0, 0.1, 1.0]}))
        if xgb is not None:
            out.append(("xgboost", "XGBoost", xgb.XGBClassifier(n_estimators=300, learning_rate=0.06, max_depth=5,
                                                                subsample=0.9, colsample_bytree=0.9, random_state=rs,
                                                                n_jobs=nj, tree_method="hist", verbosity=0,
                                                                eval_metric="logloss"),
                        {"model__max_depth": [3, 4, 5, 6, 8], "model__learning_rate": [0.03, 0.06, 0.1, 0.2],
                         "model__subsample": [0.7, 0.85, 1.0], "model__n_estimators": [150, 300, 500]}))
        if lgb is not None:
            out.append(("lightgbm", "LightGBM", lgb.LGBMClassifier(n_estimators=300, learning_rate=0.06, random_state=rs,
                                                                   n_jobs=nj, verbose=-1, class_weight=cw),
                        {"model__num_leaves": [15, 31, 63], "model__learning_rate": [0.03, 0.06, 0.1],
                         "model__min_child_samples": [5, 20, 40], "model__n_estimators": [150, 300, 500]}))
    if quick:
        keep = {"baseline", "linear", "random_forest", "hist_gradient_boosting"}
        out = [c for c in out if c[0] in keep]
    if allowed:
        allowed_set = set(allowed) | {"baseline"}
        unknown = set(allowed) - set(available_models())
        if unknown:
            raise ValidationFailed(f"Unknown model(s): {', '.join(sorted(unknown))}")
        out = [c for c in out if c[0] in allowed_set]
    if n_rows > 50_000:  # very slow without much benefit at this size
        out = [c for c in out if c[0] != "decision_tree" or allowed]
    if n_rows > 20_000:
        # Random forests grow fully on every bootstrap sample, so their cost climbs steeply with rows.
        # Subsample each tree and keep fewer, shallower trees; boosting models handle large data well.
        for key, _, est, _ in out:
            if key == "random_forest":
                est.set_params(n_estimators=100, max_samples=min(1.0, 20_000 / n_rows), min_samples_leaf=5)
        # Run the cheaper, usually stronger boosting models before the forest so the budget goes to them first.
        out.sort(key=lambda c: c[0] == "random_forest")
    return out


def _estimate_cv_seconds(pipe, X, y, n_folds: int, rs: int) -> Optional[float]:
    """Rough cost of cross-validating `pipe`, extrapolated from one fit on a small sample."""
    n_probe = 4000
    if len(X) <= 4 * n_probe:
        return None
    idx = np.random.default_rng(rs).choice(len(X), n_probe, replace=False)
    t0 = time.time()
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            clone(pipe).fit(X.iloc[idx], np.asarray(y)[idx])
    except Exception:
        return None
    per_fit = (time.time() - t0) * (len(X) * (n_folds - 1) / n_folds) / n_probe
    return per_fit * n_folds * 1.15  # + scoring overhead


def _scoring(problem: str, primary: str) -> Dict[str, str]:
    if problem == "regression":
        s = {"neg_root_mean_squared_error": "neg_root_mean_squared_error", "neg_mean_absolute_error": "neg_mean_absolute_error",
             "r2": "r2"}
    elif problem == "binary":
        s = {"roc_auc": "roc_auc", "average_precision": "average_precision", "f1": "f1",
             "balanced_accuracy": "balanced_accuracy", "accuracy": "accuracy"}
    else:
        s = {"f1_macro": "f1_macro", "balanced_accuracy": "balanced_accuracy", "accuracy": "accuracy",
             "roc_auc_ovr": "roc_auc_ovr", "f1_weighted": "f1_weighted"}
    s.setdefault(primary, primary)
    return s


def _display(metric: str, v: Optional[float]) -> Optional[float]:
    if v is None:
        return None
    return -v if metric.startswith("neg_") else v


# ─────────────────────────────────────────────────────────────────────────────
# Training
# ─────────────────────────────────────────────────────────────────────────────

def train(df: pd.DataFrame, cfg: TrainConfig, progress: Optional[Callable[[float, str], None]] = None,
          should_cancel: Optional[Callable[[], bool]] = None, schema: Optional[List[Dict]] = None) -> Dict[str, Any]:
    t_start = time.time()
    cfg.validate()
    say = progress or (lambda f, m: None)
    cancelled = should_cancel or (lambda: False)
    budget = float(cfg.time_budget_s or settings.training_time_budget_s)
    max_rows = int(cfg.max_rows or settings.training_max_rows)
    warnings_list: List[str] = []

    if cfg.target not in df.columns:
        raise ValidationFailed(f"Unknown target column '{cfg.target}'")
    if cfg.time_column and cfg.time_column not in df.columns:
        raise ValidationFailed(f"Unknown time column '{cfg.time_column}'")
    schema = schema or infer_schema(df)
    smap = {c["name"]: c for c in schema}
    say(0.02, "Validating target and features")

    data = df
    n_missing_target = int(data[cfg.target].isna().sum())
    if n_missing_target:
        data = data[data[cfg.target].notna()]
        warnings_list.append(f"{n_missing_target:,} row(s) without a target value were excluded from training.")
    if cfg.time_column:
        if not pd.api.types.is_datetime64_any_dtype(data[cfg.time_column]):
            raise ValidationFailed(f"Time column '{cfg.time_column}' must be a datetime. Convert it in the Clean step.")
        n_bad_t = int(data[cfg.time_column].isna().sum())
        if n_bad_t:
            data = data[data[cfg.time_column].notna()]
            warnings_list.append(f"{n_bad_t:,} row(s) without a timestamp were excluded (chronological split).")
    y_raw = data[cfg.target]
    problem, notes = detect_problem_type(y_raw, smap[cfg.target]["logical_type"], cfg.problem_type)
    warnings_list += notes

    # Classification: rare classes cannot be stratified / cross-validated.
    if problem != "regression":
        vc = y_raw.astype(str).value_counts()
        rare = vc[vc < max(3, cfg.cv_folds)].index.tolist()
        if rare:
            data = data[~y_raw.astype(str).isin(rare)]
            y_raw = data[cfg.target]
            warnings_list.append(f"Class(es) {', '.join(map(str, rare[:5]))} have fewer than {max(3, cfg.cv_folds)} "
                                 "examples and were excluded; the model cannot predict them.")
            if y_raw.astype(str).nunique() < 2:
                raise ValidationFailed("Fewer than two classes have enough examples to train a classifier.")
            if problem == "multiclass" and y_raw.astype(str).nunique() == 2:
                problem = "binary"
    if len(data) < 30:
        raise ValidationFailed(f"At least 30 labelled rows are needed to train and evaluate a model (found {len(data)}).")

    sampled_from = None
    if len(data) > max_rows:
        sampled_from = len(data)
        if cfg.time_column:
            data = data.sort_values(cfg.time_column).iloc[-max_rows:]
            warnings_list.append(f"Training used the most recent {max_rows:,} of {sampled_from:,} rows (TRAINING_MAX_ROWS).")
        elif problem != "regression":
            data, _ = train_test_split(data, train_size=max_rows, stratify=data[cfg.target].astype(str),
                                       random_state=cfg.random_state)
            warnings_list.append(f"Training used a stratified sample of {max_rows:,} of {sampled_from:,} rows.")
        else:
            data = data.sample(max_rows, random_state=cfg.random_state)
            warnings_list.append(f"Training used a random sample of {max_rows:,} of {sampled_from:,} rows.")
        y_raw = data[cfg.target]  # keep the target aligned with the sampled rows

    kinds, feature_report = select_features(data, cfg, schema)
    features = kinds["numeric"] + kinds["boolean"] + kinds["categorical"] + kinds["datetime"]
    X = data[features].copy()
    for c in kinds["categorical"]:
        X[c] = X[c].astype(object).where(X[c].notna(), np.nan)

    le: Optional[LabelEncoder] = None
    if problem == "regression":
        y = y_raw.astype(float).to_numpy()
        classes: List[str] = []
    else:
        le = LabelEncoder()
        y = le.fit_transform(y_raw.astype(str))
        classes = [str(c) for c in le.classes_]

    drop, suspects = leakage_checks(X, pd.Series(y, index=X.index), problem, kinds)
    if drop:
        for c in drop:
            for k in kinds:
                if c in kinds[k]:
                    kinds[k].remove(c)
            for fr in feature_report:
                if fr["column"] == c:
                    fr.update({"status": "excluded", "reason": "target leakage (see leakage report)"})
        features = [f for f in features if f not in drop]
        X = X[features]
        if not features:
            raise ValidationFailed("All features were removed as target leakage.")
    if cancelled():
        raise Cancelled()

    # Class balance / weighting
    imbalance = None
    balanced = False
    if problem != "regression":
        counts = np.bincount(y)
        imbalance = float(counts.max() / max(counts.min(), 1))
        minority_share = float(counts.min() / counts.sum())
        balanced = cfg.class_weight == "balanced" or (cfg.class_weight == "auto" and imbalance >= 3)
        if minority_share < 0.2:
            warnings_list.append(f"Imbalanced classes (smallest class {minority_share:.1%}). Accuracy is misleading here: "
                                 "a model that always predicts the majority class would score "
                                 f"{counts.max() / counts.sum():.1%}. Use balanced accuracy, recall and PR-AUC."
                                 + (" Class weights were balanced during training." if balanced else ""))

    # Hold-out split
    say(0.08, "Splitting data into training and test sets")
    if cfg.time_column:
        order = np.argsort(data[cfg.time_column].to_numpy(), kind="stable")
        X, y = X.iloc[order], y[order]
        n_test = max(int(round(len(X) * cfg.test_size)), 10)
        X_tr, X_te, y_tr, y_te = X.iloc[:-n_test], X.iloc[-n_test:], y[:-n_test], y[-n_test:]
        t_sorted = data[cfg.time_column].to_numpy()[order]
        split_info = {"method": "chronological", "train_until": str(pd.Timestamp(t_sorted[-n_test - 1])),
                      "test_from": str(pd.Timestamp(t_sorted[-n_test]))}
        cv = TimeSeriesSplit(n_splits=cfg.cv_folds)
        cv_desc = f"time-series split with {cfg.cv_folds} expanding windows"
        if problem != "regression" and len(np.unique(y_te)) < len(np.unique(y_tr)):
            warnings_list.append("Some classes do not occur in the most recent (test) period.")
    else:
        strat = y if problem != "regression" else None
        X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=cfg.test_size, random_state=cfg.random_state,
                                                  stratify=strat)
        split_info = {"method": "stratified random" if strat is not None else "random"}
        cv = (StratifiedKFold(cfg.cv_folds, shuffle=True, random_state=cfg.random_state) if problem != "regression"
              else KFold(cfg.cv_folds, shuffle=True, random_state=cfg.random_state))
        cv_desc = f"{cfg.cv_folds}-fold {'stratified ' if problem != 'regression' else ''}cross-validation"
    split_info.update({"train_rows": int(len(X_tr)), "test_rows": int(len(X_te)), "test_size": cfg.test_size})

    # Duplicate leakage between train and test
    try:
        tr_h = set(pd.util.hash_pandas_object(pd.concat([X_tr.reset_index(drop=True), pd.Series(y_tr, name="__y")], axis=1),
                                              index=False))
        te_h = pd.util.hash_pandas_object(pd.concat([X_te.reset_index(drop=True), pd.Series(y_te, name="__y")], axis=1),
                                          index=False)
        overlap = int(te_h.isin(tr_h).sum())
        if overlap:
            warnings_list.append(f"{overlap} test row(s) also appear in the training data (exact duplicates). "
                                 "Test scores may be optimistic; consider removing duplicates first.")
            suspects.append({"column": "(rows)", "reason": f"{overlap} duplicated rows across train/test",
                             "action": "remove duplicate rows in the Clean step"})
    except Exception:
        pass

    primary = cfg.primary_metric or PRIMARY[problem]
    if problem == "multiclass" and primary == "roc_auc":
        primary = "roc_auc_ovr"
    if primary not in ALLOWED_PRIMARY[problem]:
        raise ValidationFailed(f"Metric '{primary}' is not valid for {problem}. Use one of {ALLOWED_PRIMARY[problem]}.")
    scoring = _scoring(problem, primary)

    preprocessor = build_preprocessor(kinds)
    cands = _candidates(problem, balanced, len(X_tr), cfg.random_state, cfg.quick, cfg.models,
                        len(classes) if classes else 0)
    results: List[Dict[str, Any]] = []
    fitted_search: Dict[str, Any] = {}
    n_c = len(cands)
    for i, (key, name, est, space) in enumerate(cands):
        if cancelled():
            raise Cancelled()
        elapsed = time.time() - t_start
        if elapsed > budget * 0.75 and key != "baseline":
            results.append({"key": key, "model": name, "status": "skipped",
                            "reason": f"time budget reached ({budget:.0f}s)"})
            continue
        say(0.1 + 0.55 * i / n_c, f"Cross-validating {name}")
        pipe = Pipeline([("prep", preprocessor), ("model", est)])
        if key in ("random_forest", "decision_tree"):  # costs grow super-linearly; boosting scales well
            est_s = _estimate_cv_seconds(pipe, X_tr, y_tr, cfg.cv_folds, cfg.random_state)
            remaining = budget - (time.time() - t_start)
            if est_s is not None and est_s > remaining * 0.8:
                results.append({"key": key, "model": name, "status": "skipped",
                                "reason": f"estimated {est_s:.0f}s of cross-validation exceeds the remaining time budget "
                                          f"({remaining:.0f}s); raise TRAINING_TIME_BUDGET_S or train on fewer rows"})
                continue
        t0 = time.time()
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                cvr = cross_validate(pipe, X_tr, y_tr, cv=cv, scoring=scoring, return_train_score=True,
                                     n_jobs=1, error_score=np.nan)
        except Exception as e:  # incompatible model / data
            results.append({"key": key, "model": name, "status": "failed", "reason": f"{type(e).__name__}: {str(e)[:160]}"})
            continue
        if np.all(np.isnan(np.asarray(cvr[f"test_{primary}"], dtype=float))):
            results.append({"key": key, "model": name, "status": "failed",
                            "reason": "the model failed or the metric was undefined in every fold"})
            continue
        dur = time.time() - t0
        rec = {"key": key, "model": name, "status": "ok", "params": _public_params(est), "cv_seconds": round(dur, 2),
               "cv": {}, "train": {}}
        for m in scoring:
            vals = cvr.get(f"test_{m}")
            trv = cvr.get(f"train_{m}")
            if vals is None:
                continue
            vals = np.asarray(vals, dtype=float)
            rec["cv"][m] = {"mean": _display(m, float(np.nanmean(vals))), "std": float(np.nanstd(vals)),
                            "folds": [_display(m, float(v)) for v in vals]}
            if trv is not None:
                rec["train"][m] = _display(m, float(np.nanmean(trv)))
        rec["primary_cv"] = float(np.nanmean(cvr[f"test_{primary}"]))  # raw (higher is better)
        rec["primary_std"] = float(np.nanstd(cvr[f"test_{primary}"]))
        rec["generalization_gap"] = (float(np.nanmean(cvr[f"train_{primary}"])) - rec["primary_cv"]
                                     if f"train_{primary}" in cvr else None)
        results.append(rec)
        fitted_search[key] = (est, space, dur)

    ok = [r for r in results if r["status"] == "ok" and r["key"] != "baseline"]
    if not ok:
        raise ValidationFailed("No model could be trained on this data. "
                               + "; ".join(r.get("reason", "") for r in results if r["status"] == "failed")[:300])

    # Hyper-parameter search on the best two non-trivial candidates within the remaining budget.
    tuning: List[Dict[str, Any]] = []
    if cfg.tune and not cfg.quick:
        top = sorted(ok, key=lambda r: -r["primary_cv"])[:2]
        for r in top:
            est, space, dur = fitted_search[r["key"]]
            if not space:
                continue
            remaining = budget - (time.time() - t_start)
            per_iter = max(dur, 0.05)
            n_iter = int(min(cfg.n_iter, remaining * 0.6 / per_iter))
            if n_iter < 2:
                tuning.append({"model": r["model"], "status": "skipped", "reason": "not enough time budget left"})
                continue
            if cancelled():
                raise Cancelled()
            say(0.68, f"Tuning {r['model']} ({n_iter} configurations)")
            search = RandomizedSearchCV(Pipeline([("prep", preprocessor), ("model", est)]), space, n_iter=n_iter,
                                        scoring=scoring[primary], cv=cv, random_state=cfg.random_state, n_jobs=1,
                                        error_score=np.nan, refit=False)
            try:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    search.fit(X_tr, y_tr)
            except Exception as e:
                tuning.append({"model": r["model"], "status": "failed", "reason": str(e)[:160]})
                continue
            best_score = float(search.best_score_)
            improved = best_score > r["primary_cv"] + 1e-9
            tuning.append({"model": r["model"], "status": "ok", "n_iter": n_iter,
                           "best_params": {k.replace("model__", ""): to_jsonable(v) for k, v in search.best_params_.items()},
                           "cv_before": _display(primary, r["primary_cv"]), "cv_after": _display(primary, best_score),
                           "improved": improved})
            if improved:
                idx = search.best_index_
                r["tuned"] = True
                r["params"].update({k.replace("model__", ""): to_jsonable(v) for k, v in search.best_params_.items()})
                r["primary_cv"] = best_score
                r["primary_std"] = float(search.cv_results_["std_test_score"][idx])
                r["cv"][primary] = {"mean": _display(primary, best_score), "std": r["primary_std"], "folds": None}
                r["search_params"] = search.best_params_

    ranked = sorted(ok, key=lambda r: -r["primary_cv"])
    best = ranked[0]
    baseline = next((r for r in results if r["key"] == "baseline" and r["status"] == "ok"), None)

    # Refit the selected pipeline on the full training split
    say(0.8, f"Fitting {best['model']} on the full training split")
    est, _, _ = fitted_search[best["key"]]
    final = Pipeline([("prep", preprocessor), ("model", est)])
    if best.get("search_params"):
        final.set_params(**best["search_params"])
    t0 = time.time()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        final.fit(X_tr, y_tr)
    fit_seconds = time.time() - t0

    # Single evaluation on the untouched test split
    say(0.86, "Evaluating on the held-out test set")
    base_pipe = Pipeline([("prep", preprocessor), ("model", fitted_search["baseline"][0])]) if "baseline" in fitted_search else None
    if base_pipe is not None:
        base_pipe.fit(X_tr, y_tr)
    if problem == "regression":
        test_metrics, diagnostics = evaluate_regression(final, X_te, y_te)
        base_metrics = evaluate_regression(base_pipe, X_te, y_te)[0] if base_pipe is not None else None
    else:
        test_metrics, diagnostics = evaluate_classification(final, X_te, y_te, classes, problem)
        base_metrics = evaluate_classification(base_pipe, X_te, y_te, classes, problem)[0] if base_pipe is not None else None

    say(0.92, "Computing feature importance")
    importance = feature_importance(final, X_te, y_te, scoring[primary], features, cfg.random_state)

    for r in results:
        r.pop("search_params", None)
    rationale = _rationale(best, ranked, baseline, primary, cv_desc, test_metrics, base_metrics, problem)
    limitations = _limitations(problem, len(X_tr), len(X_te), imbalance, best, suspects, cfg)

    schema_out = input_schema(data, kinds, smap)
    duration = time.time() - t_start
    say(1.0, "Done")
    result = {
        "problem_type": problem, "target": cfg.target, "classes": classes, "primary_metric": primary,
        "primary_metric_label": METRIC_LABELS.get(primary, primary),
        "config": to_jsonable(asdict(cfg)), "split": split_info, "cv_description": cv_desc,
        "candidates": to_jsonable([_strip(r) for r in results]), "tuning": to_jsonable(tuning),
        "selected": {"key": best["key"], "model": best["model"], "params": to_jsonable(best["params"]),
                     "tuned": bool(best.get("tuned"))},
        "rationale": rationale, "test_metrics": to_jsonable(test_metrics),
        "baseline_test_metrics": to_jsonable(base_metrics), "diagnostics": to_jsonable(diagnostics),
        "feature_importance": to_jsonable(importance), "features": feature_report, "leakage": to_jsonable(suspects),
        "warnings": warnings_list, "limitations": limitations,
        "rows": {"used": int(len(X)), "train": int(len(X_tr)), "test": int(len(X_te)),
                 "excluded_missing_target": n_missing_target, "sampled_from": sampled_from},
        "duration_s": round(duration, 2), "fit_seconds": round(fit_seconds, 2),
        "versions": library_versions(), "input_schema": schema_out,
        "class_weight": "balanced" if balanced else "none",
    }
    return {"result": to_jsonable(result), "pipeline": final, "label_classes": classes}


def _strip(r: Dict[str, Any]) -> Dict[str, Any]:
    out = dict(r)
    if "primary_cv" in out:
        out["primary_cv"] = out["primary_cv"]
    return out


def _public_params(est) -> Dict[str, Any]:
    keep = ("alpha", "C", "max_depth", "min_samples_leaf", "n_estimators", "learning_rate", "max_leaf_nodes",
            "l2_regularization", "max_features", "num_leaves", "subsample", "class_weight", "strategy",
            "min_child_samples", "max_iter")
    p = est.get_params()
    return {k: to_jsonable(v) for k, v in p.items() if k in keep}


def _rationale(best, ranked, baseline, primary, cv_desc, test_metrics, base_metrics, problem) -> str:
    lab = METRIC_LABELS.get(primary, primary)
    bm = best["cv"][primary]["mean"]
    parts = [f"{best['model']} was selected because it had the best mean {lab} in {cv_desc} on the training split "
             f"({bm:.4g} ± {best['primary_std']:.3g})."]
    if len(ranked) > 1:
        sec = ranked[1]
        sm = sec["cv"][primary]["mean"]
        diff = abs(best["primary_cv"] - sec["primary_cv"])
        within = diff <= max(best["primary_std"], sec["primary_std"])
        parts.append(f"Runner-up: {sec['model']} ({sm:.4g})." + (" The difference is within one standard deviation "
                     "of the fold scores, so both models perform similarly." if within else ""))
    if baseline:
        parts.append(f"A naive baseline scored {baseline['cv'][primary]['mean']:.4g}.")
    parts.append("The test set was not used for model selection; it was evaluated once after selection.")
    gap = best.get("generalization_gap")
    if gap is not None and abs(best["primary_cv"]) > 0 and gap > 0.1 * abs(best["primary_cv"]):
        parts.append("Training scores are noticeably better than cross-validation scores, which indicates some overfitting.")
    return " ".join(parts)


def _limitations(problem, n_tr, n_te, imbalance, best, suspects, cfg) -> List[str]:
    lim = []
    if n_te < 100:
        lim.append(f"The test set is small ({n_te} rows), so test metrics have wide uncertainty.")
    if n_tr < 200:
        lim.append("Little training data: results may change noticeably with more data.")
    if problem != "regression":
        lim.append("Predicted probabilities are model scores and are not necessarily calibrated; check the calibration "
                   "curve before treating them as true probabilities.")
    if imbalance and imbalance >= 3:
        lim.append("Classes are imbalanced; review per-class recall rather than overall accuracy.")
    if any(s["action"].startswith("kept") for s in suspects):
        lim.append("Some features look suspiciously predictive (see the leakage report); confirm they are available "
                   "at prediction time.")
    if not cfg.time_column:
        lim.append("Rows were split randomly. If the data has a time order and you will predict the future, retrain "
                   "with a time column for a realistic estimate.")
    lim.append("Feature importance shows what the model relies on, not what causes the outcome.")
    return lim


# ─────────────────────────────────────────────────────────────────────────────
# Evaluation
# ─────────────────────────────────────────────────────────────────────────────

def evaluate_regression(model, X, y) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    pred = model.predict(X)
    y = np.asarray(y, dtype=float)
    resid = y - pred
    m = {"mae": float(mean_absolute_error(y, pred)), "rmse": float(math.sqrt(mean_squared_error(y, pred))),
         "r2": float(r2_score(y, pred)), "median_ae": float(median_absolute_error(y, pred)), "n": int(len(y))}
    if np.all(np.abs(y) > 1e-9):
        m["mape_pct"] = float(mean_absolute_percentage_error(y, pred) * 100)
    counts, edges = np.histogram(resid, bins=30)
    idx = np.linspace(0, len(y) - 1, min(len(y), 500)).astype(int)
    from scipy.stats import spearmanr
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        het = spearmanr(np.abs(resid), pred).statistic if len(y) > 5 and np.std(pred) > 0 else None
    diag = {"residual_histogram": {"edges": edges.tolist(), "counts": counts.tolist()},
            "actual_vs_predicted": [{"actual": float(y[i]), "predicted": float(pred[i])} for i in idx],
            "residual_vs_predicted": [{"predicted": float(pred[i]), "residual": float(resid[i])} for i in idx],
            "residual_mean": float(resid.mean()), "residual_std": float(resid.std()),
            "heteroscedasticity_rho": to_jsonable(het),
            "notes": (["Errors grow with the predicted value (heteroscedasticity); consider a log transform of the target."]
                      if het is not None and het > 0.3 else [])}
    return m, diag


def evaluate_classification(model, X, y, classes: List[str], problem: str) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    y = np.asarray(y)
    pred = model.predict(X)
    labels = list(range(len(classes)))
    proba = model.predict_proba(X) if hasattr(model, "predict_proba") else None
    m: Dict[str, Any] = {"accuracy": float(accuracy_score(y, pred)),
                         "balanced_accuracy": float(balanced_accuracy_score(y, pred)),
                         "precision_macro": float(precision_score(y, pred, average="macro", zero_division=0)),
                         "recall_macro": float(recall_score(y, pred, average="macro", zero_division=0)),
                         "f1_macro": float(f1_score(y, pred, average="macro", zero_division=0)),
                         "f1_weighted": float(f1_score(y, pred, average="weighted", zero_division=0)),
                         "mcc": float(matthews_corrcoef(y, pred)), "n": int(len(y))}
    diag: Dict[str, Any] = {}
    if problem == "binary":
        m.update({"precision": float(precision_score(y, pred, zero_division=0)),
                  "recall": float(recall_score(y, pred, zero_division=0)),
                  "f1": float(f1_score(y, pred, zero_division=0)), "positive_class": classes[1]})
    if proba is not None and len(np.unique(y)) > 1:
        try:
            if problem == "binary":
                p1 = proba[:, 1]
                m["roc_auc"] = float(roc_auc_score(y, p1))
                m["average_precision"] = float(average_precision_score(y, p1))
                m["brier"] = float(brier_score_loss(y, p1))
                fpr, tpr, _ = roc_curve(y, p1)
                pr, rc, _ = precision_recall_curve(y, p1)
                diag["roc_curve"] = _thin([{"fpr": float(a), "tpr": float(b)} for a, b in zip(fpr, tpr)])
                diag["pr_curve"] = _thin([{"recall": float(a), "precision": float(b)} for a, b in zip(rc, pr)])
                diag["prevalence"] = float(np.mean(y))
                diag["calibration"] = _calibration(y, p1)
            else:
                m["roc_auc_ovr"] = float(roc_auc_score(y, proba, multi_class="ovr", labels=labels))
            m["log_loss"] = float(log_loss(y, proba, labels=labels))
        except ValueError:
            pass
    cm = confusion_matrix(y, pred, labels=labels)
    p, r, f, s = precision_recall_fscore_support(y, pred, labels=labels, zero_division=0)
    diag["confusion_matrix"] = {"labels": classes, "matrix": cm.tolist()}
    diag["per_class"] = [{"class": classes[i], "precision": float(p[i]), "recall": float(r[i]), "f1": float(f[i]),
                          "support": int(s[i])} for i in labels]
    diag["majority_baseline_accuracy"] = float(np.bincount(y, minlength=len(classes)).max() / len(y))
    return m, diag


def _thin(points: List[Dict], k: int = 150) -> List[Dict]:
    if len(points) <= k:
        return points
    idx = np.linspace(0, len(points) - 1, k).astype(int)
    return [points[i] for i in idx]


def _calibration(y, p, bins: int = 10) -> List[Dict[str, Any]]:
    edges = np.linspace(0, 1, bins + 1)
    out = []
    for i in range(bins):
        mask = (p >= edges[i]) & (p < edges[i + 1] if i < bins - 1 else p <= edges[i + 1])
        if mask.sum() >= 3:
            out.append({"bin": f"{edges[i]:.1f}–{edges[i + 1]:.1f}", "mean_predicted": float(p[mask].mean()),
                        "observed_rate": float(np.mean(y[mask])), "n": int(mask.sum())})
    return out


# ─────────────────────────────────────────────────────────────────────────────
# Explanations
# ─────────────────────────────────────────────────────────────────────────────

def feature_importance(model, X, y, scoring: str, features: List[str], rs: int) -> Dict[str, Any]:
    Xs, ys = X, y
    if len(X) > 4000:
        idx = np.random.default_rng(rs).choice(len(X), 4000, replace=False)
        Xs, ys = X.iloc[idx], np.asarray(y)[idx]
    if len(features) > 80:
        return {"method": None, "items": [], "notes": ["Too many features for permutation importance (> 80)."]}
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        try:
            pi = permutation_importance(model, Xs, ys, scoring=scoring, n_repeats=5, random_state=rs, n_jobs=1)
        except Exception as e:
            return {"method": None, "items": [], "notes": [f"Permutation importance failed: {type(e).__name__}"]}
        n_cls = len(np.unique(ys))
        if not hasattr(model, "predict_proba"):
            pred = model.predict(Xs)
        elif n_cls == 2:
            pred = model.predict_proba(Xs)[:, 1]
        else:
            pred = None  # direction is undefined for multiclass outputs
    items = []
    for i, f in enumerate(features):
        direction = None
        s = Xs[f]
        if pred is not None and pd.api.types.is_numeric_dtype(s) and not pd.api.types.is_bool_dtype(s) and s.nunique() > 2:
            r = pd.Series(np.asarray(pred, dtype=float)).corr(s.reset_index(drop=True).astype(float), method="spearman")
            if pd.notna(r) and abs(r) >= 0.1:
                direction = "higher values → higher prediction" if r > 0 else "higher values → lower prediction"
        items.append({"feature": f, "importance": float(pi.importances_mean[i]), "std": float(pi.importances_std[i]),
                      "direction": direction})
    items.sort(key=lambda d: -d["importance"])
    total = sum(max(0, d["importance"]) for d in items) or 1
    for d in items:
        d["share"] = max(0, d["importance"]) / total
    return {"method": "permutation importance on the test set (5 repeats)", "items": items,
            "notes": ["Importance = drop in the primary metric when the feature's values are shuffled.",
                      "Correlated features share importance, so each may look less important than it is.",
                      "Direction describes how the model's predictions move with the feature — it is not a causal effect."]}


def explain_local(model, row: pd.DataFrame, meta: Dict[str, Any], classes: List[str]) -> Dict[str, Any]:
    """Occlusion explanation: replace each feature by a typical value and measure the change in prediction."""
    schema = meta["input_schema"]
    problem = meta["problem_type"]

    def score(df):
        if problem == "regression":
            return model.predict(df).astype(float)
        p = model.predict_proba(df)
        return p

    base_pred = score(row)
    if problem == "regression":
        target_idx = None
        base_val = float(base_pred[0])
    else:
        target_idx = int(np.argmax(base_pred[0]))
        base_val = float(base_pred[0][target_idx])
    contribs = []
    for f in schema["features"]:
        name = f["name"]
        typical = f.get("typical")
        if typical is None:
            continue
        mod = row.copy()
        if f["kind"] == "datetime":
            mod[name] = pd.Timestamp(typical)
        else:
            mod[name] = typical
        cur = row[name].iloc[0]
        try:
            if (pd.isna(cur) and pd.isna(typical)) or cur == typical:
                continue
        except (TypeError, ValueError):
            pass
        alt = score(mod)
        alt_val = float(alt[0]) if problem == "regression" else float(alt[0][target_idx])
        contribs.append({"feature": name, "value": to_jsonable(cur), "typical": to_jsonable(typical),
                         "contribution": base_val - alt_val})
    contribs.sort(key=lambda c: -abs(c["contribution"]))
    return {"method": "occlusion (feature replaced by its typical training value)",
            "explained_output": "predicted value" if problem == "regression" else f"probability of '{classes[target_idx]}'",
            "prediction": base_val, "contributions": contribs[:15],
            "notes": ["Contributions are approximate, ignore feature interactions and do not sum to the prediction.",
                      "They explain this model's behaviour, not causes in the real world."]}


def input_schema(data: pd.DataFrame, kinds: Dict[str, List[str]], smap: Dict[str, Dict]) -> Dict[str, Any]:
    feats = []
    for kind in ("numeric", "boolean", "categorical", "datetime"):
        for c in kinds[kind]:
            s = data[c]
            f: Dict[str, Any] = {"name": c, "kind": kind, "dtype": str(s.dtype), "missing_pct": float(s.isna().mean() * 100)}
            if kind == "numeric":
                v = s.dropna().astype(float)
                f.update({"min": to_jsonable(v.min()), "max": to_jsonable(v.max()), "median": to_jsonable(v.median()),
                          "mean": to_jsonable(v.mean()), "is_integer": bool(np.allclose(v, np.round(v))) if len(v) else False,
                          "typical": to_jsonable(v.median())})
            elif kind in ("categorical", "boolean"):
                vc = s.dropna().astype(str).value_counts()
                f.update({"categories": [str(k) for k in vc.index[:200]], "n_unique": int(len(vc)),
                          "typical": str(vc.index[0]) if len(vc) else None})
                if kind == "boolean":
                    f["typical"] = s.dropna().mode().iloc[0] if s.notna().any() else None
                    f["typical"] = to_jsonable(f["typical"])
            else:
                v = pd.to_datetime(s.dropna())
                f.update({"min": v.min().isoformat() if len(v) else None, "max": v.max().isoformat() if len(v) else None,
                          "typical": v.median().isoformat() if len(v) else None})
            feats.append(f)
    return {"features": feats, "order": [f["name"] for f in feats]}
