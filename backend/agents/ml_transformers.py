"""
Custom scikit-learn transformers used inside persisted model pipelines.

They live in their own module with a stable import path because saved model
artifacts reference these classes by name. Do not move or rename them without
a migration.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin


class ToString(BaseEstimator, TransformerMixin):
    """Cast categorical inputs to plain strings so encoders see one consistent type."""

    def fit(self, X, y=None):
        self.n_features_in_ = X.shape[1]
        return self

    def transform(self, X):
        df = pd.DataFrame(X).copy()
        for c in df.columns:
            s = df[c]
            df[c] = s.map(lambda v: np.nan if v is None or (isinstance(v, float) and np.isnan(v)) or v is pd.NA else str(v))
        return df.astype(object).to_numpy()

    def get_feature_names_out(self, input_features=None):
        return np.asarray(input_features if input_features is not None else [], dtype=object)


class DateFeatures(BaseEstimator, TransformerMixin):
    """Expand datetime columns into numeric calendar features (year, month, day of week, day of year, hour)."""

    def fit(self, X, y=None):
        df = pd.DataFrame(X)
        self.columns_ = [str(c) for c in df.columns]
        self.has_time_ = {}
        for c in df.columns:
            s = pd.to_datetime(df[c], errors="coerce")
            self.has_time_[str(c)] = bool(((s.dt.hour != 0) | (s.dt.minute != 0)).fillna(False).any())
        return self

    def transform(self, X):
        df = pd.DataFrame(X)
        df.columns = self.columns_[: len(df.columns)]
        out = {}
        for c in self.columns_:
            s = pd.to_datetime(df[c], errors="coerce")
            out[f"{c}__year"] = s.dt.year
            out[f"{c}__month"] = s.dt.month
            out[f"{c}__dayofweek"] = s.dt.day_of_week
            out[f"{c}__dayofyear"] = s.dt.day_of_year
            if self.has_time_.get(c):
                out[f"{c}__hour"] = s.dt.hour
        return pd.DataFrame(out).astype(float).to_numpy()

    def get_feature_names_out(self, input_features=None):
        names = []
        for c in self.columns_:
            names += [f"{c}__year", f"{c}__month", f"{c}__dayofweek", f"{c}__dayofyear"]
            if self.has_time_.get(c):
                names.append(f"{c}__hour")
        return np.asarray(names, dtype=object)


class BoolToFloat(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        df = pd.DataFrame(X)
        out = df.apply(lambda s: s.map(lambda v: np.nan if v is None or v is pd.NA else float(bool(v))
                                       if not isinstance(v, str) else {"true": 1.0, "false": 0.0, "yes": 1.0,
                                                                       "no": 0.0, "1": 1.0, "0": 0.0}.get(v.strip().lower(), np.nan)))
        return out.astype(float).to_numpy()

    def get_feature_names_out(self, input_features=None):
        return np.asarray(input_features if input_features is not None else [], dtype=object)
