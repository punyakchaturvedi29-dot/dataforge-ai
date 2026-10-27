"""
Multi-file analysis: schema comparison, evidence-based join discovery,
validated joins and stacking.

Join keys are recommended from the *values*, not just the names:
  * containment – share of each side's distinct keys found on the other side,
  * row coverage – share of left rows that would find a match,
  * key-ness – how unique the key is on each side,
  * cardinality (1:1, 1:N, N:1, N:M) and the exact number of output rows,
with name similarity used only as a minor tie-breaker. A join that would
multiply rows (N:M "join explosion") or match few rows requires explicit
confirmation.
"""
from __future__ import annotations

import difflib
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from core.errors import ValidationFailed
from core.jsonsafe import records, to_jsonable

from .schema import infer_schema, is_text_dtype, name_tokens

MAX_KEY_UNIQUES = 500_000


def key_series(s: pd.Series, case_insensitive: bool = False) -> pd.Series:
    """Normalise join-key values so 1001, 1001.0 and ' 1001 ' compare equal."""
    if pd.api.types.is_bool_dtype(s):
        return s.astype("string")
    if pd.api.types.is_numeric_dtype(s):
        v = s.astype(float)
        if np.all(np.isclose(v.dropna() % 1, 0)):
            return v.round().astype("Int64").astype("string")
        return v.astype("string")
    if pd.api.types.is_datetime64_any_dtype(s):
        return s.dt.strftime("%Y-%m-%dT%H:%M:%S").astype("string")
    t = s.astype("string").str.strip()
    t = t.str.replace(r"^(-?\d+)\.0+$", r"\1", regex=True)
    return t.str.lower() if case_insensitive else t


def _kind(s: pd.Series) -> str:
    if pd.api.types.is_datetime64_any_dtype(s):
        return "datetime"
    if pd.api.types.is_numeric_dtype(s) and not pd.api.types.is_bool_dtype(s):
        v = s.dropna()
        return "integer" if len(v) and np.allclose(v % 1, 0) else "float"
    return "text"


def compare_schemas(datasets: List[Tuple[str, str, pd.DataFrame]]) -> Dict[str, Any]:
    """datasets: [(id, name, df)]"""
    all_cols: Dict[str, Dict[str, Any]] = {}
    infos = []
    for did, name, df in datasets:
        sch = {c["name"]: c for c in infer_schema(df)}
        infos.append({"id": did, "name": name, "rows": int(len(df)), "columns": int(len(df.columns))})
        for c in df.columns:
            all_cols.setdefault(c, {})[did] = {"dtype": str(df[c].dtype), "logical_type": sch[c]["logical_type"],
                                               "missing_pct": sch[c]["missing_pct"], "n_unique": sch[c]["n_unique"]}
    rows = []
    for c, per in all_cols.items():
        types = {v["logical_type"] for v in per.values()}
        rows.append({"column": c, "present_in": list(per.keys()), "per_dataset": per,
                     "in_all": len(per) == len(datasets), "type_conflict": len(types) > 1})
    rows.sort(key=lambda r: (not r["in_all"], r["column"]))
    return to_jsonable({"datasets": infos, "columns": rows,
                        "common_columns": [r["column"] for r in rows if r["in_all"]]})


def _is_row_counter(s: pd.Series) -> bool:
    if _kind(s) != "integer" and not (is_text_dtype(s) and s.dropna().astype(str).str.fullmatch(r"\d+").all()):
        return False
    v = pd.to_numeric(s, errors="coerce").dropna()
    if len(v) < 10 or v.nunique() != len(v):
        return False
    u = np.sort(v.to_numpy())
    return bool((np.diff(u) == 1).mean() > 0.95)


def _name_similarity(a: str, b: str) -> float:
    ta, tb = set(name_tokens(a)), set(name_tokens(b))
    tok = len(ta & tb) / len(ta | tb) if ta | tb else 0
    return max(tok, difflib.SequenceMatcher(None, a.lower(), b.lower()).ratio())


def _profile_key(s: pd.Series):
    k = key_series(s)
    vc = k.value_counts(dropna=True)
    return k, vc


def join_candidates(df_a: pd.DataFrame, df_b: pd.DataFrame, top: int = 8) -> List[Dict[str, Any]]:
    cols_a = [c for c in df_a.columns if _kind(df_a[c]) != "float" and df_a[c].nunique() >= 2]
    cols_b = [c for c in df_b.columns if _kind(df_b[c]) != "float" and df_b[c].nunique() >= 2]
    prof_a = {c: _profile_key(df_a[c]) for c in cols_a[:60] if df_a[c].nunique() <= MAX_KEY_UNIQUES}
    prof_b = {c: _profile_key(df_b[c]) for c in cols_b[:60] if df_b[c].nunique() <= MAX_KEY_UNIQUES}
    out = []
    for ca, (ka, vca) in prof_a.items():
        set_a = set(vca.index)
        for cb, (kb, vcb) in prof_b.items():
            if _kind(df_a[ca]) != _kind(df_b[cb]) and not ({_kind(df_a[ca]), _kind(df_b[cb])} <= {"integer", "text"}):
                continue
            common = set_a & set(vcb.index)
            if not common:
                continue
            cont_a = len(common) / len(vca)
            cont_b = len(common) / len(vcb)
            if max(cont_a, cont_b) < 0.05:
                continue
            ca_counts = vca.loc[list(common)]
            cb_counts = vcb.loc[list(common)]
            row_cov_a = float(ca_counts.sum() / max(ka.notna().sum(), 1))
            row_cov_b = float(cb_counts.sum() / max(kb.notna().sum(), 1))
            uniq_a = len(vca) / max(ka.notna().sum(), 1)
            uniq_b = len(vcb) / max(kb.notna().sum(), 1)
            dup_a = bool((ca_counts > 1).any())
            dup_b = bool((cb_counts > 1).any())
            card = {(False, False): "1:1", (False, True): "1:N", (True, False): "N:1", (True, True): "N:M"}[(dup_a, dup_b)]
            inner_rows = int((ca_counts * cb_counts).sum())
            left_rows = inner_rows + int(ka.notna().sum() - ca_counts.sum()) + int(ka.isna().sum())
            name_sim = _name_similarity(str(ca), str(cb))
            keyness = max(uniq_a, uniq_b)
            # Evidence that matters depends on the relationship: for N:1 the many side's keys must be
            # found on the one side; for 1:1 both directions should overlap.
            if card == "N:1":
                fk_cont, coverage = cont_a, row_cov_a
            elif card == "1:N":
                fk_cont, coverage = cont_b, row_cov_b
            else:
                fk_cont, coverage = float(np.sqrt(cont_a * cont_b)), min(row_cov_a, row_cov_b)
            score = 0.45 * fk_cont + 0.25 * coverage + 0.15 * keyness + 0.15 * name_sim
            reasons = [f"{cont_a:.0%} of '{ca}' values appear in '{cb}' and {cont_b:.0%} the other way.",
                       f"{row_cov_a:.0%} of left rows would find a match.",
                       f"Uniqueness: {uniq_a:.0%} (left), {uniq_b:.0%} (right) → {card}."]
            if card == "N:M":
                score *= 0.5
            if len(common) < 3:
                score *= 0.5
            if _is_row_counter(df_a[ca]) and _is_row_counter(df_b[cb]) and name_sim < 0.8:
                score *= 0.6
                reasons.append("Both columns look like auto-increment counters (1, 2, 3, …); overlapping values may be "
                               "coincidental rather than a real relationship.")
            if card == "N:M":
                reasons.append(f"Many-to-many: an inner join would produce {inner_rows:,} rows from "
                               f"{len(df_a):,} × {len(df_b):,} — likely a join explosion.")
            if _kind(df_a[ca]) != _kind(df_b[cb]):
                reasons.append("Types differ (number vs text); values are compared after normalisation.")
            out.append({"left_on": ca, "right_on": cb, "score": round(score, 4), "cardinality": card,
                        "overlap_keys": len(common), "containment_left": cont_a, "containment_right": cont_b,
                        "row_coverage_left": row_cov_a, "row_coverage_right": row_cov_b,
                        "uniqueness_left": uniq_a, "uniqueness_right": uniq_b, "name_similarity": name_sim,
                        "estimated_rows": {"inner": inner_rows, "left": left_rows}, "reasons": reasons})
    out.sort(key=lambda r: -r["score"])
    return to_jsonable(out[:top])


def _merge(df_a, df_b, left_on, right_on, how, case_insensitive=False):
    if how not in ("inner", "left", "right", "outer"):
        raise ValidationFailed("how must be inner, left, right or outer")
    if left_on not in df_a.columns:
        raise ValidationFailed(f"Unknown left key '{left_on}'")
    if right_on not in df_b.columns:
        raise ValidationFailed(f"Unknown right key '{right_on}'")
    a = df_a.copy()
    b = df_b.copy()
    a["__k"] = key_series(a[left_on], case_insensitive)
    b["__k"] = key_series(b[right_on], case_insensitive)
    collisions = [c for c in b.columns if c in a.columns and c != "__k" and not (c == right_on and left_on == right_on)]
    if left_on == right_on:
        b = b.drop(columns=[right_on])
    merged = pd.merge(a, b, on="__k", how=how, suffixes=("", "_right"), indicator=True)
    return merged, collisions


def preview_join(df_a: pd.DataFrame, df_b: pd.DataFrame, left_on: str, right_on: str, how: str = "left",
                 case_insensitive: bool = False, sample: int = 20) -> Dict[str, Any]:
    ka, kb = key_series(df_a[left_on], case_insensitive), key_series(df_b[right_on], case_insensitive)
    vca, vcb = ka.value_counts(), kb.value_counts()
    common = vca.index.intersection(vcb.index)
    est_inner = int((vca[common] * vcb[common]).sum())
    est = {"inner": est_inner,
           "left": est_inner + int(len(ka) - vca[common].sum()),
           "right": est_inner + int(len(kb) - vcb[common].sum()),
           "outer": est_inner + int(len(ka) - vca[common].sum()) + int(len(kb) - vcb[common].sum())}
    if est[how] > 5_000_000:
        raise ValidationFailed(f"This join would produce about {est[how]:,} rows, which exceeds the 5,000,000-row limit. "
                               "Check the key columns: this is usually a many-to-many join explosion.")
    merged, collisions = _merge(df_a, df_b, left_on, right_on, how, case_insensitive)
    ind = merged["_merge"]
    dup_a = int((vca > 1).sum())
    dup_b = int((vcb > 1).sum())
    card = {(False, False): "1:1", (False, True): "1:N", (True, False): "N:1", (True, True): "N:M"}[
        (bool((vca[common] > 1).any()) if len(common) else False, bool((vcb[common] > 1).any()) if len(common) else False)]
    matched_left = int(ka.isin(common).sum())
    unmatched_left_keys = ka[~ka.isin(common) & ka.notna()].unique()[:10].tolist()
    unmatched_right_keys = kb[~kb.isin(common) & kb.notna()].unique()[:10].tolist()
    explosion = len(merged) / max(len(df_a), 1)
    warnings: List[str] = []
    if card == "N:M":
        warnings.append("Many-to-many join: rows are duplicated for every combination of matching keys. "
                        "Aggregate one side first or use a more specific key.")
    if explosion > 1.05 and how in ("left", "inner"):
        warnings.append(f"The result has {explosion:.2f}× as many rows as the left dataset, so left-side "
                        "totals will be double-counted.")
    match_rate = matched_left / max(int(ka.notna().sum()), 1)
    if match_rate < 0.5:
        warnings.append(f"Only {match_rate:.0%} of left rows find a match. Check that the keys refer to the same thing.")
    if ka.isna().any() or kb.isna().any():
        warnings.append(f"Missing keys never match: {int(ka.isna().sum())} on the left, {int(kb.isna().sum())} on the right.")
    if collisions:
        warnings.append(f"Columns present in both datasets were renamed with a '_right' suffix: {', '.join(collisions[:6])}.")
    if _kind(df_a[left_on]) != _kind(df_b[right_on]):
        warnings.append("Key types differ (number vs text); keys were compared after normalising both to text.")
    requires_confirmation = card == "N:M" or (explosion > 1.05 and how in ("left", "inner")) or match_rate < 0.5
    out_cols = [c for c in merged.columns if c not in ("__k", "_merge")]
    return to_jsonable({
        "left_on": left_on, "right_on": right_on, "how": how, "case_insensitive": case_insensitive,
        "rows_left": int(len(df_a)), "rows_right": int(len(df_b)), "rows_result": int(len(merged)),
        "columns_result": len(out_cols), "estimated_rows": est, "cardinality": card,
        "matched_left_rows": matched_left, "unmatched_left_rows": int((ind == "left_only").sum()),
        "unmatched_right_rows": int((ind == "right_only").sum()),
        "match_rate_left": match_rate, "duplicate_keys_left": dup_a, "duplicate_keys_right": dup_b,
        "null_keys_left": int(ka.isna().sum()), "null_keys_right": int(kb.isna().sum()),
        "explosion_factor": explosion, "column_collisions": collisions,
        "unmatched_left_samples": unmatched_left_keys, "unmatched_right_samples": unmatched_right_keys,
        "preview_columns": out_cols, "preview": records(merged[out_cols].head(sample)),
        "warnings": warnings, "requires_confirmation": requires_confirmation,
    })


def execute_join(df_a, df_b, left_on, right_on, how="left", case_insensitive=False) -> pd.DataFrame:
    merged, _ = _merge(df_a, df_b, left_on, right_on, how, case_insensitive)
    return merged.drop(columns=["__k", "_merge"]).reset_index(drop=True)


def stack(datasets: List[Tuple[str, pd.DataFrame]], mode: str = "union", source_column: str = "source_dataset") -> Tuple[pd.DataFrame, Dict[str, Any]]:
    if len(datasets) < 2:
        raise ValidationFailed("Select at least two datasets to stack.")
    if mode not in ("union", "intersection"):
        raise ValidationFailed("mode must be union or intersection")
    col_sets = [set(df.columns) for _, df in datasets]
    common = set.intersection(*col_sets)
    allc = []
    for _, df in datasets:
        for c in df.columns:
            if c not in allc:
                allc.append(c)
    use = [c for c in allc if c in common] if mode == "intersection" else allc
    if not common:
        raise ValidationFailed("The datasets have no columns in common; stacking would not be meaningful.")
    conflicts = []
    for c in use:
        kinds = {_kind(df[c]) for _, df in datasets if c in df.columns}
        if len(kinds - {"integer", "float"}) > 1 or (kinds & {"text"} and kinds - {"text"}):
            conflicts.append({"column": c, "types": sorted(kinds)})
    if any(source_column in df.columns for _, df in datasets):
        source_column = source_column + "_"
    parts = []
    for name, df in datasets:
        part = df.reindex(columns=use).copy()
        for cf in conflicts:
            if cf["column"] in part.columns:
                part[cf["column"]] = part[cf["column"]].map(lambda v: v if pd.isna(v) else str(v))
        part[source_column] = name
        parts.append(part)
    out = pd.concat(parts, ignore_index=True)
    report = {"mode": mode, "columns_used": len(use), "common_columns": sorted(common),
              "columns_filled_with_missing": sorted(set(use) - common),
              "type_conflicts": conflicts, "per_dataset_rows": [{"name": n, "rows": int(len(d))} for n, d in datasets],
              "rows_result": int(len(out)), "source_column": source_column,
              "warnings": ([f"{len(conflicts)} column(s) had incompatible types and were stored as text."] if conflicts else [])
                          + ([f"{len(set(use) - common)} column(s) are missing in some datasets and were filled with missing values."]
                             if mode == "union" and set(use) - common else [])}
    return out, to_jsonable(report)
