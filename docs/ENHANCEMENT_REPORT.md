# DataForge AI v5 — Enhancement Report

This report describes what was found in the supplied v4 archive (`DataForge_AI_v4_Advanced.zip`),
what was changed, and why. Verification evidence is in [VERIFICATION_REPORT.md](VERIFICATION_REPORT.md);
remaining work is in [ROADMAP.md](ROADMAP.md).

## 1. Starting point (audit of v4)

v4 was a React/Vite single page with tabbed panels over a FastAPI backend (~6,800 lines in total).
The main problems, in order of impact:

| Area | Finding in v4 | Consequence |
|---|---|---|
| Installability | Exact pins (`pandas==2.2.2`, `numpy==1.26.4`, `scikit-learn==1.4.2`, …) | `pip install` fails on current Python (3.13) because no wheels exist for those pins. |
| State | All data in a module-level dict (`store._store`) | Every restart lost all datasets and models; memory grew without bound; no multi-dataset workspaces. |
| Cleaning | `clean_dataframe` ran automatically on upload: dropped duplicates, dropped columns > 60 % missing, filled missing values, coerced strings to numbers when 85 % parsed (the other 15 % silently became missing) | Data was changed without preview or consent and the original could not be recovered. |
| Machine learning | Label-encoding and `fillna(0)` on the full dataset before the split, `cross_val_score` on all rows (including the test rows), no baseline, no leakage checks, accuracy only for classification | Optimistic, leaky scores; no way to tell whether a model beat a trivial guess. |
| Predictions | Models kept in memory; prediction inputs encoded differently from training | Inconsistent or failing predictions; models lost on restart. |
| Insights / RCA | Correlation and group differences phrased as drivers; no multiple-testing correction; no effect sizes | Over-claimed causality and spurious "insights" on noise. |
| Forecasting | Single in-sample fit, no backtest, no interval calibration check | No evidence that a forecast was better than repeating the last value. |
| Errors | `HTTPException(400, str(e))` and `traceback.print_exc()` throughout | Internal exception text returned to the browser; 400 for server faults. |
| Security | No upload size/row/column limits, no ID validation on file paths, no CSV formula escaping, wildcard-ish CORS list, `.env` with a placeholder key in the archive | Resource exhaustion and spreadsheet formula injection were possible. |
| Frontend | One large dashboard page, hard-coded colours, no loading/empty/error states on many tabs, no keyboard support, no tests | Hard to follow the workflow; fragile UI. |
| Tests | None | No evidence that anything worked. |

The `.env` file in the archive contained only the placeholder `your_key_here` (checked without printing
it); it was **not** carried into the deliverable. A stray empty directory named `{backend` was removed.

## 2. Architecture (v5)

```
FastAPI app (main.py)
 ├─ core/        settings from env, typed errors, security helpers, JSON safety, request-id middleware
 ├─ services/    WorkspaceStore (parquet + JSON metadata on disk, TTL purge, LRU cache)
 │               model registry (joblib + HMAC-SHA256 signature, verified before every load)
 │               jobs (bounded thread pool, progress, cancellation)
 │               analysis (per-dataset-version cache of schema/profile/findings/charts)
 ├─ agents/      pure analysis engines, unit-tested independently of HTTP
 └─ routers/     /api/workspaces/{id}/…  + /api/jobs + /api/health,/api/config
                 + v4 endpoints kept at their old paths (session_id == workspace id)
React 18 SPA (Vite 8, React Router 7, Recharts) — one route per workflow stage, lazy-loaded
```

Key decisions:

- **Workspaces on disk.** Each upload creates a workspace containing one or more datasets. The
  original upload is stored once as parquet and never modified; cleaning steps are an ordered list
  of operations replayed onto the original (cached as `working.parquet`). This makes undo, revert,
  lineage, re-applying steps at prediction time and reproducibility straightforward.
- **Cached analyses keyed by dataset version.** Any change to the data bumps the version and
  invalidates profile, findings, charts, etc.; saved forecasts/diagnostics computed on an older
  version are marked stale in reports.
- **Long work in jobs.** Training runs in a bounded background pool with progress messages, a time
  budget and cancellation; the HTTP request returns 202 with a job id.
- **Deterministic tools first, LLM optional.** The copilot routes questions to tested analysis
  functions. If an Anthropic key is configured, the LLM may route ambiguous questions and phrase an
  explanation, but numbers in its text are checked against the computed result and the response is
  discarded if they do not match. No model output is ever executed.

## 3. Changes by phase

### Ingestion (Phase 3)
- Formats: CSV/TSV/TXT, XLSX/XLS (sheet listing and selection), JSON (array, records, `{"data": …}`), JSONL.
- Encoding detection (BOM, UTF-8, cp1252/latin-1 fallbacks), delimiter sniffing (`, ; \t |`).
- Malformed rows are detected by a `csv`-module pre-scan (pandas silently truncates or shifts them in
  some versions), skipped, counted, and listed with line numbers and values.
- Duplicate and blank headers are renamed and reported; files disguised with a wrong extension are refused.
- Limits on bytes, rows, columns and cells; empty files and header-only files give specific errors.
- Only lossless conversions happen automatically (e.g. numeric text → numbers when *every* value parses);
  dates that fail to parse are never coerced to missing.
- `POST /inspect` previews format/sheets/delimiter before ingesting.

### Profiling & EDA (Phase 4)
- Logical type inference (numeric, categorical, datetime, boolean, identifier, text, constant, empty)
  and name-based semantic hints using token matching (v4's substring matching labelled
  `discount_pct` as quantity and `sales_channel` as revenue).
- Per-column statistics with sampling disclosed above `PROFILE_SAMPLE_ROWS`; histograms with
  Freedman–Diaconis bins; IQR and robust z-score outliers; missing-value co-occurrence.
- Correlations: Pearson/Spearman with p-values and Fisher CIs, Cramér's V (bias-corrected), η²;
  Benjamini–Hochberg correction across all tested pairs.
- Target analysis for classification and regression targets.

### Cleaning (Phase 5)
- 16 operations (duplicates, empty rows, missing values, imputation with stored fitted values,
  type conversion, trimming, case, value mapping, placeholder tokens, winsorising, outlier removal,
  conditional removal/nullification, number+unit splitting, rename, drop).
- Every operation can be previewed: before/after sample, rows removed, values changed/filled/lost.
- Operations that remove data return **409 `confirmation_required`** unless `confirm: true` is sent;
  the UI requires a checkbox acknowledgement.
- "Apply safe fixes" runs only non-destructive suggestions. Undo, revert-to-step and full history.
- Data-quality rules (not null, unique, range, allowed values, regex with ReDoS screening) evaluated on demand.

### AutoML (Phase 6)
- scikit-learn `Pipeline` + `ColumnTransformer` so imputation/encoding/scaling are fitted inside each
  CV fold; rare categories grouped (`OneHotEncoder(min_frequency)`).
- Hold-out test split made **before** any model selection; cross-validation on the training split
  only; the test set is used exactly once for the selected model and the baseline.
- Baselines (mean / class prior) always trained and reported.
- Leakage checks: identifiers, columns that alone predict the target almost perfectly, relabelled
  copies of the target, exact duplicate rows across train/test.
- Metrics: RMSE/MAE/R² for regression; ROC-AUC, PR-AUC, F1, balanced accuracy and confusion matrix for
  classification; class weighting for imbalance; explicit warning when accuracy is misleading.
- Time-aware option: chronological split and `TimeSeriesSplit`.
- Candidates: Ridge/Logistic, decision tree, random forest, histogram gradient boosting, optional
  XGBoost/LightGBM; randomized search on the top two within the remaining time budget.
- Permutation importance with direction hints; residual and calibration diagnostics.
- **Fixed during benchmarking:** above `TRAINING_MAX_ROWS` the target was not re-aligned with the
  sampled rows (crash at 500k rows); random forests consumed the whole budget on 100k rows.
  Random forests now subsample per tree on large data, run after the boosting models, and are skipped
  with an explanation when a probe fit predicts they would exceed the remaining budget.
  (100k-row quick training: 232 s → 54 s; see the verification report.)

### Predictions & model lifecycle (Phase 7)
- Models are persisted with metadata (metrics, input schema, cleaning steps, library versions, data
  fingerprint) and an HMAC-SHA256 signature verified before each load. There is no endpoint that
  accepts a serialized model.
- Single predictions use a form generated from the input schema with type/range/category validation;
  unseen categories and out-of-range values produce warnings, not silent garbage.
- Batch scoring of CSV/Excel with the same cleaning steps replayed; downloads are formula-escaped.
- A warning is shown if the server's scikit-learn version differs from the training version.

### Forecasting (Phase 8)
- Automatic frequency (escalates from daily to weekly/monthly when most periods are empty), explicit
  handling of empty periods, incomplete final period detection.
- Methods: naive, seasonal naive, drift, simple/Holt/Holt-Winters exponential smoothing, ARIMA,
  linear trend (with seasonal dummies when there is enough history).
- Rolling-origin backtests that never train on the future; selection by MASE; sMAPE, RMSE and 80 %
  interval coverage reported; prediction intervals; non-negative series clipped and disclosed.

### Diagnostics / RCA (Phase 9)
- Period-over-period decomposition into mix and rate effects (exact by construction, tested),
  segment contributions, significance tests with effect sizes, Simpson's-paradox check by
  stratification, binary outcomes supported.
- Every result is labelled *observation*, *association*, *potential driver* or *diagnostic
  hypothesis*, with confounders, alternative explanations and how to validate.

### Multi-file (Phase 10)
- Schema comparison, join-key candidates ranked by value containment and coverage (penalising
  auto-increment IDs that overlap by accident), join preview with cardinality and row-explosion
  checks, joins and stacking that create new datasets in the workspace.

### Copilot (Phase 11)
- 17 deterministic tools (overview, insights, missing values, compare groups, column stats, outliers, model
  training proposal, model explanation, forecast, management summary, next steps, …) with evidence
  tables/charts; clarifying questions when a question is ambiguous; follow-ups use context.
- Training is proposed with a button, never started by the copilot itself.

### Reports (Phase 12)
- Executive summary built from computed results. Report builder with selectable sections, findings,
  charts and model; on-screen preview identical to the export; PDF (ReportLab + Matplotlib charts,
  DejaVu fonts, escaped markup), Markdown and JSON.

### Backend quality & security (Phases 13–14)
- Typed errors with a uniform JSON shape and request IDs; internal errors are logged, not returned.
- ID normalisation and `safe_child` path checks; filename sanitisation and RFC 5987 headers;
  CSV formula escaping; regex screening; body size limits; CORS restricted to configured origins with
  explicit methods; `nosniff`, `no-referrer`, `no-store` headers; secret key file with mode 600;
  workspace TTL purge; optional disabling of workspace listing. Details in
  [SECURITY_AND_PRIVACY.md](SECURITY_AND_PRIVACY.md).

### Frontend (Phases 2, 15)
- New app shell: collapsible sidebar grouped by workflow stage, breadcrumbs, workflow progress rail,
  command palette (Ctrl/⌘ K), notifications, help panel, settings (theme, sidebar), light/dark themes
  from CSS tokens, validated chart palette.
- Pages: Home (upload with client-side validation, recent workspaces, sample data), Overview, Data
  (server-side paging/sort/search/filters, schema, validation report), Profile, Clean, Sources & joins,
  Insights, Visualize (auto dashboard + chart builder), Diagnose, Forecast, Train models, Predict,
  Copilot, Report.
- Loading, empty and error states everywhere; errors show the request ID; keyboard-reachable
  controls, labelled inputs, `aria-live` regions, focus rings, reduced-motion support.
- Dependencies upgraded to clear all `npm audit` advisories (Vite 8, Vitest 5, React Router 7).

### Differentiators (Phase 20)
Persistent workspaces with activity log; reproducibility block (data fingerprint, steps, versions,
seeds) in models and reports; data-quality rules; prioritised insights with evidence and claim types;
experiment comparison and CSV export; report customisation; guided next steps on the Overview page.

## 4. Compatibility

The v4 endpoints (`/api/upload`, `/api/analyze/*`, `/api/chat`, `/api/report`, `/api/predict/*`,
`/api/features/*`, `/api/advanced/*`) still exist and are covered by tests; their `session_id` is now a
workspace id. The v4 behaviour of cleaning data automatically on upload was intentionally **not**
kept: the legacy upload returns the same response shape, but data is only changed by explicit steps.

## 5. Honest limitations

See the verification report's "Known issues" and the roadmap. The most important: no authentication
(by design for a local tool — do not expose it publicly without a proxy), single-process job runner,
the LLM path was tested with a mocked API only, and the Docker image was not built in this environment.
