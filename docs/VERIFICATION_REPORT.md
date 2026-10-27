# Verification report — DataForge AI v5

Every result below was produced by actually running the command. Nothing marked "not run" was run.
Environment: Linux x86-64 container, 2 vCPU, 7 GB RAM, Python 3.13.17, Node 22.22.0, Chromium (Playwright 1.64).

## 1. Summary

| Check | Result |
|---|---|
| Backend tests, stack A (pandas 3.1, numpy 2.5, scikit-learn 1.9, statsmodels 0.15, + XGBoost/LightGBM) | **133 passed** |
| Backend tests, stack B (pandas 2.2.3, numpy 2.1, scikit-learn 1.5.2, statsmodels 0.14.4, no XGBoost/LightGBM) | **133 passed** |
| Backend line coverage (excluding tests) | **82 %** |
| Frontend unit/component tests (Vitest 5) | **23 passed** |
| End-to-end browser tests (Playwright, Chromium) | **4 passed** — against the dev setup and again against the fresh single-process build |
| Production build (`vite build`) | succeeds |
| `pip-audit -r requirements.lock.txt` | no known vulnerabilities |
| `npm audit` | 0 vulnerabilities (after upgrades, see §5) |
| Fresh unzip → install → test → start | succeeded (§2) |
| Docker image build | **not run** — no Docker daemon in this environment |
| Live Anthropic API calls | **not run** — tested with a mocked HTTP layer only |
| Windows `.bat` scripts | **not run** — no Windows machine available; reviewed by hand |

## 2. Fresh-install verification

Performed on the packaged archive, unzipped into an empty directory:

```bash
cd backend
python3 -m venv venv
./venv/bin/pip install -r requirements-dev.txt         # resolved fastapi 0.162, pandas 3.1.0, sklearn 1.9.1, xgboost 3.4.1, lightgbm 4.7.0
./venv/bin/python -m pytest -q --cov=. --cov-report=term
#   133 passed, 1 warning in 83.04s
cd ../frontend
npm ci                                                  # found 0 vulnerabilities
npm test                                                # Test Files 4 passed (4) / Tests 23 passed (23)
npm run build                                           # ✓ built
cd ../backend && cp .env.example .env
uvicorn main:app --port 8800                            # single process serving API + frontend/dist
curl /api/health  -> {"status":"ok", "checks":{"data_dir_writable":true, ... "llm_configured":false}}
curl /api/config  -> limits only; no secrets
curl /            -> index.html;  /w/<id>/profile (deep link) -> 200
curl /api/nonexistent -> 404 JSON;  /assets/..%2F..%2Fbackend%2F.env -> 404
cd ../frontend
E2E_BASE_URL=http://127.0.0.1:8800 npx playwright test  # 4 passed (25.4s)
```

The server log contained no errors after the E2E run. Screenshots of every stage were reviewed
(Overview, Data, Profile, Clean, Visualize, Model, Predict, Forecast, Diagnose, Copilot, Report,
dark theme); they render correctly.

## 3. Backend test suite

`backend/tests/` — 133 tests in 8 files:

| File | Tests | What it proves |
|---|---:|---|
| `test_ingest.py` | 26 | delimiters (`, ; \t \|`), encodings (UTF-8/BOM/cp1252/latin-1), malformed rows counted with line numbers, duplicate/blank headers, empty/header-only files, disguised files, Excel sheets, JSON variants, size/row limits, wide data, no silent date coercion, type and token-based semantic inference |
| `test_profiling.py` | 8 | statistics equal numpy; Pearson/Spearman p-values equal scipy; Benjamini–Hochberg equals a reference implementation; effect sizes; Cramér's V for independent vs dependent data; target analysis |
| `test_cleaning.py` | 12 | issue detection, conversions and impact counts, stored imputation values reused, **409 until confirmed** for destructive steps, undo/revert/history, deterministic replay, apply-safe only non-destructive, parameter validation, outlier ops, cache invalidation, quality rules |
| `test_insights_charts.py` | 10 | a planted effect is found; **no findings on pure noise**; identifiers never used as measures; misleading charts rejected; sampling disclosed; gaps not drawn as zero |
| `test_ml.py` | 17 | known relationship learned (R² > 0.9, RMSE < baseline/3); imbalanced classification reports balanced metrics; leakage columns removed; chronological split; train/test duplicates flagged; tampered model refused; cleaning replayed at prediction; job cancellation; clustering/anomalies; **row sampling keeps the target aligned** (new) |
| `test_forecast_rca_multifile.py` | 13 | seasonal series beats naive; backtests never see the future; short series and gaps; Simpson's paradox detected; mix/rate decomposition exact; claim labels; join candidates by value evidence; join preview counts; join/stack API |
| `test_copilot.py` | 20 | all 11 questions from the brief route to the right tool; answers equal computed values; clarification; follow-ups; training proposed not executed; **code never executed**; LLM routing and the number-grounding guard (mocked); graceful LLM failure |
| `test_reports_security_legacy.py` | 27 | PDF/Markdown/JSON reports with real metrics and markup-safe names; path traversal IDs rejected; filename sanitisation; formula escaping in downloads; ReDoS screening; body limits; no internal error leakage; CORS; security headers; deletion, TTL purge, listing switch; secret file mode 600; health/config expose no secrets; v4 endpoints |

The two `test_row_sampling_keeps_target_aligned` cases were added after benchmarking exposed the bug.
They were confirmed to **fail** with the fix removed and pass with it. Both dependency stacks were
re-run on the final code (stack A 133 passed in 83 s incl. coverage; stack B 133 passed in 44 s).

Uncovered areas worth noting: `agents/quality_rules.py` 60 %, `core/jsonsafe.py` 62 %,
`routers/workspaces.py` 68 % (XLSX download and some listing paths), `store.py` 0 % (v4 import shim, unused).

## 4. Frontend tests

- **Vitest** (`frontend/tests/unit`, 23 tests): formatting helpers (including the regression where
  "Jack Taylor" was rendered "Jack aylor"), client-side upload validation, Markdown renderer never
  renders HTML/script, DataTable missing values and empty state, ErrorState shows request ID and retry,
  claim labels avoid causal wording, finding details expand, the cleaning dialog **disables "Apply"
  for destructive steps until the checkbox is ticked** and sends `confirm: true`, preview errors shown.
- **Playwright** (`frontend/e2e/workflow.spec.js`, 4 tests): unsupported/empty file messages;
  the full workflow (upload sample → validation report → apply a cleaning fix → profile → insights →
  train a model → single prediction → forecast → diagnostics → copilot question → report preview);
  command palette and Escape handling; dark theme tokens.

## 5. Dependency audit

- `pip-audit -r backend/requirements.lock.txt`: *No known vulnerabilities found*.
- `npm audit` initially reported 8 advisories (2 critical, 1 high, 5 moderate). Only `react-router`
  (moderate, open redirect in `<Link>` with backslashes; SSR hydration issue not applicable) shipped to
  browsers; the rest were build/test tooling (Vite 5/esbuild, Vitest 2/tinypool). Upgraded to
  React Router 7.18, Vite 8.3, @vitejs/plugin-react 6.1, Vitest 5.0, jsdom 27. `manualChunks` was
  rewritten in function form for Vite 8's bundler. Afterwards: **0 vulnerabilities**; build, unit and
  E2E tests re-run and passing.

## 6. Performance (measured)

`python backend/scripts/benchmark.py --rows 10000 100000 500000 1000000` — synthetic 12-column
retail data, in-process ASGI client (no network), 2 vCPU. Seconds per request:

| Step | 10k rows (0.7 MB) | 100k (7.1 MB) | 500k (35.8 MB) | 1M (71.7 MB) |
|---|---:|---:|---:|---:|
| Upload + ingest + type inference | 0.09 | 0.24 | 1.05 | 2.17 |
| Data page (sorted) | 0.04 | 0.07 | 0.32 | 0.69 |
| Profile (all columns) | 0.27 | 0.61 | 2.39 | 5.32 |
| Insights (ranked findings) | 0.26 | 1.06 | 5.02 | 10.28 |
| Auto dashboard | 0.17 | 0.64 | 3.22 | 7.14 |
| Cleaning suggestions* | 0.00 | 0.00 | 0.00 | 0.00 |
| Forecast (auto) | 1.08 | 1.06 | 0.99 | 1.20 |
| AutoML quick (regression) | 24.5 | 54.5 | 76.9 | 77.9 |

\* already computed by the profile request and served from the per-version cache.
Above 200k rows AutoML trains on a sample (`TRAINING_MAX_ROWS`), which is why 500k and 1M take similar time.

**Problems the benchmark found and that were fixed:**
1. At 500k rows training crashed (`Length of values (500000) does not match length of index (200000)`):
   the target was not re-sampled with the features. Fixed and covered by tests.
2. At 100k rows a quick training run took **232 s** against a 120 s budget, because one random-forest
   cross-validation took 160 s and the budget is only checked between models. Random forests now
   subsample rows per tree on large data, run after the boosting models, and are skipped with an
   explanation if a probe fit predicts they would exceed the remaining budget → **54 s**.

## 7. Known issues and limitations

- No authentication or per-user isolation (documented; see SECURITY_AND_PRIVACY.md and ROADMAP P0).
- One uvicorn worker only; background jobs are in-process and lost on restart.
- The time budget is enforced between models; a single slow model can still overrun it on unusual
  data (probe estimates exist for tree/forest models only).
- Insights on 1M rows take ~10 s on 2 vCPU on first request (cached afterwards).
- Unknown `/api/...` paths return FastAPI's default `{"detail": "Not found"}` rather than the
  uniform error shape.
- Toast notifications can briefly overlap page content on narrow screens.
- The LLM integration was verified only with mocked responses; the default model name
  (`ANTHROPIC_MODEL`) should be checked against your account.
- The Dockerfile and Windows batch files were not executed.
- Google Fonts are requested by the page; the sandbox blocked them, so screenshots use the fallback
  system font. Offline installs fall back the same way.

## 8. Not run

Docker build/run; Windows scripts; macOS; Python 3.10–3.12 (only 3.13 tested); live LLM calls;
load testing with concurrent users; automated accessibility (axe) scan; browsers other than Chromium.
