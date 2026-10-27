# DataForge AI

A self-hosted analytics workbench: upload a CSV, Excel or JSON file and work through
**Upload → Validate → Profile → Clean → Analyze → Model → Visualize → Export** with every
number computed from your data, every change previewed and reversible, and every claim
labelled for what it is (observation, association, hypothesis).

- **Ingestion that reports instead of guessing** – encoding, delimiter and Excel sheet detection,
  malformed rows listed by line number, duplicate/blank headers renamed and reported, size limits.
- **Profiling & insights** – per-column statistics, distributions, outliers, missing-value
  patterns, correlations (Pearson/Spearman/Cramér's V/η²) with multiple-testing correction, and
  ranked findings that show their evidence and explain what they do *not* prove.
- **Auditable cleaning** – suggestions with a preview of affected rows, explicit confirmation for
  anything that removes data, undo/revert, lineage, and the original upload kept unchanged.
  The same steps are replayed on new data at prediction time.
- **AutoML with guard-rails** – leakage checks (identifiers, near-perfect predictors, post-outcome
  columns), cross-validation on a training split, a single evaluation on a held-out test set,
  baselines, imbalance-aware metrics, permutation importance, time-aware splits, and a time budget.
- **Predictions** – single-record forms generated from the model's input schema and batch scoring
  of CSV/Excel files; saved models are HMAC-signed and never loaded from user uploads.
- **Forecasting** – method comparison by rolling-origin backtests (MASE, sMAPE, interval coverage),
  automatic period/frequency handling, approximate prediction intervals.
- **Diagnostics** – "why did the metric change?" with mix/rate decomposition, segment contributions,
  significance tests and Simpson's-paradox checks; results are worded as hypotheses.
- **Multi-file** – compare schemas, ranked join-key candidates with coverage, validated joins
  (cardinality and row-explosion checks) and stacking.
- **Copilot** – plain-language questions answered by deterministic, tested analysis tools with
  evidence tables and charts; an LLM is optional, never executes code, and its numbers are checked.
- **Reports** – executive summary, customisable PDF / Markdown / JSON reports with methodology,
  limitations and reproducibility details; CSV exports are protected against formula injection.

## Quick start

Requirements: **Python 3.10+** (tested on 3.13) and **Node.js 20.19+ or 22.12+**.

| | |
|---|---|
| macOS / Linux | `./start.sh` |
| Windows | double-click `start.bat` (see [SETUP_WINDOWS.md](SETUP_WINDOWS.md)) |

Then open <http://localhost:5173> and drop a file, or click **Try the sample dataset**.

### Manual setup

```bash
# backend
cd backend
python -m venv venv && source venv/bin/activate      # Windows: venv\Scripts\activate
pip install -r requirements.txt                       # + requirements-ml-extras.txt for XGBoost/LightGBM
cp .env.example .env                                  # optional settings
uvicorn main:app --port 8000

# frontend (second terminal)
cd frontend
npm install
npm run dev                                           # http://localhost:5173, proxies /api to :8000
```

### Single-process mode / Docker

```bash
cd frontend && npm run build          # produces frontend/dist
cd ../backend && uvicorn main:app --port 8000     # serves the app and the API at http://localhost:8000
```

```bash
docker build -t dataforge .
docker run -p 8000:8000 -v dataforge-data:/data dataforge
```

Run **one** worker process: background jobs and caches live in the process (see the roadmap).

## Configuration

All settings are environment variables (or `backend/.env`); every one is optional.
See [`backend/.env.example`](backend/.env.example) for the full list. The important ones:

| Variable | Default | Purpose |
|---|---|---|
| `ANTHROPIC_API_KEY` | – | Enables optional LLM routing/explanations. Everything works without it. |
| `DATAFORGE_DATA_DIR` | `backend/data` | Workspaces, datasets (parquet), signed models. |
| `WORKSPACE_TTL_DAYS` | 14 | Idle workspaces are deleted automatically. |
| `MAX_UPLOAD_MB` / `MAX_ROWS` / `MAX_COLUMNS` | 100 / 2M / 1000 | Upload limits. |
| `TRAINING_MAX_ROWS` / `TRAINING_TIME_BUDGET_S` | 200k / 120 | AutoML resource limits. |
| `DATAFORGE_SECRET_KEY` | generated | Signs model files; set it when several processes share the data dir. |
| `CORS_ORIGINS` | localhost:5173 | Only needed when the frontend is served from another origin. |

## Tests

```bash
cd backend && pip install -r requirements-dev.txt && pytest -q          # 133 tests
cd frontend && npm test                                                  # 23 unit tests (Vitest)
cd frontend && npx playwright install chromium && npm run test:e2e       # 4 browser tests; starts the backend itself
python backend/scripts/benchmark.py --rows 100000                        # timings on synthetic data
```

Results of the last verification run are in [docs/VERIFICATION_REPORT.md](docs/VERIFICATION_REPORT.md).

## Project layout

```
backend/
  main.py            app factory: middleware, CORS, error handling, routers, SPA serving
  core/              config, errors, security helpers, JSON safety, middleware
  services/          workspace store (parquet + metadata), model registry (signed), jobs, cached analyses
  agents/            analysis engines: ingest, schema, cleaner, profiling, insights, charts, automl,
                     inference, forecasting, rca, multifile, copilot, report_builder, ...
  routers/           /api/workspaces/... (+ legacy v4 routes kept for compatibility)
  tests/             pytest suite
  scripts/benchmark.py
frontend/
  src/pages/         one page per workflow stage
  src/components/    app shell, UI kit, charts, findings
  tests/unit/        Vitest + Testing Library
  e2e/               Playwright workflow tests
docs/                enhancement report, verification report, setup guide, roadmap, security & privacy
```

The interactive API reference is at <http://localhost:8000/docs>.

## Documentation

- [docs/SETUP_GUIDE.md](docs/SETUP_GUIDE.md) – install, configure, deploy, troubleshoot
- [docs/ENHANCEMENT_REPORT.md](docs/ENHANCEMENT_REPORT.md) – what changed from v4 and why
- [docs/VERIFICATION_REPORT.md](docs/VERIFICATION_REPORT.md) – what was tested, how, and what was not
- [docs/SECURITY_AND_PRIVACY.md](docs/SECURITY_AND_PRIVACY.md) – threat model, controls, data retention
- [docs/ROADMAP.md](docs/ROADMAP.md) – prioritised next steps

## Limitations

DataForge has **no user accounts**. Anyone who can reach the server can use it, and workspace
links act as bearer tokens. Run it on your own machine or behind an authenticating reverse proxy;
see the security document before exposing it to a network.
