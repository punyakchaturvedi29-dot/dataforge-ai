# Setup guide

## Requirements

| Component | Version | Notes |
|---|---|---|
| Python | 3.10 – 3.13 | Verified on 3.13. 64-bit required for pandas/numpy wheels. |
| Node.js | 20.19+ or 22.12+ | Required by Vite 8. Verified on 22.22. |
| RAM | 2 GB minimum, 8 GB for files near the 100 MB / 2M-row defaults | Datasets are held in memory while analysed. |
| Disk | ~1.5 GB for dependencies + your data | Parquet copies are typically 3–10× smaller than CSV. |

## Development setup

```bash
git clone <your copy> dataforge && cd dataforge   # or unzip the archive
./start.sh                                        # macOS/Linux; Windows: start.bat
```

`start.sh` creates `backend/venv`, installs `backend/requirements.txt`, copies `.env.example` to
`.env` if missing, installs frontend packages, then runs the API on :8000 and Vite on :5173
(which proxies `/api`). Set `DATAFORGE_ML_EXTRAS=1` to also install XGBoost and LightGBM, or
`PYTHON=/path/to/python3.12` to pick an interpreter.

Manual steps are in the [README](../README.md#manual-setup).

### Dependency files

| File | Use |
|---|---|
| `backend/requirements.txt` | Runtime dependencies with tested version ranges. |
| `backend/requirements-ml-extras.txt` | Adds XGBoost and LightGBM (optional; AutoML skips them when missing). |
| `backend/requirements-dev.txt` | Adds pytest, pytest-cov, pip-audit. |
| `backend/requirements.lock.txt` | Exact versions of the verified environment (Python 3.13). |
| `frontend/package-lock.json` | Exact frontend versions; use `npm ci` for reproducible installs. |

## Configuration

Copy `backend/.env.example` to `backend/.env` and uncomment what you need; environment variables
override the file. All settings and defaults are listed in `.env.example`; the important ones are in
the README. Restart the backend after changing them.

To enable the optional LLM features set `ANTHROPIC_API_KEY` (and optionally `ANTHROPIC_MODEL`).
The app checks for placeholder values and treats them as "not configured". With no key, every
feature still works; the copilot answers from its deterministic tools.

## Production-style deployment

1. Build the frontend: `cd frontend && npm ci && npm run build`.
2. Run the backend with a persistent data directory and a fixed secret:

   ```bash
   cd backend
   export DATAFORGE_DATA_DIR=/var/lib/dataforge
   export DATAFORGE_SECRET_KEY="$(python -c 'import secrets;print(secrets.token_hex(32))')"  # store it safely
   export WORKSPACE_LISTING=false
   uvicorn main:app --host 127.0.0.1 --port 8000 --workers 1 --proxy-headers
   ```

   The backend serves `frontend/dist` at `/` (override with `FRONTEND_DIST`), so one process serves
   both. Use **one worker**: jobs and caches are in-process.
3. Put a reverse proxy in front (TLS + authentication), e.g. Caddy or nginx with OAuth2 Proxy, and set
   its upload limit at least to `MAX_UPLOAD_MB`. CORS settings are not needed when the app and API
   share an origin.
4. Back up `DATAFORGE_DATA_DIR`. Deleting it removes all workspaces and models.

### Docker

```bash
docker build -t dataforge .
docker run -d -p 127.0.0.1:8000:8000 -v dataforge-data:/data \
  -e DATAFORGE_SECRET_KEY=... -e WORKSPACE_LISTING=false dataforge
```

The image builds the frontend in a Node stage and runs as a non-root user with `/data` as the data
directory. (The Dockerfile was written for this release but could not be built in the verification
environment, which has no Docker daemon — build it once locally before relying on it.)

## Running the tests

```bash
cd backend && source venv/bin/activate
pip install -r requirements-dev.txt
pytest -q                          # unit + API tests (about 1 minute)
pytest --cov=. --cov-report=term   # with coverage

cd ../frontend
npm test                           # Vitest unit/component tests
npx playwright install chromium    # once
npm run test:e2e                   # starts its own backend on :8765 with a throwaway data dir
```

Playwright environment variables: `PYTHON` (interpreter for the backend, default `python`),
`E2E_BASE_URL` (test an already-running server), `CHROME_PATH` (use an existing Chromium),
`E2E_SHOTS=dir` (save screenshots of each stage).

Benchmarks: `python backend/scripts/benchmark.py --rows 10000 100000`.

## Troubleshooting

| Problem | Cause / fix |
|---|---|
| `pip` tries to compile numpy/pandas | No wheel for your Python version or architecture; use 64-bit CPython 3.11–3.13. |
| `Cannot reach the DataForge server` in the browser | The backend is not running on :8000, or you changed its port without setting `DATAFORGE_API` for Vite. |
| `413 payload_too_large` | Raise `MAX_UPLOAD_MB` (and your proxy's limit). |
| Training says models were skipped for time | Raise `TRAINING_TIME_BUDGET_S`, lower `TRAINING_MAX_ROWS`, or use fewer features. |
| `integrity_check_failed` when predicting | The model file or `DATAFORGE_SECRET_KEY` changed. Retrain the model. |
| Version warning when predicting | The model was trained with a different scikit-learn; retrain. |
| PDF has missing glyphs | Matplotlib's bundled DejaVu fonts are used; reinstall matplotlib if they are missing. |
| Workspaces disappear | They expire after `WORKSPACE_TTL_DAYS` without changes, or `DATAFORGE_DATA_DIR` changed between runs. |
