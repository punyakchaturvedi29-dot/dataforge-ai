#!/usr/bin/env bash
# DataForge AI — development start (macOS / Linux).
# Creates backend/venv and installs dependencies on first run, then starts
# the API on :8000 and the web app on :5173. Ctrl+C stops both.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")" && pwd)"
PY="${PYTHON:-python3}"

command -v "$PY" >/dev/null || { echo "Python 3.10+ not found (set PYTHON=/path/to/python)"; exit 1; }
command -v npm  >/dev/null || { echo "Node.js 20.19+ / 22.12+ and npm not found (https://nodejs.org)"; exit 1; }
"$PY" -c 'import sys; sys.exit(0 if sys.version_info >= (3, 10) else 1)' || { echo "Python 3.10 or newer is required"; exit 1; }

echo "→ Backend dependencies"
cd "$ROOT/backend"
[ -d venv ] || "$PY" -m venv venv
# shellcheck disable=SC1091
source venv/bin/activate
pip install -q --disable-pip-version-check -r requirements.txt
if [ "${DATAFORGE_ML_EXTRAS:-0}" = "1" ]; then pip install -q -r requirements-ml-extras.txt; fi
[ -f .env ] || cp .env.example .env

echo "→ Frontend dependencies"
cd "$ROOT/frontend"
[ -d node_modules ] || npm install --no-audit --no-fund

cd "$ROOT/backend"
uvicorn main:app --host 127.0.0.1 --port 8000 --log-level info &
BACK=$!
cd "$ROOT/frontend"
npm run dev -- --host 127.0.0.1 &
FRONT=$!
trap 'echo; echo "Stopping…"; kill $BACK $FRONT 2>/dev/null || true; exit 0' INT TERM

echo
echo "  App      http://localhost:5173"
echo "  API docs http://localhost:8000/docs"
echo "  Ctrl+C to stop"
wait
