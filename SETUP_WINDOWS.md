# DataForge AI on Windows

## 1. Install the prerequisites (once)

1. **Python 3.10 or newer** (3.12 or 3.13 recommended) from <https://www.python.org/downloads/>.
   On the first installer screen tick **"Add python.exe to PATH"**.
2. **Node.js LTS** (20.19+ or 22.12+) from <https://nodejs.org/>.

Open a new *Command Prompt* and check:

```bat
python --version
node --version
```

## 2. Start the app

Double-click **`start.bat`** in the project folder. It opens two windows:

- **DataForge AI - Backend** – creates `backend\venv`, installs Python packages (a few minutes the
  first time) and starts the API on <http://localhost:8000>.
- **DataForge AI - Frontend** – installs npm packages (first time only) and starts the web app on
  <http://localhost:5173>, which opens in your browser.

Keep both windows open while you work; close them to stop the servers.
You can also run `start_backend.bat` and `start_frontend.bat` separately.

Optional: to include XGBoost and LightGBM in AutoML, run once in a Command Prompt
`set DATAFORGE_ML_EXTRAS=1` before `start_backend.bat`, or install them manually:

```bat
cd backend
venv\Scripts\activate
pip install -r requirements-ml-extras.txt
```

## 3. Optional settings

`start_backend.bat` copies `backend\.env.example` to `backend\.env` on first run. Edit `.env` in
Notepad to change limits or add an `ANTHROPIC_API_KEY` for the optional AI explanations. Never
share or commit this file.

## Troubleshooting

| Symptom | Fix |
|---|---|
| `'python' is not recognized` | Reinstall Python with "Add to PATH", or use the `py` launcher: edit the .bat files and replace `python` with `py -3`. |
| `pip install` fails building numpy/pandas | You are probably on a very new or 32-bit Python. Install 64-bit Python 3.12 or 3.13. |
| The page says it cannot reach the server | The backend window must show `Uvicorn running on http://127.0.0.1:8000`. Check it for errors. |
| Port 8000 or 5173 already in use | Close the other program, or start the backend with `uvicorn main:app --port 8001` and set `set DATAFORGE_API=http://127.0.0.1:8001` before `npm run dev`. |
| Execution policy errors in PowerShell | Use Command Prompt (`cmd.exe`) for the .bat files, or run `venv\Scripts\activate.bat`. |
| Antivirus quarantines files in `venv` | Add an exclusion for the project folder, delete `backend\venv`, and start again. |
