@echo off
setlocal
title DataForge AI - Backend
cd /d "%~dp0backend"
where python >nul 2>&1 || (echo Python 3.10+ not found. Install from https://www.python.org/downloads/ and tick "Add Python to PATH". & pause & exit /b 1)
if not exist venv (
  echo Creating virtual environment...
  python -m venv venv || (echo Could not create the virtual environment. & pause & exit /b 1)
)
call venv\Scripts\activate.bat
echo Installing dependencies (first run takes a few minutes)...
pip install -r requirements.txt --quiet --disable-pip-version-check || (echo pip install failed - see the message above. & pause & exit /b 1)
if "%DATAFORGE_ML_EXTRAS%"=="1" pip install -r requirements-ml-extras.txt --quiet
if not exist .env copy .env.example .env >nul
echo.
echo  Backend:  http://localhost:8000
echo  API docs: http://localhost:8000/docs
echo  Press CTRL+C to stop
echo.
uvicorn main:app --host 127.0.0.1 --port 8000
