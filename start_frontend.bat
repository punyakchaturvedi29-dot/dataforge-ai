@echo off
setlocal
title DataForge AI - Frontend
cd /d "%~dp0frontend"
where npm >nul 2>&1 || (echo Node.js 20.19+ or 22.12+ not found. Install the LTS from https://nodejs.org/ & pause & exit /b 1)
if not exist node_modules (
  echo Installing npm packages (first run only)...
  call npm install --no-audit --no-fund || (echo npm install failed - see the message above. & pause & exit /b 1)
)
echo.
echo  App: http://localhost:5173   (the backend must be running on port 8000)
echo.
call npm run dev -- --host 127.0.0.1
