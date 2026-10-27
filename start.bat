@echo off
setlocal
title DataForge AI
echo Starting DataForge AI: two windows will open (backend and frontend).
start "DataForge AI - Backend" cmd /k ""%~dp0start_backend.bat""
timeout /t 5 /nobreak >nul
start "DataForge AI - Frontend" cmd /k ""%~dp0start_frontend.bat""
timeout /t 6 /nobreak >nul
start "" "http://localhost:5173"
echo Close the two windows to stop the servers.
