@echo off
REM ============================================================
REM  AI Clipper - one-click launcher
REM  Sets up Python + all dependencies (once) and opens the app.
REM ============================================================
setlocal enableextensions
cd /d "%~dp0"
title AI Clipper

set "VENV=%~dp0.venv"
set "REQ=%~dp0ai_clipper\requirements.txt"
set "STAMP=%VENV%\.deps_ok"

echo(
echo ==== AI Clipper ====
echo(

REM --- 1. Find Python -------------------------------------------------
set "PY="
py -3 --version >nul 2>&1 && set "PY=py -3"
if not defined PY (
    python --version >nul 2>&1 && set "PY=python"
)
if not defined PY (
    echo [X] Python 3 was not found.
    echo     Please install Python 3.10+ from https://www.python.org/downloads/
    echo     IMPORTANT: tick "Add Python to PATH" during install, then re-run this file.
    start "" https://www.python.org/downloads/
    pause
    exit /b 1
)
echo [OK] Using Python: %PY%

REM --- 2. Create virtual environment (first run only) ----------------
if not exist "%VENV%\Scripts\python.exe" (
    echo [..] Creating virtual environment ^(first run^)...
    %PY% -m venv "%VENV%"
    if errorlevel 1 (
        echo [X] Could not create the virtual environment.
        pause
        exit /b 1
    )
)
set "VPY=%VENV%\Scripts\python.exe"

REM --- 3. Install dependencies (first run / after updates) -----------
if not exist "%STAMP%" (
    echo [..] Installing dependencies. This can take a few minutes the first time...
    "%VPY%" -m pip install --upgrade pip
    "%VPY%" -m pip install -r "%REQ%"
    if errorlevel 1 (
        echo [X] Could not install dependencies. Check your internet connection and re-run.
        pause
        exit /b 1
    )
    echo ok> "%STAMP%"
    echo [OK] Dependencies installed.
) else (
    echo [OK] Dependencies already installed.
)

REM --- 4. Check FFmpeg (required for cutting/audio) ------------------
where ffmpeg >nul 2>&1
if errorlevel 1 (
    echo [..] FFmpeg not found on PATH. Trying to install it with winget...
    winget install --id Gyan.FFmpeg -e --accept-source-agreements --accept-package-agreements
    where ffmpeg >nul 2>&1
    if errorlevel 1 (
        echo [!] FFmpeg still not detected. If clips fail, install it from:
        echo     https://www.gyan.dev/ffmpeg/builds/  ^(add its \bin folder to PATH^)
        echo     Continuing anyway...
    )
) else (
    echo [OK] FFmpeg found.
)

REM --- 5. Launch the app --------------------------------------------
echo(
echo [OK] Starting AI Clipper in your browser...
echo     ^(Leave this window open while you use the app. Close it to stop.^)
echo(
"%VPY%" -m streamlit run "%~dp0ai_clipper\web_app.py" --server.address 127.0.0.1 --server.port 8501

pause
