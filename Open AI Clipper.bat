@echo off
setlocal
cd /d "%~dp0"
title AI Clipper

py -3 -m streamlit --version >nul 2>&1
if errorlevel 1 (
    echo Installing dependencies for the first time...
    py -3 -m pip install -r "%~dp0ai_clipper\requirements.txt"
    if errorlevel 1 (
        echo.
        echo Could not install dependencies. Check your internet connection and try again.
        pause
        exit /b 1
    )
)

echo Starting AI Clipper in your browser...
py -3 -m streamlit run ai_clipper\web_app.py --server.address 127.0.0.1 --server.port 8501
