@echo off
setlocal
cd /d "%~dp0"
title AI Clipper

py -3 -m streamlit --version >nul 2>&1
if errorlevel 1 (
    echo Installing the web interface for the first time...
    py -3 -m pip install streamlit
    if errorlevel 1 (
        echo.
        echo Could not install Streamlit. Check your internet connection and try again.
        pause
        exit /b 1
    )
)

echo Starting AI Clipper in your browser...
py -3 -m streamlit run ai_clipper\web_app.py --server.address 127.0.0.1 --server.port 8501
