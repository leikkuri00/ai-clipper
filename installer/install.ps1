# Install script for Local AI Coding Agent (Windows)
# Creates a virtualenv and installs Python dependencies.

$ErrorActionPreference = 'Stop'

Write-Host "Creating virtual environment .venv..."
py -3 -m venv .venv
Write-Host "Activating virtual environment and installing requirements..."
.\.venv\Scripts\Activate.ps1
pip install --upgrade pip
if (Test-Path requirements.txt) {
    pip install -r requirements.txt
} else {
    Write-Warning "requirements.txt not found."
}

Write-Host "Install complete. Run 'start-ai.ps1' in the scripts folder to start the agent."