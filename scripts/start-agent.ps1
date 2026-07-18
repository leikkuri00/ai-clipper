# Start full agent: ensure LLM server is running and then run agent core
.\.venv\Scripts\Activate.ps1
py -m src.manager start
Start-Sleep -Seconds 1
py -m src.agent_core --index
Write-Host "Agent started. Use src.agent_core to interact (e.g., --plan)"