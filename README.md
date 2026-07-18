# Local AI Coding Agent — clippingtool33-integration

This repository scaffolds a production-grade, fully-local AI coding agent environment.

Components included (initial scaffold):

- installer scripts (PowerShell)
- basic Python agent core for hardware detection and CLI
- config templates
- VS Code recommendations
- Obsidian integration folder layout

This is the first iterative deliverable: a runnable scaffold and agent entrypoint that detects hardware and recommends a local model/runtime.

Run the quick check:

```powershell
# From workspace root (Windows PowerShell)
py -3 -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
py -m src.agent --help
```
