# Start a recommended local llama.cpp server command (advisory)
param(
    [string]$ModelPath = "models",
    [int]$Nctx = 8192
)

Write-Host "LLM runtime starter (advisory). This script prints a recommended command to start a local runtime."
Write-Host "Ensure you have an appropriate llama.cpp-compatible binary on PATH."

py -c "from src.runtime import recommendation_report, assemble_llama_command, recommend_backend; import json; r=recommendation_report(); print('Recommendation:', json.dumps(r, indent=2));" 

Write-Host "To actually run a binary, run the assembled command below (if you have a compatible binary):"
py - <<'PY'
from pathlib import Path
from src.runtime import assemble_llama_command, recommend_backend
model = Path($env:MODEL_PATH) if 'MODEL_PATH' in $env else Path(r'.\models')
try:
    cmd = assemble_llama_command(model, backend=recommend_backend())
    print(' '.join(cmd))
except Exception as e:
    print('Could not assemble runtime command:', e)
PY
