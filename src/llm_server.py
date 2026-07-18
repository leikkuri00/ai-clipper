"""Minimal local LLM HTTP server adapter.

Provides a tiny FastAPI server that exposes runtime recommendation and a
simple `/generate` endpoint which attempts to invoke a local llama.cpp-style
binary if present. This is intentionally conservative: when no native runtime
is available the server returns a helpful message and the recommendation.
"""

from __future__ import annotations

from pathlib import Path
import json
import subprocess
import shlex
from typing import Any

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from src.runtime import recommendation_report, assemble_llama_command, find_llama_cpp_binary, recommend_backend

app = FastAPI(title="Local LLM Adapter")


class GenerateRequest(BaseModel):
    prompt: str
    max_tokens: int = 256


class GenerateResponse(BaseModel):
    output: str
    metadata: dict[str, Any]


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/recommend")
def recommend():
    return recommendation_report()


@app.post("/generate", response_model=GenerateResponse)
def generate(req: GenerateRequest):
    # If no llama binary found, return recommendation only
    bin_path = find_llama_cpp_binary()
    report = recommendation_report()
    if not bin_path:
        raise HTTPException(status_code=503, detail={"error": "No local runtime found", "recommendation": report})

    # Attempt to assemble a command; the exact command semantics depend on the binary.
    # We attempt a conservative pattern: run the binary with model path and send prompt via stdin.
    model_dir = Path("models")
    model_file = model_dir / "model.gguf"
    if not model_file.exists():
        raise HTTPException(status_code=404, detail={"error": "Model not found", "expected": str(model_file)})

    try:
        cmd = assemble_llama_command(model_file, backend=recommend_backend())
    except Exception as e:
        raise HTTPException(status_code=500, detail={"error": "Could not assemble runtime command", "exc": str(e)})

    # Many llama.cpp forks accept input on stdin; we'll try passing the prompt via stdin.
    # This is best-effort and may need runtime-specific flags for optimal results.
    try:
        process = subprocess.run([str(x) for x in cmd], input=req.prompt, capture_output=True, text=True, timeout=120)
    except subprocess.TimeoutExpired:
        raise HTTPException(status_code=504, detail={"error": "Runtime timeout"})
    except Exception as e:
        raise HTTPException(status_code=500, detail={"error": "Runtime execution failed", "exc": str(e)})

    output = process.stdout.strip() or process.stderr.strip()
    return GenerateResponse(output=output, metadata={"cmd": cmd, "rc": process.returncode})


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=5100)
