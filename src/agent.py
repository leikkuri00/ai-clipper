"""Local AI Coding Agent — minimal core for hardware detection and CLI

This module is an entrypoint for the local agent. It detects system hardware
and recommends a backend (Vulkan/CUDA/CPU) and candidate models.
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

import psutil
from rich import print


RECOMMENDED_MODELS = [
    "qwen2.5-coder",
    "qwen3-coder",
    "starcoder2",
    "deepseek-coder",
]


def detect_gpu():
    """Return a dict describing available GPU backends."""
    info = {"nvidia": False, "vulkan": False, "amd": False}
    if shutil.which("nvidia-smi"):
        info["nvidia"] = True
    # vulkaninfo may not be present; use which to detect
    if shutil.which("vulkaninfo"):
        info["vulkan"] = True
    # rudimentary check for AMD ROCm tools
    if shutil.which("rocm-smi"):
        info["amd"] = True
    return info


def recommend_backend():
    gpu = detect_gpu()
    if gpu["nvidia"]:
        return "cuda"
    if gpu["vulkan"]:
        return "vulkan"
    if gpu["amd"]:
        return "metal/rocm"
    return "cpu"


def system_summary():
    mem = psutil.virtual_memory()
    return {
        "cpu_count": psutil.cpu_count(logical=False),
        "cpu_count_logical": psutil.cpu_count(logical=True),
        "total_ram_gb": round(mem.total / (1024 ** 3), 2),
        "available_ram_gb": round(mem.available / (1024 ** 3), 2),
        "recommended_backend": recommend_backend(),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="local-agent", description="Local AI coding agent entry")
    parser.add_argument("--summary", action="store_true", help="Print system summary and recommended backend")
    parser.add_argument("--list-models", action="store_true", help="List recommended models")
    args = parser.parse_args(argv)

    if args.summary:
        summary = system_summary()
        print(json.dumps(summary, indent=2))
        return 0

    if args.list_models:
        print("Recommended models:")
        for m in RECOMMENDED_MODELS:
            print(" -", m)
        return 0

    # Default behaviour: print small interactive summary
    print("Local AI Coding Agent — minimal runner")
    print()
    print("System summary:")
    print(json.dumps(system_summary(), indent=2))
    print()
    print("Run with --list-models or --summary for automation.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
