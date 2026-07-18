"""Local LLM runtime helper for the agent.

Provides hardware detection, backend recommendation, and model/runtime suggestion
logic for llama.cpp-style runtimes. This module does not download models — it
recommends settings and assembles commands the user can run.
"""

from __future__ import annotations

import json
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Dict

import psutil


@dataclass
class HardwareProfile:
    cpu_physical_cores: int
    cpu_logical_cores: int
    total_ram_gb: float
    has_nvidia: bool
    has_vulkan: bool
    has_amd: bool


def detect_gpu_backends() -> Dict[str, bool]:
    """Detect available GPU-related tooling.

    Returns a dict with keys: nvidia, vulkan, amd
    """
    info = {"nvidia": False, "vulkan": False, "amd": False}
    if shutil.which("nvidia-smi"):
        info["nvidia"] = True
    if shutil.which("vulkaninfo"):
        info["vulkan"] = True
    if shutil.which("rocm-smi"):
        info["amd"] = True
    return info


def system_profile() -> HardwareProfile:
    mem = psutil.virtual_memory()
    gpu = detect_gpu_backends()
    return HardwareProfile(
        cpu_physical_cores=psutil.cpu_count(logical=False) or 1,
        cpu_logical_cores=psutil.cpu_count(logical=True) or 1,
        total_ram_gb=round(mem.total / (1024 ** 3), 2),
        has_nvidia=gpu["nvidia"],
        has_vulkan=gpu["vulkan"],
        has_amd=gpu["amd"],
    )


def recommend_backend(profile: HardwareProfile | None = None) -> str:
    """Recommend best backend: cuda, vulkan, metal/rocm, or cpu."""
    p = profile or system_profile()
    if p.has_nvidia:
        return "cuda"
    if p.has_vulkan:
        return "vulkan"
    if p.has_amd:
        return "metal/rocm"
    return "cpu"


def recommend_model_size(profile: HardwareProfile | None = None) -> Dict[str, str]:
    """Recommend a model family and quantization settings based on RAM/VRAM heuristics.

    Returns a dict with keys: family, quant, context
    """
    p = profile or system_profile()
    ram = p.total_ram_gb
    # Conservative recommendations
    if ram >= 64:
        return {"family": "qwen3-coder or qwen2.5-coder (large)", "quant": "q4_0/q4_1", "context": "32k"}
    if ram >= 32:
        return {"family": "qwen2.5-coder (mid)", "quant": "q4_1/q4_0", "context": "16k"}
    if ram >= 16:
        return {"family": "starcoder2 or qwen2.5-coder (small)", "quant": "q4_0 or q8_0", "context": "8k"}
    return {"family": "starcoder2 (tiny) or GLM", "quant": "q4_0/q8_0", "context": "4k"}


def find_llama_cpp_binary() -> Path | None:
    """Try to locate a llama.cpp-style native binary on PATH.

    Common names: `llama`, `llama.cpp`, `main`, `llama.exe` — this is heuristic.
    """
    candidates = ["llama", "llama.cpp", "llama.exe", "main", "quantize", "server"]
    for name in candidates:
        path = shutil.which(name)
        if path:
            return Path(path)
    return None


def assemble_llama_command(model_path: Path, backend: str | None = None, n_ctx: int | None = None) -> list[str]:
    """Assemble a recommended command to run a local llama.cpp-style binary.

    The returned command is advisory — users must have an appropriate binary.
    """
    bin_path = find_llama_cpp_binary()
    if not bin_path:
        raise FileNotFoundError("No llama.cpp-style binary found on PATH. Install llama.cpp or provide a runtime.")
    cmd = [str(bin_path), "-m", str(model_path)]
    if backend:
        if backend == "cuda":
            cmd += ["--gpu"]
        elif backend == "vulkan":
            cmd += ["--vulkan"]
    if n_ctx:
        cmd += ["--n_ctx", str(n_ctx)]
    return cmd


def recommendation_report() -> dict:
    """Return a serializable recommendation report for tooling/UI."""
    profile = system_profile()
    return {
        "hardware": profile.__dict__,
        "recommended_backend": recommend_backend(profile),
        "model_recommendation": recommend_model_size(profile),
        "llama_cpp_binary": str(find_llama_cpp_binary() or "(not found)"),
    }


if __name__ == "__main__":
    import argparse

    p = argparse.ArgumentParser(description="LLM runtime helper: recommend backend and models")
    p.add_argument("--report", action="store_true", help="Print JSON recommendation report")
    args = p.parse_args()
    if args.report:
        print(json.dumps(recommendation_report(), indent=2))
    else:
        print("Run with --report to get a JSON recommendation for model/runtime settings.")
