"""
Audio analysis utilities.
Detects loudness peaks, speech segments, and audio energy.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import List, Tuple

import numpy as np

logger = logging.getLogger(__name__)


def analyze_audio_energy(
    audio_path: Path,
    sample_rate: int = 16000,
    window_ms: int = 100,
) -> List[Tuple[float, float]]:
    """
    Compute RMS energy over sliding windows.

    Returns:
        List of (time_seconds, energy) tuples
    """
    try:
        import soundfile as sf
    except ImportError:
        logger.warning(
            "soundfile not installed. Install with: pip install soundfile"
        )
        return []

    try:
        audio, sr = sf.read(str(audio_path))
    except Exception as e:
        logger.error(f"Failed to read audio: {e}")
        return []

    # Convert to mono if stereo
    if audio.ndim > 1:
        audio = audio.mean(axis=1)

    # Resample if needed
    if sr != sample_rate:
        logger.warning(f"Resampling {sr}Hz → {sample_rate}Hz not implemented")
        # Simple approach: just work with native sample rate
        sample_rate = sr

    window_samples = int(sample_rate * window_ms / 1000)
    energy_data: List[Tuple[float, float]] = []

    for i in range(0, len(audio), window_samples):
        chunk = audio[i : i + window_samples]
        if len(chunk) < window_samples // 2:
            break
        rms = np.sqrt(np.mean(chunk ** 2))
        time_sec = i / sample_rate
        energy_data.append((time_sec, float(rms)))

    return energy_data


def detect_high_energy_regions(
    energy_data: List[Tuple[float, float]],
    threshold_multiplier: float = 1.3,
    min_duration: float = 2.0,
) -> List[Tuple[float, float]]:
    """
    Find regions where audio energy exceeds the mean * threshold.
    
    Returns:
        List of (start_time, end_time) for high-energy regions
    """
    if not energy_data:
        return []

    energies = np.array([e[1] for e in energy_data])
    mean_energy = np.mean(energies)
    threshold = mean_energy * threshold_multiplier

    logger.info(
        f"Audio energy: mean={mean_energy:.4f}, "
        f"threshold={threshold:.4f}"
    )

    # Find contiguous high-energy regions
    regions: List[Tuple[float, float]] = []
    in_region = False
    region_start = 0.0

    for time_sec, energy in energy_data:
        is_high = energy > threshold
        if is_high and not in_region:
            region_start = time_sec
            in_region = True
        elif not is_high and in_region:
            duration = time_sec - region_start
            if duration >= min_duration:
                regions.append((region_start, time_sec))
            in_region = False

    if in_region:
        end_time = energy_data[-1][0]
        duration = end_time - region_start
        if duration >= min_duration:
            regions.append((region_start, end_time))

    logger.info(f"Found {len(regions)} high-energy regions")
    return regions
