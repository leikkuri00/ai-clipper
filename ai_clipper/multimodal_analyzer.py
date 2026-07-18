"""
Multimodal Analyzer — Step 2 of the Master Prompt.

Analyzes audio and video simultaneously for richer scoring signals:
  Audio: voice excitement, energy, volume, pitch, pauses, silence, laughter, emotion, music
  Video: scene changes, camera movement, facial expressions (optional), motion, brightness
  Language: context, meaning, intent, humor, sarcasm (delegated to LLM scorer)

Provides per-timestamp feature vectors used by the scorer and retention predictor.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class AudioFeatures:
    """Per-segment audio analysis results."""
    # Energy / loudness
    mean_rms: float = 0.0
    peak_rms: float = 0.0
    rms_variance: float = 0.0      # how much energy varies
    
    # Pitch / prosody  
    mean_pitch_hz: float = 0.0
    pitch_variance: float = 0.0
    pitch_range_hz: float = 0.0
    
    # Voice quality
    speaking_rate_wps: float = 0.0  # words per second
    pause_ratio: float = 0.0        # fraction of time in silence
    pause_count: int = 0
    mean_pause_duration: float = 0.0
    
    # Emotion indicators
    voice_excitement: float = 0.0   # 0-1 composite score
    volume_dynamics: float = 0.0    # 0-1 loudness variation
    emotional_arousal: float = 0.0  # 0-1 composite arousal
    
    # Speech presence
    speech_ratio: float = 1.0       # fraction of time with speech vs silence/music


@dataclass 
class VideoFeatures:
    """Per-segment video analysis results."""
    # Scene / visual
    mean_motion: float = 0.0        # average optical flow magnitude
    motion_variance: float = 0.0    # how dynamic is the scene
    scene_change_count: int = 0     # number of scene cuts
    scene_stability: float = 1.0    # 0-1, how stable is the frame
    
    # Brightness / quality
    mean_brightness: float = 0.0
    brightness_variance: float = 0.0
    is_dark: bool = False
    
    # Visual interest
    visual_variety: float = 0.0     # 0-1, how much the frame changes
    face_present: bool = False
    face_count: int = 0
    face_centered: bool = False


@dataclass
class MultimodalSegment:
    """Combined audio + video analysis for one time segment."""
    start_time: float
    end_time: float
    audio: AudioFeatures = field(default_factory=AudioFeatures)
    video: VideoFeatures = field(default_factory=VideoFeatures)


def analyze_audio_segment(
    audio_path: Path,
    start_time: float,
    end_time: float,
    sample_rate: int = 16000,
) -> AudioFeatures:
    """
    Analyze audio features for a specific time segment.
    
    Extracts RMS energy, attempts pitch estimation, measures pauses,
    and computes composite excitement/arousal scores.
    """
    features = AudioFeatures()
    
    try:
        import soundfile as sf
        audio, sr = sf.read(str(audio_path))
    except Exception as e:
        logger.warning(f"soundfile not available: {e}")
        return features

    if audio.ndim > 1:
        audio = audio.mean(axis=1)

    # Extract segment
    start_sample = int(start_time * sr)
    end_sample = int(end_time * sr)
    start_sample = max(0, min(start_sample, len(audio)))
    end_sample = max(start_sample, min(end_sample, len(audio)))
    
    segment = audio[start_sample:end_sample]
    if len(segment) < sr * 0.1:  # less than 100ms
        return features

    segment_sr = sr
    duration = len(segment) / segment_sr

    # ── RMS Energy ──────────────────────────────────────
    rms = np.sqrt(np.mean(segment ** 2))
    if rms > 0:
        # RMS variance over 100ms windows
        window = int(segment_sr * 0.1)
        rms_windows = []
        for i in range(0, len(segment), window):
            chunk = segment[i:i+window]
            if len(chunk) > window // 2:
                rms_windows.append(np.sqrt(np.mean(chunk ** 2)))
        
        features.mean_rms = float(rms)
        features.peak_rms = float(np.max(np.abs(segment)))
        features.rms_variance = float(np.var(rms_windows)) if rms_windows else 0.0
        
        # Volume dynamics (normalized variance)
        max_rms = max(rms_windows) if rms_windows else 1.0
        features.volume_dynamics = float(min(1.0, features.rms_variance / (max_rms + 1e-8)))
    else:
        features.mean_rms = 0.0
        features.peak_rms = 0.0
        features.rms_variance = 0.0
        features.speech_ratio = 0.0
        return features

    # ── Silence / Pause Detection ────────────────────────
    silence_threshold = rms * 0.15 if rms > 0 else 0.001
    min_silence_samples = int(segment_sr * 0.3)  # 300ms minimum pause
    
    is_silence = np.abs(segment) < silence_threshold
    
    # Find silence runs
    pause_durations = []
    silence_start = None
    for i, silent in enumerate(is_silence):
        if silent and silence_start is None:
            silence_start = i
        elif not silent and silence_start is not None:
            dur = (i - silence_start) / segment_sr
            if dur >= 0.3:
                pause_durations.append(dur)
            silence_start = None
    
    if silence_start is not None:
        dur = (len(is_silence) - silence_start) / segment_sr
        if dur >= 0.3:
            pause_durations.append(dur)

    total_silence = sum(pause_durations)
    features.pause_ratio = float(total_silence / duration) if duration > 0 else 0.0
    features.pause_count = len(pause_durations)
    features.mean_pause_duration = float(np.mean(pause_durations)) if pause_durations else 0.0
    features.speech_ratio = float(1.0 - features.pause_ratio)

    # ── Pitch Estimation (simple autocorrelation) ───────
    try:
        # Only attempt on voiced segments (RMS > threshold)
        voiced = np.abs(segment) > silence_threshold * 2
        if np.sum(voiced) > segment_sr * 0.1:  # at least 100ms of voiced audio
            voiced_segment = segment[voiced]
            if len(voiced_segment) > segment_sr * 0.1:
                # Autocorrelation pitch estimation
                pitches = _estimate_pitch_autocorr(voiced_segment, segment_sr)
                if pitches:
                    features.mean_pitch_hz = float(np.mean(pitches))
                    features.pitch_variance = float(np.var(pitches))
                    features.pitch_range_hz = float(np.max(pitches) - np.min(pitches))
    except Exception as e:
        logger.debug(f"Pitch estimation failed: {e}")

    # ── Voice Excitement (composite) ────────────────────
    # Combines: RMS variance, pitch variance, and speaking rate proxy
    excitement_factors = []
    if features.rms_variance > 0:
        excitement_factors.append(min(1.0, features.rms_variance * 50))
    if features.pitch_variance > 0:
        excitement_factors.append(min(1.0, features.pitch_variance / 5000))
    if features.speech_ratio > 0.3:
        excitement_factors.append(features.speech_ratio)
    
    features.voice_excitement = float(np.mean(excitement_factors)) if excitement_factors else 0.0
    features.emotional_arousal = float(
        0.5 * features.voice_excitement + 0.5 * features.volume_dynamics
    )

    return features


def analyze_video_segment(
    video_path: Path,
    start_time: float,
    end_time: float,
) -> VideoFeatures:
    """
    Analyze video features for a specific time segment.
    
    Uses OpenCV for motion estimation, scene change detection,
    and basic visual quality metrics.
    """
    features = VideoFeatures()
    
    try:
        import cv2
    except ImportError:
        logger.debug("OpenCV not available for video analysis")
        return features

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        logger.warning(f"Cannot open video: {video_path}")
        return features

    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    if fps <= 0:
        cap.release()
        return features

    start_frame = int(start_time * fps)
    end_frame = int(end_time * fps)
    start_frame = max(0, min(start_frame, total_frames - 1))
    end_frame = max(start_frame + 1, min(end_frame, total_frames))

    # Sample frames (every 5th frame for performance)
    sample_step = max(1, min(5, (end_frame - start_frame) // 30))
    
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
    
    prev_gray = None
    motions = []
    brightnesses = []
    scene_changes = 0
    sampled = 0

    for fnum in range(start_frame, end_frame, sample_step):
        ret, frame = cap.read()
        if not ret:
            break
        
        sampled += 1
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Brightness
        brightness = float(np.mean(gray))
        brightnesses.append(brightness)
        
        # Motion (optical flow magnitude)
        if prev_gray is not None:
            flow = cv2.calcOpticalFlowFarneback(
                prev_gray, gray, None, 0.5, 3, 15, 3, 5, 1.2, 0
            )
            mag = np.sqrt(flow[..., 0]**2 + flow[..., 1]**2)
            motions.append(float(np.mean(mag)))
            
            # Scene change detection: sudden motion spike
            if motions and len(motions) > 1:
                if motions[-1] > np.mean(motions[:-1]) * 3 + 2:
                    scene_changes += 1

        prev_gray = gray

    cap.release()

    if sampled == 0:
        return features

    features.mean_motion = float(np.mean(motions)) if motions else 0.0
    features.motion_variance = float(np.var(motions)) if motions else 0.0
    features.scene_change_count = scene_changes
    features.scene_stability = float(1.0 / (1.0 + features.motion_variance))
    features.mean_brightness = float(np.mean(brightnesses)) if brightnesses else 0.0
    features.brightness_variance = float(np.var(brightnesses)) if brightnesses else 0.0
    features.is_dark = features.mean_brightness < 40
    features.visual_variety = float(min(1.0, features.motion_variance / 20))

    return features


def _estimate_pitch_autocorr(
    signal: np.ndarray, 
    sample_rate: int,
    min_freq: float = 50.0,
    max_freq: float = 500.0,
) -> List[float]:
    """
    Estimate pitch using autocorrelation method.
    Works on short chunks and returns detected pitches.
    """
    chunk_duration = 0.05  # 50ms chunks
    chunk_samples = int(sample_rate * chunk_duration)
    
    pitches = []
    
    for i in range(0, len(signal) - chunk_samples, chunk_samples // 2):
        chunk = signal[i: i + chunk_samples]
        if len(chunk) < chunk_samples // 2:
            break
        
        # Autocorrelation
        corr = np.correlate(chunk, chunk, mode='full')
        corr = corr[len(corr)//2:]  # take second half
        
        # Find peaks (excluding zero lag)
        min_lag = int(sample_rate / max_freq)
        max_lag = int(sample_rate / min_freq)
        
        if min_lag >= len(corr):
            continue
        
        search_range = corr[min_lag: min(max_lag, len(corr))]
        if len(search_range) == 0:
            continue
        
        peak_idx = np.argmax(search_range)
        peak_val = search_range[peak_idx]
        
        if peak_val > corr[0] * 0.3:  # minimum correlation threshold
            lag = peak_idx + min_lag
            pitch = sample_rate / lag
            if min_freq <= pitch <= max_freq:
                pitches.append(pitch)
    
    return pitches


def compute_multimodal_profile(
    audio_path: Path,
    video_path: Path,
    segments: List[Tuple[float, float]],
    skip_video: bool = False,
) -> List[MultimodalSegment]:
    """
    Compute combined audio+video features for multiple time segments.

    Args:
        audio_path: Path to extracted audio WAV
        video_path: Path to video file
        segments: List of (start_time, end_time) tuples
        skip_video: Skip video analysis for speed

    Returns:
        List of MultimodalSegment with combined features
    """
    results = []
    
    for i, (start, end) in enumerate(segments):
        audio_feat = analyze_audio_segment(audio_path, start, end)
        
        video_feat = VideoFeatures()
        if not skip_video:
            try:
                video_feat = analyze_video_segment(video_path, start, end)
            except Exception as e:
                logger.debug(f"Video analysis failed for segment {i}: {e}")
        
        results.append(MultimodalSegment(
            start_time=start,
            end_time=end,
            audio=audio_feat,
            video=video_feat,
        ))

    return results
