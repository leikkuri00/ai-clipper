"""
Transcription module.
- Groq Whisper API (default, ~1 min for 60-min file, free 2000 req/day)
- faster-whisper local (offline fallback, CPU/int8)
- Silero VAD filtering to cut hallucinated text in silence
"""

from __future__ import annotations

import json
import logging
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import List, Optional

import numpy as np
from pydantic import BaseModel

logger = logging.getLogger(__name__)


# ── Data models ────────────────────────────────────────────

class WordTimestamp(BaseModel):
    word: str
    start: float
    end: float
    confidence: float = 1.0


class TranscriptSegment(BaseModel):
    text: str
    start: float
    end: float
    words: List[WordTimestamp] = []


class Transcript(BaseModel):
    segments: List[TranscriptSegment]
    language: str = "en"
    duration: float = 0.0

    @property
    def full_text(self) -> str:
        return " ".join(seg.text for seg in self.segments)

    @property
    def all_words(self) -> List[WordTimestamp]:
        words: List[WordTimestamp] = []
        for seg in self.segments:
            words.extend(seg.words)
        return words

    def to_srt(self) -> str:
        """Export as SRT subtitle file content."""
        lines = []
        for i, seg in enumerate(self.segments, 1):
            start_srt = _seconds_to_srt_time(seg.start)
            end_srt = _seconds_to_srt_time(seg.end)
            lines.append(str(i))
            lines.append(f"{start_srt} --> {end_srt}")
            lines.append(seg.text)
            lines.append("")
        return "\n".join(lines)

    def get_text_between(self, t_start: float, t_end: float) -> str:
        """Get transcript text between two timestamps."""
        words_in_range = [
            w for w in self.all_words
            if w.start >= t_start and w.end <= t_end
        ]
        return " ".join(w.word for w in words_in_range)


def _seconds_to_srt_time(seconds: float) -> str:
    """Convert seconds to SRT timestamp format: HH:MM:SS,mmm"""
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    millis = int((seconds % 1) * 1000)
    return f"{hours:02d}:{minutes:02d}:{secs:02d},{millis:03d}"


# ── Silero VAD ──────────────────────────────────────────────

def _load_silero_vad():
    """Lazy-load Silero VAD model."""
    try:
        import torch
        model, utils = torch.hub.load(
            repo_or_dir="snakers4/silero-vad",
            model="silero_vad",
            force_reload=False,
            trust_repo=True,
        )
        (get_speech_timestamps, _, read_audio, _, _) = utils
        return model, get_speech_timestamps, read_audio
    except Exception as e:
        logger.warning(f"Silero VAD not available: {e}. VAD filtering disabled.")
        return None, None, None


def _filter_silence_vad(
    audio_path: Path,
    words: List[WordTimestamp],
) -> List[WordTimestamp]:
    """
    Use Silero VAD to detect speech segments and filter out words
    that fall in silent regions (hallucinated text).
    """
    model, get_speech_timestamps, read_audio = _load_silero_vad()
    if model is None:
        return words  # can't filter, return all

    try:
        wav = read_audio(str(audio_path), sampling_rate=16000)
        speech_ts = get_speech_timestamps(wav, model, return_seconds=True)

        if not speech_ts:
            logger.warning("Silero VAD found no speech — returning all words unfiltered")
            return words

        filtered = []
        for w in words:
            word_mid = (w.start + w.end) / 2
            for sp in speech_ts:
                if sp["start"] <= word_mid <= sp["end"]:
                    filtered.append(w)
                    break

        removed = len(words) - len(filtered)
        if removed > 0:
            logger.info(f"Silero VAD: removed {removed}/{len(words)} hallucinated words")
        return filtered

    except Exception as e:
        logger.warning(f"Silero VAD filtering failed: {e}")
        return words


# ── Groq API transcription ─────────────────────────────────

def _transcribe_groq(
    audio_path: Path,
    api_key: str,
    language: Optional[str] = None,
    prompt: Optional[str] = None,
) -> Transcript:
    """
    Transcribe using Groq's Whisper API (fast, free tier).
    Returns word-level timestamps.
    """
    import requests

    if not api_key:
        raise ValueError("Groq API key not set. Set GROQ_API_KEY env var or pass --groq-api-key.")

    url = "https://api.groq.com/openai/v1/audio/transcriptions"

    # Groq has a 25 MB file limit; chunk if needed
    file_size_mb = audio_path.stat().st_size / (1024 * 1024)
    if file_size_mb > 24:
        logger.warning(f"Audio file is {file_size_mb:.1f} MB; Groq limit is 25 MB. "
                       f"Falling back to local transcription.")
        raise ValueError("File too large for Groq API (25 MB limit).")

    logger.info(f"Sending {file_size_mb:.1f} MB to Groq Whisper API...")
    t0 = time.time()

    with open(audio_path, "rb") as f:
        data = {
            "model": "whisper-large-v3",
            "response_format": "verbose_json",
            "timestamp_granularities[]": "word",
        }
        if language and language != "auto":
            data["language"] = language
        if prompt:
            data["prompt"] = prompt

        files = {"file": (audio_path.name, f, "audio/wav")}

        resp = requests.post(
            url,
            data=data,
            files=files,
            headers={"Authorization": f"Bearer {api_key}"},
            timeout=300,
        )

    elapsed = time.time() - t0
    logger.info(f"Groq response in {elapsed:.1f}s (status={resp.status_code})")

    if resp.status_code != 200:
        error_msg = resp.text[:500]
        logger.error(f"Groq API error ({resp.status_code}): {error_msg}")
        raise RuntimeError(f"Groq API returned {resp.status_code}: {error_msg}")

    result = resp.json()

    # Parse word-level timestamps
    segments: List[TranscriptSegment] = []
    all_words: List[WordTimestamp] = []

    for seg in result.get("segments", []):
        seg_words = []
        for w in seg.get("words", []):
            wt = WordTimestamp(
                word=w.get("word", "").strip(),
                start=w.get("start", 0),
                end=w.get("end", 0),
                confidence=w.get("confidence", 1.0),
            )
            seg_words.append(wt)
            all_words.append(wt)

        segments.append(TranscriptSegment(
            text=seg.get("text", "").strip(),
            start=seg.get("start", 0),
            end=seg.get("end", 0),
            words=seg_words,
        ))

    lang = result.get("language", language or "en")
    duration = result.get("duration", segments[-1].end if segments else 0)

    logger.info(
        f"Groq transcript: {len(segments)} segments, "
        f"{len(all_words)} words, lang={lang}"
    )

    return Transcript(
        segments=segments,
        language=lang,
        duration=duration,
    )


# ── Local faster-whisper transcription ──────────────────────

def _transcribe_local(
    audio_path: Path,
    model_size: str = "large-v3-turbo",
    device: str = "cpu",
    compute_type: str = "int8",
    language: Optional[str] = None,
) -> Transcript:
    """Transcribe using local faster-whisper."""
    from faster_whisper import WhisperModel

    logger.info(f"Loading Whisper model '{model_size}' on '{device}' ({compute_type})...")
    t0 = time.time()

    model = WhisperModel(model_size, device=device, compute_type=compute_type)

    logger.info(f"Model loaded in {time.time() - t0:.1f}s. Transcribing...")
    t0 = time.time()

    segments_iter, info = model.transcribe(
        str(audio_path),
        language=language,
        word_timestamps=True,
        vad_filter=True,
        vad_parameters=dict(min_silence_duration_ms=500),
    )

    segments: List[TranscriptSegment] = []
    for seg in segments_iter:
        words = [
            WordTimestamp(
                word=w.word.strip(),
                start=w.start,
                end=w.end,
                confidence=w.probability,
            )
            for w in (seg.words or [])
        ]
        segments.append(TranscriptSegment(
            text=seg.text.strip(),
            start=seg.start,
            end=seg.end,
            words=words,
        ))

    result = Transcript(
        segments=segments,
        language=info.language,
        duration=info.duration,
    )

    elapsed = time.time() - t0
    logger.info(
        f"Local transcript: {len(segments)} segments, "
        f"lang={result.language}, {elapsed:.1f}s"
    )
    return result


# ── Main entry point ────────────────────────────────────────

def transcribe_audio(
    audio_path: Path,
    provider: str = "groq",
    model_size: str = "large-v3-turbo",
    device: str = "cpu",
    compute_type: str = "int8",
    language: Optional[str] = None,
    api_key: str = "",
    use_vad_filter: bool = True,
    offline_mode: bool = False,
) -> Transcript:
    """
    Transcribe an audio file with word-level timestamps.

    Args:
        audio_path: Path to 16kHz mono WAV
        provider: "groq" (cloud, fast) or "local" (faster-whisper)
        model_size: Whisper model for local mode
        device: "cpu", "cuda", "auto"
        compute_type: "int8", "float16", "auto"
        language: Language code or None for auto-detect
        api_key: Groq API key
        use_vad_filter: Apply Silero VAD post-filter
        offline_mode: Force local transcription

    Returns:
        Transcript with word-level timestamps
    """
    if offline_mode:
        provider = "local"

    # Whisper expects None (not "auto"/"") for language auto-detection.
    if language in (None, "", "auto"):
        language = None

    # ── Transcribe ─────────────────────────────────────────
    if provider == "groq":
        try:
            transcript = _transcribe_groq(audio_path, api_key, language)
        except Exception as e:
            logger.warning(f"Groq transcription failed: {e}")
            logger.warning("Falling back to local faster-whisper...")
            transcript = _transcribe_local(audio_path, model_size, device, compute_type, language)
    else:
        transcript = _transcribe_local(audio_path, model_size, device, compute_type, language)

    # ── Post-filter with Silero VAD ────────────────────────
    if use_vad_filter:
        for seg in transcript.segments:
            seg.words = _filter_silence_vad(audio_path, seg.words)

    return transcript


# ── Cost tracking ───────────────────────────────────────────

_GROQ_CALL_COUNT = 0
_GROQ_RESET_DATE = ""


def get_groq_usage_today() -> int:
    """Return approximate Groq API call count for today."""
    global _GROQ_CALL_COUNT, _GROQ_RESET_DATE
    from datetime import date
    today = str(date.today())
    if _GROQ_RESET_DATE != today:
        _GROQ_CALL_COUNT = 0
        _GROQ_RESET_DATE = today
    return _GROQ_CALL_COUNT


def _increment_groq_count():
    global _GROQ_CALL_COUNT
    _GROQ_CALL_COUNT += 1
