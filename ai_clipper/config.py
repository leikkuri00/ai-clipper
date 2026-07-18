"""
Configuration for AI Clipper — Ultimate Video Clip Selection & Story Engine.
Every tunable value is a CLI flag or config field — nothing hardcoded.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal


@dataclass
class ClipperConfig:
    # ── Source ──────────────────────────────────────────────
    source_url: str = ""
    source_path: Path | None = None

    # ── Model / provider settings ─────────────────────────
    # Whisper: "groq" (cloud, fast) or "local" (faster-whisper CPU)
    transcribe_provider: Literal["groq", "local"] = "groq"
    whisper_model: str = "large-v3-turbo"  # for local faster-whisper
    device: str = "auto"                    # "cpu", "cuda", "auto"
    compute_type: str = "int8"             # "int8", "float16", "auto"

    # Groq API key (free tier: 2000 req/day, no card needed)
    groq_api_key: str = field(
        default_factory=lambda: os.environ.get("GROQ_API_KEY", "")
    )

    # LLM scoring
    llm_provider: Literal["local", "ollama", "openai", "router"] = "local"
    llm_model: str = field(default_factory=lambda: os.environ.get("LLAMA_MODEL_PATH", ""))
    cloud_model: str = "gpt-4o-mini"
    cloud_api_base: str = "https://api.openai.com/v1"

    # LLM router script path (PowerShell multi-provider router)
    llm_router_script: str = "llm-router.ps1"

    # ── Diarization (auto-enabled for 2+ speakers) ────────
    enable_diarization: bool = False   # auto-enabled by pipeline
    hf_token: str = field(
        default_factory=lambda: os.environ.get("HF_TOKEN", "")
    )

    # ── Master Prompt: Clip Parameters ────────────────────
    # Output A: Standalone clips
    num_clips: int = 8                          # Master prompt: exactly 8
    target_clip_duration: float = 148.0          # Master prompt: 2 min 28 sec
    clip_duration_tolerance: float = 3.0         # ±3 seconds for natural speech
    min_clip_duration: float = 15.0              # seconds (for legacy compat)
    max_clip_duration: float = 180.0             # seconds (for legacy compat)
    min_viral_score: float = 60.0                # 0-100 scale
    overlap_dedup_threshold: float = 0.3          # drop if overlap >30% with higher scorer

    # ── Master Prompt: Candidate Generation ───────────────
    min_candidates: int = 300                    # minimum candidates to generate
    iterative_rounds: int = 3                    # optimization refinement rounds

    # ── Master Prompt: Output B - Story Series ────────────
    # One continuous 3-episode series that tells the full most-viral story,
    # each episode picking up chronologically where the previous ended.
    num_story_series: int = 1                     # best complete story arc
    episodes_per_series: int = 3                  # 3 chronological episodes
    episode_target_duration: float = 148.0        # 2 min 28 sec per episode

    # ── Segment / chunking ─────────────────────────────────
    chunk_duration_sec: float = 1200.0  # ~20 min chunks for LLM
    chunk_overlap_sec: float = 120.0    # 2 min overlap between chunks
    words_per_segment: int = 100        # Words per scoring window (legacy)
    segment_overlap: int = 25           # Word overlap between windows (legacy)

    # ── Semantic Segmentation ─────────────────────────────
    semantic_segmentation: bool = True            # use meaning-based segmentation
    max_segment_words: int = 300                  # max words per semantic block
    min_segment_words: int = 30                   # min words per semantic block

    # ── Multimodal Analysis ──────────────────────────────
    enable_audio_emotion: bool = True             # pitch, energy, emotion analysis
    enable_video_analysis: bool = True            # scene detection, motion analysis
    enable_face_analysis: bool = False            # facial expressions (heavy, optional)

    # ── Scene detection ────────────────────────────────────
    use_scene_detection: bool = True
    min_scene_length: float = 1.0
    scene_threshold: float = 27.0  # PySceneDetect threshold

    # ── Audio ──────────────────────────────────────────────
    loudness_normalize: bool = True
    target_lufts: float = -14.0
    audio_energy_boost: bool = True
    energy_boost_factor: float = 1.3
    trim_silence: bool = True
    silence_threshold_db: float = -40.0
    min_silence_duration: float = 0.3

    # ── Captions ──────────────────────────────────────────
    caption_style: Literal["karaoke", "subtitle", "none"] = "karaoke"
    caption_font_size: int = 48
    caption_max_chars_per_line: int = 28
    caption_position: Literal["center", "bottom"] = "center"
    caption_font: str = "Arial"
    caption_highlight_color: str = "#FFD700"  # gold karaoke highlight

    # ── Hook title ────────────────────────────────────────
    show_hook_title: bool = True
    hook_title_duration: float = 2.5  # seconds to show the hook line

    # ── Reframe / aspect ratio ────────────────────────────
    target_aspect: Literal["9:16", "1:1", "4:5", "original"] = "9:16"
    reframe_strategy: Literal["center", "tracked", "smart", "split"] = "tracked"
    vertical_width: int = 1080
    vertical_height: int = 1920

    # ── Tracking (MediaPipe / YOLO) ──────────────────────
    use_person_tracking: bool = True
    tracking_model: Literal["mediapipe", "yolo"] = "mediapipe"
    tracking_smoothing: float = 0.3  # EMA smoothing for pan-zoom

    # ── Background music ──────────────────────────────────
    bgm_enabled: bool = False
    bgm_path: Path | None = None
    bgm_volume: float = 0.15  # relative to speech

    # ── Output ─────────────────────────────────────────────
    output_dir: Path = field(default_factory=lambda: Path("./output"))
    keep_download: bool = False
    keep_intermediates: bool = False

    # ── Language ───────────────────────────────────────────
    language: str = "auto"

    # ── Mode flags ─────────────────────────────────────────
    offline_mode: bool = False
    verbose: bool = False
    host_binding: str = ""  # e.g. "0.0.0.0" for network access

    # ── Content moderation ────────────────────────────────
    moderate_content: bool = True
    moderation_threshold: float = 0.5  # lower = stricter

    # ── Hashtag generation ─────────────────────────────────
    generate_hashtags: bool = True
    max_hashtags: int = 10
    hashtags_in_filename: bool = True

    # ── Download ───────────────────────────────────────────
    max_resolution: int = 1080
    # Cookies for sites that require sign-in (e.g. YouTube bot checks).
    # Either a browser name to read cookies from, or a cookies.txt path.
    cookies_from_browser: str = field(
        default_factory=lambda: os.environ.get("AICLIPPER_COOKIES_FROM_BROWSER", "")
    )
    cookiefile: str = field(
        default_factory=lambda: os.environ.get("AICLIPPER_COOKIEFILE", "")
    )

    # ── Cost control ───────────────────────────────────────
    groq_daily_limit: int = 1800  # below 2000 to leave margin

    # ── Mode flags ─────────────────────────────────────────
    skip_multimodal: bool = False       # skip video/audio analysis for speed
    skip_story_series: bool = False     # skip story series generation
    story_only: bool = False            # only generate story series, no standalone clips

    # ── Backward-compatible aliases ────────────────────────
    @property
    def vertical_crop(self) -> bool:
        """True if output is vertical (9:16 or 4:5)."""
        return self.target_aspect in ("9:16", "4:5")

    @property
    def crop_aspect_w(self) -> int:
        return self.resolve_aspect()[0]

    @property
    def crop_aspect_h(self) -> int:
        return self.resolve_aspect()[1]

    def resolve_aspect(self) -> tuple[int, int]:
        """Return (w, h) ratio for the target aspect."""
        mapping = {"9:16": (9, 16), "1:1": (1, 1), "4:5": (4, 5), "original": (16, 9)}
        return mapping.get(self.target_aspect, (9, 16))

    def is_vertical(self) -> bool:
        return self.target_aspect in ("9:16", "4:5")

    def video_output_size(self) -> tuple[int, int]:
        """Return (width, height) for the output video."""
        if self.target_aspect == "9:16":
            return (1080, 1920)
        elif self.target_aspect == "4:5":
            return (1080, 1350)
        elif self.target_aspect == "1:1":
            return (1080, 1080)
        else:
            return (1920, 1080)  # keep original 16:9
