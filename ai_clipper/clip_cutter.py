"""
Clip cutter and caption burner.
Uses FFmpeg to extract clips and burn subtitles/captions.
Supports vertical 9:16 cropping for TikTok/Reels/Shorts.
"""

from __future__ import annotations

import logging
import subprocess
import tempfile
from pathlib import Path
from typing import List, Optional

from .config import ClipperConfig

logger = logging.getLogger(__name__)


def extract_audio(video_path: Path, output_path: Optional[Path] = None) -> Path:
    """
    Extract audio from video as 16kHz mono WAV (optimal for Whisper).
    """
    if output_path is None:
        output_path = Path(tempfile.mktemp(suffix=".wav"))

    cmd = [
        "ffmpeg", "-y",
        "-i", str(video_path),
        "-vn",
        "-acodec", "pcm_s16le",
        "-ar", "16000",
        "-ac", "1",
        str(output_path),
    ]

    logger.info(f"Extracting audio: {video_path} → {output_path}")
    subprocess.run(cmd, check=True, capture_output=True)
    return output_path


def create_ass_subtitle_file(
    words: list,
    output_path: Path,
    config: ClipperConfig,
    shift_start: float = 0.0,
) -> Path:
    """
    Create an ASS subtitle file from word timestamps.
    Automatically adjusts PlayRes for vertical crop if enabled.
    """
    font_size = config.caption_font_size
    max_chars = config.caption_max_chars_per_line
    position = config.caption_position

    # Adjust PlayRes for vertical crop
    if config.vertical_crop:
        play_res_x = config.vertical_width
        play_res_y = config.vertical_height
    else:
        play_res_x = 1920
        play_res_y = 1080

    # Determine vertical position
    if position == "center":
        alignment = 5  # center-center
        margin_v = 0
    else:
        alignment = 2  # bottom-center
        margin_v = 50

    lines = [
        "[Script Info]",
        "ScriptType: v4.00+",
        f"PlayResX: {play_res_x}",
        f"PlayResY: {play_res_y}",
        "WrapStyle: 2",
        "",
        "[V4+ Styles]",
        "Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, "
        "OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, "
        "ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, "
        "Alignment, MarginL, MarginR, MarginV, Encoding",
        (
            f"Style: Default, Arial, {font_size}, &H00FFFFFF, &H000000FF, "
            f"&H00000000, &H80000000, 1, 0, 0, 0, 100, 100, 0, 0, 1, "
            f"3.5, 1.5, {alignment}, 60, 60, {margin_v}, 1"
        ),
        "",
        "[Events]",
        "Format: Layer, Start, End, Style, Name, MarginL, MarginR, "
        "MarginV, Effect, Text",
    ]

    # Group words into caption lines
    current_line: List[str] = []
    current_start: Optional[float] = None
    current_end: Optional[float] = None

    def flush_line():
        nonlocal current_line, current_start, current_end
        if not current_line:
            return ""
        line_text = " ".join(current_line)
        line_text = line_text.replace("\\", "\\\\").replace("{", "\\{").replace("}", "\\}")

        start_ass = _format_ass_time(max(0, (current_start or 0) - shift_start))
        end_ass = _format_ass_time(max(0, (current_end or 0) - shift_start))

        current_line.clear()
        current_start = None
        current_end = None

        return f"Dialogue: 0,{start_ass},{end_ass},Default,,0,0,0,,{line_text}\n"

    events = []
    for w in words:
        w_start = w.start - shift_start
        w_end = w.end - shift_start

        if w_start < 0:
            continue

        if current_line and len(" ".join(current_line + [w.word])) > max_chars:
            events.append(flush_line())

        current_line.append(w.word)
        if current_start is None:
            current_start = w_start
        current_end = w_end

    if current_line:
        events.append(flush_line())

    lines.extend(events)

    output_path.write_text("\n".join(lines), encoding="utf-8")
    logger.info(f"Created ASS subtitle: {output_path}")
    return output_path


def _format_ass_time(seconds: float) -> str:
    """Convert seconds to ASS time format: H:MM:SS.cc"""
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    centiseconds = int((seconds % 1) * 100)
    return f"{hours}:{minutes:02d}:{secs:02d}.{centiseconds:02d}"


def _build_ffmpeg_filter(
    ass_path_str: str,
    config: ClipperConfig,
) -> str:
    """
    Build the FFmpeg video filter chain.
    Handles subtitle burning and optional 9:16 vertical cropping.
    """
    filters = []

    if config.vertical_crop:
        # Crop horizontal (16:9) to vertical (9:16) — center crop
        # e.g. 1920x1080 source → crop 608x1080 center → scale to 1080x1920
        crop_w = f"ih*{config.crop_aspect_w}/{config.crop_aspect_h}"
        filters.append(f"crop={crop_w}:ih:(iw-ow)/2:0")
        filters.append(f"scale={config.vertical_width}:{config.vertical_height}")
        logger.info(
            f"Vertical crop: {config.crop_aspect_w}:{config.crop_aspect_h} "
            f"→ {config.vertical_width}x{config.vertical_height}"
        )

    # Burn subtitles on top
    filters.append(f"ass={ass_path_str}")

    return ",".join(filters)


def cut_and_caption_clip(
    video_path: Path,
    output_path: Path,
    start_time: float,
    end_time: float,
    words: list,
    config: ClipperConfig,
    extra_padding: float = 0.3,
) -> Path:
    """
    Cut a clip from the video and burn captions onto it.
    Supports optional 9:16 vertical cropping.
    """
    clip_start = max(0, start_time - extra_padding)
    clip_end = end_time + extra_padding
    duration = clip_end - clip_start

    # Filter words that fall within the clip
    clip_words = [
        w for w in words
        if w.end > clip_start and w.start < clip_end
    ]

    if not clip_words:
        logger.warning(f"No words in clip range {clip_start:.1f}-{clip_end:.1f}")
        return _cut_simple(video_path, output_path, clip_start, duration, config)

    # Create ASS subtitle file
    ass_path = output_path.with_suffix(".ass")
    create_ass_subtitle_file(clip_words, ass_path, config, shift_start=clip_start)

    # Build filter chain (crop + subtitles)
    ass_path_str = str(ass_path).replace("\\", "/").replace(":", "\\\\:")
    filter_chain = _build_ffmpeg_filter(ass_path_str, config)

    cmd = [
        "ffmpeg", "-y",
        "-ss", str(clip_start),
        "-i", str(video_path),
        "-t", str(duration),
        "-vf", filter_chain,
        "-c:v", "libx264",
        "-c:a", "aac",
        "-preset", "fast",
        "-crf", "23",
        str(output_path),
    ]

    logger.info(f"Cutting clip: {clip_start:.1f}s - {clip_end:.1f}s → {output_path}")
    result = subprocess.run(cmd, capture_output=True, text=True)

    if result.returncode != 0:
        logger.error(f"FFmpeg failed: {result.stderr[:500]}")
        logger.info("Retrying without subtitles...")
        return _cut_simple(video_path, output_path, clip_start, duration, config)

    return output_path


def _cut_simple(
    video_path: Path,
    output_path: Path,
    start: float,
    duration: float,
    config: ClipperConfig,
) -> Path:
    """Simple cut without captions (fallback)."""
    vf_args = []
    if config.vertical_crop:
        crop_w = f"ih*{config.crop_aspect_w}/{config.crop_aspect_h}"
        vf_args = [
            "-vf", f"crop={crop_w}:ih:(iw-ow)/2:0,scale={config.vertical_width}:{config.vertical_height}"
        ]

    cmd = [
        "ffmpeg", "-y",
        "-ss", str(start),
        "-i", str(video_path),
        "-t", str(duration),
        *vf_args,
        "-c:v", "libx264",
        "-c:a", "aac",
        "-preset", "fast",
        "-crf", "23",
        str(output_path),
    ]
    subprocess.run(cmd, check=True, capture_output=True)
    return output_path
