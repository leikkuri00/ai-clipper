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
from .reframe import compute_subject_center_x

logger = logging.getLogger(__name__)


# High-impact words worth emphasizing in captions, and emotion → emoji hints.
_EMPHASIS_WORDS = {
    "never", "always", "everyone", "nobody", "everything", "nothing",
    "secret", "crazy", "insane", "unbelievable", "shocking", "wild",
    "money", "million", "millions", "billion", "billions", "dollars",
    "free", "first", "best", "worst", "biggest", "huge", "massive",
    "died", "dead", "kill", "killed", "win", "won", "lost", "lose",
    "love", "hate", "amazing", "incredible", "impossible", "record",
}
_EMOJI_RULES = [
    (("money", "million", "billion", "dollars", "cash", "rich", "profit"), "💰"),
    (("crazy", "insane", "unbelievable", "shocking", "wild", "mind"), "🤯"),
    (("funny", "laugh", "hilarious", "joke", "haha"), "😂"),
    (("love", "heart", "beautiful", "amazing"), "❤️"),
    (("fire", "best", "incredible", "awesome", "epic"), "🔥"),
    (("died", "dead", "kill", "killed", "scary", "fear"), "😱"),
    (("win", "won", "record", "champion", "first"), "🏆"),
]


def _hex_to_ass_color(hex_color: str) -> str:
    """Convert #RRGGBB to ASS &H00BBGGRR& inline color override."""
    h = hex_color.lstrip("#")
    if len(h) != 6:
        return "&H0000D7FF&"  # gold fallback
    rr, gg, bb = h[0:2], h[2:4], h[4:6]
    return f"&H00{bb}{gg}{rr}&".upper()


def _emoji_for_line(raw_text: str) -> str:
    low = raw_text.lower()
    for keywords, emoji in _EMOJI_RULES:
        if any(k in low for k in keywords):
            return emoji
    return ""


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
    hook_title: str = "",
) -> Path:
    """
    Create an ASS subtitle file from word timestamps.
    Automatically adjusts PlayRes for vertical crop if enabled.
    When ``hook_title`` is set and ``config.show_hook_title`` is on, a bold
    title is burned near the top for the first ``config.hook_title_duration``s.
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

    title_font_size = int(font_size * 1.3)
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
        # Bold hook title anchored near the top (alignment 8 = top-center).
        (
            f"Style: HookTitle, Arial, {title_font_size}, &H0000FFFF, &H000000FF, "
            f"&H00000000, &H80000000, 1, 0, 0, 0, 100, 100, 0, 0, 1, "
            f"4.0, 2.0, 8, 60, 60, 120, 1"
        ),
        "",
        "[Events]",
        "Format: Layer, Start, End, Style, Name, MarginL, MarginR, "
        "MarginV, Effect, Text",
    ]

    if hook_title and config.show_hook_title:
        # Wrap the title so long lines don't overflow the frame edges
        # (WrapStyle 2 disables auto-wrapping, so we insert \N manually).
        title_wrap = max(12, int(max_chars * font_size / max(1, title_font_size)))
        wrapped = _wrap_title(hook_title, title_wrap, max_lines=3)
        title_text = (
            wrapped.replace("\\", "\\\\").replace("{", "\\{").replace("}", "\\}")
        )
        # Restore the intended ASS line break after escaping backslashes.
        title_text = title_text.replace("\\\\N", "\\N")
        title_end = _format_ass_time(max(0.5, config.hook_title_duration))
        lines.append(
            f"Dialogue: 1,0:00:00.00,{title_end},HookTitle,,0,0,0,,{title_text}"
        )

    # Group words into caption lines
    current_line: List[str] = []
    current_start: Optional[float] = None
    current_end: Optional[float] = None

    highlight = _hex_to_ass_color(config.caption_highlight_color)

    def _decorate(word: str) -> str:
        esc = word.replace("\\", "\\\\").replace("{", "\\{").replace("}", "\\}")
        if config.caption_emphasis:
            bare = "".join(c for c in word.lower() if c.isalnum())
            if bare in _EMPHASIS_WORDS:
                return f"{{\\c{highlight}}}{esc}{{\\c&H00FFFFFF&}}"
        return esc

    def flush_line():
        nonlocal current_line, current_start, current_end
        if not current_line:
            return ""
        raw_text = " ".join(current_line)
        line_text = " ".join(_decorate(w) for w in current_line)
        if config.caption_emojis:
            emoji = _emoji_for_line(raw_text)
            if emoji:
                line_text = f"{line_text} {emoji}"

        # current_start/current_end are already clip-relative (shifted below).
        start_ass = _format_ass_time(max(0, current_start or 0))
        end_ass = _format_ass_time(max(0, current_end or 0))

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


def _wrap_title(text: str, width: int, max_lines: int = 3) -> str:
    """Word-wrap a hook title to fit the frame, joined with ASS line breaks (\\N).

    Keeps at most ``max_lines`` lines; if the text is longer it is truncated
    with an ellipsis so the title never overflows the video edges.
    """
    words = text.split()
    if not words:
        return text
    lines: List[str] = []
    current = ""
    for word in words:
        candidate = f"{current} {word}".strip()
        if len(candidate) <= width or not current:
            current = candidate
        else:
            lines.append(current)
            current = word
            if len(lines) == max_lines:
                break
    if len(lines) < max_lines and current:
        lines.append(current)
    # If we ran out of line budget before consuming all words, add an ellipsis.
    consumed = sum(len(l.split()) for l in lines)
    if consumed < len(words):
        if lines:
            lines[-1] = lines[-1].rstrip(".") + "…"
    return "\\N".join(lines)


def _format_ass_time(seconds: float) -> str:
    """Convert seconds to ASS time format: H:MM:SS.cc"""
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    centiseconds = int((seconds % 1) * 100)
    return f"{hours}:{minutes:02d}:{secs:02d}.{centiseconds:02d}"


def _crop_filter(config: ClipperConfig, center_x: Optional[float] = None) -> str:
    """
    Build the crop expression for vertical output. When ``center_x`` (a 0..1
    fraction of the source width, e.g. the tracked speaker's face) is given, the
    crop window is centered on it and clamped to the frame; otherwise a plain
    center crop is used.
    """
    crop_w = f"ih*{config.crop_aspect_w}/{config.crop_aspect_h}"
    if center_x is None:
        x_expr = "(iw-ow)/2"
    else:
        cx = min(0.95, max(0.05, center_x))
        # Center the crop window on the subject, clamped to [0, iw-ow].
        x_expr = f"min(max(iw*{cx:.4f}-ow/2\\,0)\\,iw-ow)"
    return f"crop={crop_w}:ih:{x_expr}:0"


def _build_ffmpeg_filter(
    ass_path_str: str,
    config: ClipperConfig,
    center_x: Optional[float] = None,
) -> str:
    """
    Build the FFmpeg video filter chain.
    Handles subtitle burning and optional 9:16 vertical cropping (optionally
    reframed onto a tracked subject).
    """
    filters = []

    if config.vertical_crop:
        filters.append(_crop_filter(config, center_x))
        filters.append(f"scale={config.vertical_width}:{config.vertical_height}")
        logger.info(
            f"Vertical crop: {config.crop_aspect_w}:{config.crop_aspect_h} "
            f"→ {config.vertical_width}x{config.vertical_height}"
            + (f" (reframed x={center_x:.2f})" if center_x is not None else " (center)")
        )

    # Burn subtitles on top
    filters.append(f"ass={ass_path_str}")

    return ",".join(filters)


def _audio_filter(config: ClipperConfig) -> str:
    """Loudness-normalize speech to the target LUFS for platform-consistent volume."""
    if config.loudness_normalize:
        return f"loudnorm=I={config.target_lufts:.1f}:TP=-1.5:LRA=11"
    return ""


def cut_and_caption_clip(
    video_path: Path,
    output_path: Path,
    start_time: float,
    end_time: float,
    words: list,
    config: ClipperConfig,
    extra_padding: float = 0.3,
    hook_title: str = "",
) -> Path:
    """
    Cut a clip from the video and burn captions onto it.
    Supports optional 9:16 vertical cropping and a hook title overlay.
    """
    clip_start = max(0, start_time - extra_padding)
    clip_end = end_time + extra_padding

    # Filter words that fall within the clip
    clip_words = [
        w for w in words
        if w.end > clip_start and w.start < clip_end
    ]

    # Trim leading/trailing dead air so the clip opens on speech (hook-first).
    if config.trim_silence and clip_words:
        first_w = min(w.start for w in clip_words)
        last_w = max(w.end for w in clip_words)
        clip_start = max(clip_start, first_w - 0.15)
        clip_end = min(clip_end, last_w + 0.30)

    duration = clip_end - clip_start

    if not clip_words:
        logger.warning(f"No words in clip range {clip_start:.1f}-{clip_end:.1f}")
        return _cut_simple(video_path, output_path, clip_start, duration, config)

    # Create ASS subtitle file
    ass_path = output_path.with_suffix(".ass")
    create_ass_subtitle_file(
        clip_words, ass_path, config, shift_start=clip_start, hook_title=hook_title
    )

    # Active-speaker reframe: find the subject so the vertical crop follows them.
    center_x: Optional[float] = None
    if (
        config.vertical_crop
        and config.use_person_tracking
        and config.reframe_strategy in ("tracked", "smart")
    ):
        center_x = compute_subject_center_x(video_path, clip_start, duration)

    # Build filter chain (crop + subtitles)
    ass_path_str = str(ass_path).replace("\\", "/").replace(":", "\\\\:")
    filter_chain = _build_ffmpeg_filter(ass_path_str, config, center_x)
    af = _audio_filter(config)

    use_bgm = bool(
        config.bgm_enabled and config.bgm_path and Path(config.bgm_path).exists()
    )

    if use_bgm:
        speech_chain = af or "anull"
        filter_complex = (
            f"[0:v]{filter_chain}[v];"
            f"[0:a]{speech_chain}[sp];"
            f"[1:a]volume={config.bgm_volume}[bg];"
            f"[sp][bg]amix=inputs=2:duration=first:dropout_transition=0[a]"
        )
        cmd = [
            "ffmpeg", "-y",
            "-ss", str(clip_start), "-i", str(video_path), "-t", str(duration),
            "-stream_loop", "-1", "-i", str(config.bgm_path),
            "-filter_complex", filter_complex,
            "-map", "[v]", "-map", "[a]",
            "-c:v", "libx264", "-c:a", "aac",
            "-preset", "fast", "-crf", "23", "-shortest",
            str(output_path),
        ]
    else:
        cmd = [
            "ffmpeg", "-y",
            "-ss", str(clip_start), "-i", str(video_path), "-t", str(duration),
            "-vf", filter_chain,
            *(["-af", af] if af else []),
            "-c:v", "libx264", "-c:a", "aac",
            "-preset", "fast", "-crf", "23",
            str(output_path),
        ]

    logger.info(f"Cutting clip: {clip_start:.1f}s - {clip_end:.1f}s → {output_path}")
    result = subprocess.run(cmd, capture_output=True, text=True)

    if result.returncode != 0:
        logger.error(f"FFmpeg failed: {result.stderr[:500]}")
        logger.info("Retrying without subtitles...")
        return _cut_simple(video_path, output_path, clip_start, duration, config)

    if config.generate_thumbnail:
        try:
            thumb_path = output_path.with_name(output_path.stem + "_thumb.jpg")
            thumb_time = clip_start + min(1.5, duration / 2.0)
            generate_thumbnail(
                video_path, thumb_path, thumb_time, hook_title, config, center_x
            )
        except Exception as e:  # noqa: BLE001 - thumbnails must never fail the clip
            logger.warning(f"Thumbnail generation failed: {e}")

    return output_path


_FONT_CANDIDATES = [
    "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
    "/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf",
    "C:/Windows/Fonts/arialbd.ttf",
    "C:/Windows/Fonts/Arial.ttf",
    "/System/Library/Fonts/Supplemental/Arial Bold.ttf",
]


def _load_font(size: int):
    from PIL import ImageFont

    for path in _FONT_CANDIDATES:
        if Path(path).exists():
            try:
                return ImageFont.truetype(path, size)
            except Exception:  # noqa: BLE001 - fall through to next candidate
                continue
    return ImageFont.load_default()


def generate_thumbnail(
    source_video: Path,
    thumb_path: Path,
    at_seconds: float,
    title: str,
    config: ClipperConfig,
    center_x: Optional[float] = None,
) -> Optional[Path]:
    """
    Extract a clean (un-captioned) frame from the *source* video, reframe it to
    the target aspect, and overlay a bold outlined title to produce a
    click-worthy thumbnail. Uses the source so it never double-prints captions.
    """
    from PIL import Image, ImageDraw

    frame_path = thumb_path.with_name(thumb_path.stem + "_frame.png")

    vf_args: List[str] = []
    if config.vertical_crop:
        vf = f"{_crop_filter(config, center_x)},scale={config.vertical_width}:{config.vertical_height}"
        vf_args = ["-vf", vf]

    subprocess.run(
        [
            "ffmpeg", "-y", "-ss", str(at_seconds), "-i", str(source_video),
            *vf_args, "-frames:v", "1", str(frame_path),
        ],
        check=True, capture_output=True,
    )

    img = Image.open(frame_path).convert("RGB")
    w, h = img.size
    draw = ImageDraw.Draw(img)

    text = (title or "").strip().upper()
    if text:
        font = _load_font(max(28, int(w / 12)))
        # Word-wrap to fit within 90% of the width.
        max_w = w * 0.9
        words = text.split()
        lines: List[str] = []
        cur = ""
        for word in words:
            trial = f"{cur} {word}".strip()
            if draw.textlength(trial, font=font) <= max_w or not cur:
                cur = trial
            else:
                lines.append(cur)
                cur = word
        if cur:
            lines.append(cur)

        bbox = font.getbbox("Ag")
        line_h = (bbox[3] - bbox[1]) + 12
        y = int(h * 0.06)
        for line in lines:
            lw = draw.textlength(line, font=font)
            x = (w - lw) / 2
            draw.text(
                (x, y), line, font=font, fill=(255, 221, 0),
                stroke_width=max(3, int(w / 240)), stroke_fill=(0, 0, 0),
            )
            y += line_h

    img.save(thumb_path, "JPEG", quality=90)
    try:
        frame_path.unlink()
    except OSError:
        pass
    logger.info(f"Created thumbnail: {thumb_path}")
    return thumb_path


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
