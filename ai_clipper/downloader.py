"""
Video downloader using yt-dlp.
Supports YouTube, TikTok, Instagram, Twitter/X, and 1000+ other sites.
"""

from __future__ import annotations

import logging
import os
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Optional

DEFAULT_DOWNLOAD_ROOT = Path(os.environ.get("AICLIPPER_DOWNLOAD_ROOT", "E:/ai_clipper_downloads"))


def _drive_available(drive_letter: str) -> bool:
    """Return True if the drive exists."""
    drive_root = Path(f"{drive_letter}\\")
    return drive_root.exists()


def _resolve_download_dir(output_dir: Path) -> Path:
    """Force downloads to E: when possible, with a fallback if E: is unavailable."""
    output_dir = output_dir.expanduser()
    if not output_dir.is_absolute():
        try:
            output_dir = output_dir.resolve()
        except Exception:
            pass

    requested_drive = output_dir.drive.upper()
    if requested_drive != "E:" and _drive_available("E:"):
        target_name = output_dir.name or "download"
        if target_name in (".", ""):
            target_name = "download"
        output_dir = DEFAULT_DOWNLOAD_ROOT / target_name
        logger.info(f"Redirecting download path to E: drive: {output_dir}")
    elif requested_drive == "E:" and not _drive_available("E:"):
        fallback = Path(tempfile.gettempdir()) / "ai_clipper_downloads"
        logger.warning(
            "E: drive is unavailable; falling back to temp downloads at %s",
            fallback,
        )
        output_dir = fallback
    return output_dir


def _find_ytdlp_command() -> list[str]:
    """Return the yt-dlp command, or fall back to python -m yt_dlp."""
    candidates = ["yt-dlp", "yt_dlp"]
    for name in candidates:
        if shutil.which(name):
            return [name]
    logger.warning(
        "Could not find yt-dlp executable on PATH. Falling back to python -m yt_dlp."
    )
    return [sys.executable, "-m", "yt_dlp"]


def prepare_download_dir(subfolder: str) -> Path:
    """Create a download folder under the resolved E: download root."""
    output_dir = _resolve_download_dir(DEFAULT_DOWNLOAD_ROOT / subfolder)
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir

logger = logging.getLogger(__name__)


def is_url(path: str) -> bool:
    """Check if a string is a URL (starts with http/https)."""
    return bool(re.match(r"^https?://", path))


def _find_downloaded_file(output_dir: Path, before_files: set[Path], preferred_format: str) -> Path | None:
    """Find the newly created downloaded file in the output directory."""
    candidates = [
        p for p in output_dir.rglob(f"*.{preferred_format}")
        if p not in before_files and p.is_file()
    ]
    if candidates:
        return max(candidates, key=lambda p: p.stat().st_mtime)

    # Fallback: any new file created by yt-dlp
    candidates = [
        p for p in output_dir.rglob("*")
        if p not in before_files and p.is_file() and not p.suffix.endswith("part")
    ]
    if candidates:
        return max(candidates, key=lambda p: p.stat().st_mtime)

    return None


def download_video(
    url: str,
    output_dir: Path,
    preferred_format: str = "mp4",
    max_resolution: int = 1080,
) -> Path:
    """
    Download a video from a URL using yt-dlp.

    Args:
        url: Video URL (YouTube, TikTok, etc.)
        output_dir: Directory to save the downloaded video
        preferred_format: Preferred container format
        max_resolution: Maximum video height (720, 1080, etc.)

    Returns:
        Path to the downloaded video file
    """
    output_dir = _resolve_download_dir(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    existing_files = set(output_dir.rglob("*"))

    # Output template: use video title + ID
    output_template = str(output_dir / "%(title).100s_%(id)s.%(ext)s")

    def _run_ytdlp(command):
        logger.info(f"Running yt-dlp: {' '.join(command)}")
        return subprocess.run(command, capture_output=True, text=True)

    has_ffmpeg = shutil.which("ffmpeg") is not None
    if has_ffmpeg:
        format_spec = "bestvideo+bestaudio/best"
        merge_args = ["--merge-output-format", preferred_format]
    else:
        logger.warning("ffmpeg not found, using best available combined format.")
        format_spec = "best"
        merge_args = []

    cmd = _find_ytdlp_command() + [
        "--format", format_spec,
        *merge_args,
        "--output", output_template,
        "--print", "filename",
        "--no-warnings",
        "--no-progress",
        url,
    ]

    logger.info(f"Downloading: {url}")
    logger.debug(f"Command: {' '.join(cmd)}")

    result = _run_ytdlp(cmd)

    if result.returncode != 0 or not result.stdout.strip():
        logger.warning("Primary yt-dlp format failed, trying fallback formats.")
        fallback_formats = [
            "best",
            "worst",
        ]
        for fmt in fallback_formats:
            cmd[cmd.index("--format") + 1] = fmt
            result = _run_ytdlp(cmd)
            if result.returncode == 0 and result.stdout.strip():
                break

    if result.returncode != 0:
        raise RuntimeError(
            f"yt-dlp failed to download: {result.stderr[:500]}"
        )

    # Try to parse the output filename from stdout first, then stderr.
    def _candidate_paths_from_line(line: str) -> list[Path]:
        raw = line.strip().strip('"')
        if "Destination:" in raw:
            raw = raw.split("Destination:", 1)[1].strip().strip('"')
        candidate = Path(raw)
        paths = []
        if candidate.is_absolute():
            paths.append(candidate)
        else:
            paths.append(output_dir / candidate)
            paths.append(output_dir / candidate.name)
        return paths

    downloaded_path = None
    output_lines = [line.strip() for line in result.stdout.splitlines() if line.strip()]
    if not output_lines and result.stderr:
        output_lines = [line.strip() for line in result.stderr.splitlines() if line.strip()]

    for line in reversed(output_lines):
        if not line:
            continue
        for candidate in _candidate_paths_from_line(line):
            if candidate.exists() and not candidate.suffix.endswith(".part"):
                downloaded_path = candidate
                break
        if downloaded_path:
            break

        # Search for any file sharing the same stem
        candidate = Path(line.strip().strip('"'))
        stem_matches = list(output_dir.rglob(f"{candidate.stem}*"))
        if stem_matches:
            downloaded_path = max(stem_matches, key=lambda p: p.stat().st_mtime)
            break

    if downloaded_path is None or not downloaded_path.exists():
        downloaded_path = _find_downloaded_file(output_dir, existing_files, preferred_format)

    if downloaded_path is None or not downloaded_path.exists():
        # As a last resort, search for the newest non-part file in the directory
        downloaded_path = _find_downloaded_file(output_dir, existing_files, preferred_format)

    if downloaded_path is None or not downloaded_path.exists():
        raise FileNotFoundError(
            f"Downloaded file not found in {output_dir}. yt-dlp output:\n{result.stdout}\n{result.stderr}"
        )

    logger.info(f"Downloaded: {downloaded_path.name} ({downloaded_path.stat().st_size / 1e6:.1f} MB)")
    return downloaded_path


def get_video_info(url: str) -> dict:
    """
    Get video metadata without downloading.

    Returns dict with: title, duration, uploader, view_count, etc.
    """
    cmd = _find_ytdlp_command() + [
        "--dump-json",
        "--no-playlist",
        url,
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(
            f"yt-dlp info failed: {result.stderr[:300]}"
        )

    import json
    info = json.loads(result.stdout.strip().split("\n")[0])
    return {
        "title": info.get("title", "Unknown"),
        "duration": info.get("duration", 0),
        "uploader": info.get("uploader", "Unknown"),
        "view_count": info.get("view_count", 0),
        "url": info.get("webpage_url", url),
    }
