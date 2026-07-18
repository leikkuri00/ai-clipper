"""clipping28.py

Single-file wrapper for AI Clipper that downloads all URL sources to E: and runs the pipeline.
"""

from __future__ import annotations

import argparse
import json
import logging
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Optional

from ai_clipper.config import ClipperConfig
from ai_clipper.pipeline import AIClipper

DEFAULT_ROOT = Path("E:/clipping28")
DEFAULT_DOWNLOAD_ROOT = DEFAULT_ROOT / "downloads"
DEFAULT_OUTPUT_ROOT = DEFAULT_ROOT / "output"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("clipping28")


def _find_yt_dlp_command() -> list[str]:
    candidates = ["yt-dlp", "yt_dlp"]
    for name in candidates:
        if shutil.which(name):
            return [name]
    logger.warning("yt-dlp not found on PATH, falling back to python -m yt_dlp")
    return [sys.executable, "-m", "yt_dlp"]


def prepare_dir(path: Path) -> Path:
    path = path.expanduser()
    if not path.is_absolute():
        try:
            path = path.resolve()
        except Exception:
            pass
    path.mkdir(parents=True, exist_ok=True)
    return path


def download_video(
    url: str,
    output_dir: Path,
    preferred_format: str = "mp4",
    max_resolution: int = 1080,
) -> Path:
    output_dir = prepare_dir(output_dir)
    existing_files = set(output_dir.rglob("*"))

    output_template = str(output_dir / "%(title).100s_%(id)s.%(ext)s")

    has_ffmpeg = shutil.which("ffmpeg") is not None
    if has_ffmpeg:
        format_spec = "bestvideo+bestaudio/best"
        merge_args = ["--merge-output-format", preferred_format]
    else:
        logger.warning("ffmpeg not found; using best combined format.")
        format_spec = "best"
        merge_args = []

    cmd = _find_yt_dlp_command() + [
        "--format", format_spec,
        *merge_args,
        "--output", output_template,
        "--print", "filename",
        "--no-warnings",
        "--no-progress",
        url,
    ]

    logger.info("Running yt-dlp...")
    logger.debug("Command: %s", " ".join(cmd))
    result = subprocess.run(cmd, capture_output=True, text=True)
    logger.debug("yt-dlp stdout=%s", result.stdout)
    logger.debug("yt-dlp stderr=%s", result.stderr)

    if result.returncode != 0:
        raise RuntimeError(f"yt-dlp download failed: {result.stderr.strip()}")

    filename = None
    for line in reversed(result.stdout.splitlines()):
        line = line.strip().strip('"')
        if not line:
            continue
        if line.startswith("Destination:"):
            line = line.split("Destination:", 1)[1].strip().strip('"')
        candidate = Path(line)
        if not candidate.is_absolute():
            candidate = output_dir / candidate
        if candidate.exists() and candidate.is_file() and not candidate.suffix.endswith(".part"):
            filename = candidate
            break

    if filename is None:
        candidates = [
            p for p in output_dir.rglob("*")
            if p.is_file() and p not in existing_files and not p.suffix.endswith(".part")
        ]
        if candidates:
            filename = max(candidates, key=lambda p: p.stat().st_mtime)

    if filename is None or not filename.exists():
        raise FileNotFoundError(
            f"Downloaded file not found in {output_dir}.\nstdout={result.stdout}\nstderr={result.stderr}"
        )

    logger.info("Downloaded %s", filename)
    return filename


def get_video_info(url: str) -> dict:
    cmd = ["yt-dlp", "--dump-json", url]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"yt-dlp info failed: {result.stderr.strip()}")
    info = json.loads(result.stdout.splitlines()[0])
    return {
        "title": info.get("title", "Unknown"),
        "duration": info.get("duration", 0),
        "uploader": info.get("uploader", "Unknown"),
        "url": info.get("webpage_url", url),
    }


def build_config(output_dir: Path) -> ClipperConfig:
    return ClipperConfig(
        output_dir=output_dir,
        keep_download=True,
        keep_intermediates=False,
        max_resolution=1080,
    )


def run_pipeline(video_path: Path, output_dir: Path) -> list[Path]:
    config = build_config(output_dir)
    clipper = AIClipper(config)
    result = clipper.run(video_path)
    return result.clips


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="clipping28: download to E: and run AI Clipper in one file")
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--url", type=str, help="Video URL to download")
    source.add_argument("--input", type=str, help="Local video file path")
    parser.add_argument("--output", type=str, default=str(DEFAULT_OUTPUT_ROOT), help="Output directory on E:")
    parser.add_argument("--download-dir", type=str, default=str(DEFAULT_DOWNLOAD_ROOT), help="Download directory on E:")
    parser.add_argument("--keep-download", action="store_true", help="Keep raw download files")
    parser.add_argument("--debug", action="store_true", help="Enable debug logging")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.debug:
        logger.setLevel(logging.DEBUG)

    output_dir = prepare_dir(Path(args.output))
    download_dir = prepare_dir(Path(args.download_dir))

    if args.url:
        logger.info("Fetching video info for %s", args.url)
        try:
            info = get_video_info(args.url)
            logger.info("Title: %s", info["title"])
            logger.info("Uploader: %s", info["uploader"])
            logger.info("Duration: %s seconds", info["duration"])
        except Exception as exc:
            logger.warning("Could not fetch metadata: %s", exc)

        download_subdir = download_dir / "url"
        download_subdir = prepare_dir(download_subdir)
        video_path = download_video(args.url, download_subdir)
    else:
        video_path = Path(args.input).expanduser()
        if not video_path.exists():
            raise FileNotFoundError(f"Input file not found: {video_path}")
        logger.info("Using local file: %s", video_path)

    logger.info("Running AI Clipper pipeline...")
    clips = run_pipeline(video_path, output_dir)
    logger.info("Pipeline produced %d clips", len(clips))
    for clip in clips:
        logger.info("Clip: %s", clip)

    if not args.keep_download and args.url:
        logger.info("Keeping downloads disabled. Raw download folder: %s", download_subdir)

    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as err:
        logger.exception(err)
        raise
