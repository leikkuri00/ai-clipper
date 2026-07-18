from __future__ import annotations

import argparse
import logging
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Optional

# Ensure the local ai_clipper package is importable when run from the workspace root.
ROOT_DIR = Path(__file__).resolve().parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from ai_clipper.config import ClipperConfig
from ai_clipper.pipeline import AIClipper
from ai_clipper.downloader import (
    DEFAULT_DOWNLOAD_ROOT,
    download_video,
    get_video_info,
    is_url,
    prepare_download_dir,
)

DEFAULT_ROOT = Path("E:/clippingtool33")
DEFAULT_DOWNLOAD_ROOT = DEFAULT_ROOT / "downloads"
DEFAULT_OUTPUT_ROOT = DEFAULT_ROOT / "output"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("clippingtool33")


def prepare_dir(path: Path) -> Path:
    path = path.expanduser()
    if not path.is_absolute():
        try:
            path = path.resolve()
        except Exception:
            pass
    path.mkdir(parents=True, exist_ok=True)
    return path


def download_source(url: str, output_dir: Path, max_resolution: int = 1080) -> Path:
    output_dir = prepare_dir(output_dir)
    logger.info("Downloading source to %s", output_dir)
    video_path = download_video(url, output_dir, max_resolution=max_resolution)
    return video_path


def build_config(args: argparse.Namespace, output_dir: Path) -> ClipperConfig:
    return ClipperConfig(
        source_url=args.url or "",
        source_path=Path(args.input) if args.input else None,
        output_dir=output_dir,
        num_clips=args.num_clips,
        min_viral_score=args.min_score,
        target_clip_duration=args.target_duration,
        clip_duration_tolerance=args.tolerance,
        min_clip_duration=args.min_clip_duration,
        max_clip_duration=args.max_clip_duration,
        num_story_series=args.num_story_series,
        episodes_per_series=args.episodes_per_series,
        episode_target_duration=args.episode_duration,
        skip_story_series=args.skip_story_series,
        story_only=args.story_only,
        min_candidates=args.min_candidates,
        iterative_rounds=args.iterative_rounds,
        offline_mode=args.offline,
        skip_multimodal=args.skip_multimodal,
        host_binding=args.host,
        verbose=args.verbose,
        target_aspect=args.aspect,
        reframe_strategy=args.reframe,
        caption_style=args.caption_style,
        show_hook_title=not args.no_hook_title,
        caption_font_size=args.caption_font_size,
        transcribe_provider="local" if args.offline else args.transcribe,
        whisper_model=args.whisper_model,
        groq_api_key=args.groq_api_key or "",
        llm_provider=args.llm_provider,
        llm_model=args.llm_model,
        chunk_duration_sec=args.chunk_minutes * 60.0,
        bgm_enabled=args.bgm is not None,
        bgm_path=Path(args.bgm) if args.bgm else None,
        bgm_volume=args.bgm_volume,
        keep_download=args.keep_download,
        keep_intermediates=args.keep_intermediates,
        language=args.language,
        max_resolution=args.max_resolution,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="clippingtool33: unified AI Clipper launcher with E: download/output enforcement",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--url", type=str, help="Video URL to download")
    source.add_argument("--input", type=str, help="Local video file path")

    parser.add_argument("--output", type=str, default=str(DEFAULT_OUTPUT_ROOT), help="Output directory on E:")
    parser.add_argument("--download-dir", type=str, default=str(DEFAULT_DOWNLOAD_ROOT), help="Download directory on E:")
    parser.add_argument("--num-clips", type=int, default=8, help="Number of standalone clips")
    parser.add_argument("--min-score", type=float, default=60.0, help="Minimum viral score")
    parser.add_argument("--target-duration", type=float, default=148.0, help="Target clip duration in seconds")
    parser.add_argument("--tolerance", type=float, default=3.0, help="Duration tolerance in seconds")
    parser.add_argument("--min-clip-duration", type=float, default=30.0, help="Minimum clip duration")
    parser.add_argument("--max-clip-duration", type=float, default=180.0, help="Maximum clip duration")
    parser.add_argument("--num-story-series", type=int, default=3, help="Number of story series")
    parser.add_argument("--episodes-per-series", type=int, default=3, help="Episodes per story series")
    parser.add_argument("--episode-duration", type=float, default=120.0, help="Target episode duration")
    parser.add_argument("--skip-story-series", action="store_true", help="Skip story series generation")
    parser.add_argument("--story-only", action="store_true", help="Only generate story series")
    parser.add_argument("--min-candidates", type=int, default=300, help="Minimum number of candidates to generate")
    parser.add_argument("--iterative-rounds", type=int, default=3, help="Optimization rounds")
    parser.add_argument("--offline", action="store_true", help="Offline mode: use local Whisper and skip cloud calls")
    parser.add_argument("--skip-multimodal", action="store_true", help="Skip audio/video multimodal analysis")
    parser.add_argument("--host", type=str, default="", help="Bind web UI address (not used by default)")
    parser.add_argument("--verbose", action="store_true", help="Verbose logging")
    parser.add_argument("--aspect", type=str, default="9:16", choices=["9:16", "1:1", "4:5", "original"], help="Target aspect ratio")
    parser.add_argument("--reframe", type=str, default="tracked", choices=["center", "tracked", "smart", "split"], help="Reframe strategy")
    parser.add_argument("--caption-style", type=str, default="karaoke", choices=["karaoke", "subtitle", "none"], help="Caption style")
    parser.add_argument("--no-hook-title", action="store_true", help="Disable hook title on clips")
    parser.add_argument("--caption-font-size", type=int, default=48, help="Caption font size")
    parser.add_argument("--transcribe", type=str, default="groq", choices=["groq", "local"], help="Transcription provider")
    parser.add_argument("--whisper-model", type=str, default="large-v3-turbo", help="Local Whisper model")
    parser.add_argument("--groq-api-key", type=str, default="", help="Groq API key")
    parser.add_argument("--llm-provider", type=str, default="bionic", choices=["bionic", "ollama", "openai", "router"], help="LLM provider")
    parser.add_argument("--llm-model", type=str, default="local-model", help="LLM model name")
    parser.add_argument("--chunk-minutes", type=int, default=20, help="Chunk duration in minutes")
    parser.add_argument("--bgm", type=str, default=None, help="Optional background music file")
    parser.add_argument("--bgm-volume", type=float, default=0.15, help="BGM volume")
    parser.add_argument("--keep-download", action="store_true", help="Keep raw downloaded source files")
    parser.add_argument("--keep-intermediates", action="store_true", help="Keep intermediate files")
    parser.add_argument("--language", type=str, default="auto", help="Language code for transcription")
    parser.add_argument("--max-resolution", type=int, default=1080, help="Maximum download resolution")
    parser.add_argument("--debug", action="store_true", help="Enable debug logging")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.debug:
        logger.setLevel(logging.DEBUG)

    output_dir = prepare_dir(Path(args.output))
    download_dir = prepare_dir(Path(args.download_dir))

    if args.url:
        try:
            metadata = get_video_info(args.url)
            logger.info("Metadata: %s", metadata)
        except Exception as exc:
            logger.warning("Could not fetch metadata: %s", exc)

        source_path = download_source(args.url, download_dir / "url", max_resolution=args.max_resolution)
    else:
        source_path = Path(args.input).expanduser()
        if not source_path.exists():
            raise FileNotFoundError(f"Input file not found: {source_path}")
        logger.info("Using local input file: %s", source_path)

    config = build_config(args, output_dir)
    clipper = AIClipper(config)

    logger.info("Running AI Clipper pipeline...")
    result = clipper.run(source_path)

    logger.info("Pipeline finished: %d clips", len(result.clips))
    for clip in result.clips:
        logger.info("Clip: %s", clip)
    if result.gallery_path:
        logger.info("Gallery: %s", result.gallery_path)

    if args.url and not args.keep_download:
        logger.info("Cleaning up downloaded source files: %s", download_dir)
        shutil.rmtree(download_dir, ignore_errors=True)

    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        logger.exception("Unhandled error")
        raise
