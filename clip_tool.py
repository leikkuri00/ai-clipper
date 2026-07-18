#!/usr/bin/env python3
"""
AI Clipping Tool — Ultimate AI Video Clip Selection & Story Engine.

Usage:
    python clip_tool.py --url "https://youtube.com/watch?v=..."
    python clip_tool.py --input my_video.mp4 --num-clips 8
    python clip_tool.py --url "..." --skip-story-series --skip-multimodal
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
from pathlib import Path

# Ensure the ai_clipper package is importable
sys.path.insert(0, str(Path(__file__).resolve().parent))

from ai_clipper.config import ClipperConfig
from ai_clipper.pipeline import AIClipper

logger = logging.getLogger("clip_tool")


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="AI Clipping Tool — paste a URL, get viral short clips",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python clip_tool.py --url "https://youtube.com/watch?v=dQw4w9WgXcQ"
  python clip_tool.py --input podcast.mp4 --num-clips 8
  python clip_tool.py --url "..." --skip-story-series
  python clip_tool.py --url "..." --target-duration 148 --tolerance 3
        """,
    )

    # ── Source ─────────────────────────────────────────────
    src = p.add_mutually_exclusive_group(required=True)
    src.add_argument("--url", type=str, help="Video URL (YouTube, TikTok, Instagram, etc.)")
    src.add_argument("--input", type=str, help="Path to local video file")

    # ── Output ─────────────────────────────────────────────
    p.add_argument("--output", type=str, default="./output", help="Output directory (default: ./output)")
    p.add_argument("--num-clips", type=int, default=8, help="Number of standalone clips (default: 8)")
    p.add_argument("--min-score", type=float, default=60.0, help="Minimum overall score 0-100 (default: 60)")
    p.add_argument("--target-duration", type=float, default=148.0, help="Target clip duration in seconds (default: 148 = 2:28)")
    p.add_argument("--tolerance", type=float, default=3.0, help="Duration tolerance in seconds (default: 3)")
    p.add_argument("--min-clip-duration", type=float, default=30.0, help="Minimum clip seconds (default: 30)")
    p.add_argument("--max-clip-duration", type=float, default=180.0, help="Maximum clip seconds (default: 180)")

    # ── Story Series ─────────────────────────────────────
    p.add_argument("--num-story-series", type=int, default=3, help="Number of story series (default: 3)")
    p.add_argument("--episodes-per-series", type=int, default=3, help="Episodes per story series (default: 3)")
    p.add_argument("--episode-duration", type=float, default=120.0, help="Target episode duration in seconds (default: 120)")
    p.add_argument("--skip-story-series", action="store_true", help="Skip story series generation")
    p.add_argument("--story-only", action="store_true", help="Only generate story series, no standalone clips")

    # ── Candidate Generation ─────────────────────────────
    p.add_argument("--min-candidates", type=int, default=300, help="Minimum candidates to generate (default: 300)")
    p.add_argument("--iterative-rounds", type=int, default=3, help="Optimization rounds (default: 3)")

    # ── Mode flags ─────────────────────────────────────────
    p.add_argument("--offline", action="store_true", help="Offline mode: use local Whisper, skip cloud calls")
    p.add_argument("--skip-multimodal", action="store_true", help="Skip audio/video multimodal analysis")
    p.add_argument("--host", type=str, default="", help="Bind web UI to address (e.g. 0.0.0.0 for LAN)")
    p.add_argument("--verbose", "-v", action="store_true", help="Verbose logging")

    # ── Aspect & reframing ─────────────────────────────────
    p.add_argument("--aspect", type=str, default="9:16",
                   choices=["9:16", "1:1", "4:5", "original"],
                   help="Target aspect ratio (default: 9:16)")
    p.add_argument("--reframe", type=str, default="tracked",
                   choices=["center", "tracked", "smart"],
                   help="Reframe strategy")

    # ── Captions ──────────────────────────────────────────
    p.add_argument("--caption-style", type=str, default="karaoke",
                   choices=["karaoke", "subtitle", "none"],
                   help="Caption style (default: karaoke word-highlight)")
    p.add_argument("--no-hook-title", action="store_true", help="Disable hook-line on-screen title")
    p.add_argument("--caption-font-size", type=int, default=48, help="Caption font size")

    # ── Transcription ──────────────────────────────────────
    p.add_argument("--transcribe", type=str, default="groq",
                   choices=["groq", "local"],
                   help="Transcription provider")
    p.add_argument("--whisper-model", type=str, default="large-v3-turbo", help="Local Whisper model")
    p.add_argument("--groq-api-key", type=str, default="", help="Groq API key")

    # ── LLM scoring ────────────────────────────────────────
    p.add_argument("--llm-provider", type=str, default="bionic",
                   choices=["bionic", "ollama", "openai", "router"],
                   help="LLM provider")
    p.add_argument("--llm-model", type=str, default="local-model", help="LLM model name")
    p.add_argument("--chunk-minutes", type=int, default=20, help="Chunk duration in minutes")

    # ── Audio ──────────────────────────────────────────────
    p.add_argument("--bgm", type=str, default=None, help="Path to background music file")
    p.add_argument("--bgm-volume", type=float, default=0.15, help="BGM volume relative to speech")

    # ── Misc ───────────────────────────────────────────────
    p.add_argument("--keep-download", action="store_true", help="Keep raw download after processing")
    p.add_argument("--keep-intermediates", action="store_true", help="Keep intermediate files")
    p.add_argument("--language", type=str, default="auto", help="Force language code")
    p.add_argument("--max-resolution", type=int, default=1080, choices=[360, 480, 720, 1080])

    return p


def main():
    # Windows consoles default to cp1252, force UTF-8
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8")
        except Exception:
            pass

    parser = build_parser()
    args = parser.parse_args()

    # ── Logging ────────────────────────────────────────────
    level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)-7s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )

    # ── Build config from args ─────────────────────────────
    config = ClipperConfig(
        source_url=args.url or "",
        source_path=Path(args.input) if args.input else None,
        output_dir=Path(args.output),
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
        groq_api_key=args.groq_api_key or os.environ.get("GROQ_API_KEY", ""),
        llm_provider=args.llm_provider,
        llm_model=args.llm_model,
        chunk_duration_sec=args.chunk_minutes * 60.0,
        bgm_path=Path(args.bgm) if args.bgm else None,
        bgm_volume=args.bgm_volume,
        bgm_enabled=args.bgm is not None,
        keep_download=args.keep_download,
        keep_intermediates=args.keep_intermediates,
        language=args.language,
        max_resolution=args.max_resolution,
    )

    logger.info("╔══════════════════════════════════════════════════════════╗")
    logger.info("║       ✂️  AI Clipping Tool — Master Prompt Engine v1.0     ║")
    logger.info("╚══════════════════════════════════════════════════════════╝")
    logger.info(f"  Source:       {'URL: ' + config.source_url if config.source_url else 'File: ' + str(config.source_path)}")
    logger.info(f"  Output:       {config.output_dir.resolve()}")
    logger.info(f"  Standalone:   {config.num_clips} clips @ {config.target_clip_duration}s ±{config.clip_duration_tolerance}s")
    logger.info(f"  Story Series: {config.num_story_series if not config.skip_story_series else 0} series × {config.episodes_per_series} episodes")
    logger.info(f"  Min score:    {config.min_viral_score}/100")
    logger.info(f"  Aspect:       {config.target_aspect}")
    logger.info(f"  Captions:     {config.caption_style}")
    logger.info(f"  Transcribe:   {config.transcribe_provider}")
    logger.info(f"  LLM:          {config.llm_provider}/{config.llm_model}")
    logger.info(f"  Offline:      {config.offline_mode}")
    logger.info(f"  Multimodal:   {'off' if config.skip_multimodal else 'on'}")

    # ── Run ────────────────────────────────────────────────
    try:
        clipper = AIClipper(config)
        result = clipper.run()

        if result and result.clips:
            print("\n" + "=" * 60)
            print(f"  ✅  Done!  {len(result.clips)} clips ready.")
            print(f"  📁  {config.output_dir.resolve()}")
            print(f"  🌐  Open: {result.gallery_path}")
            standalone_count = sum(1 for c in result.report.get('candidates', []) if c.get('type') == 'standalone')
            series_count = sum(1 for c in result.report.get('candidates', []) if c.get('type') == 'story_series')
            print(f"  🎬  Standalone clips: {standalone_count}")
            print(f"  📖  Story series episodes: {series_count}")
            print("=" * 60)
        else:
            print("\n⚠️  No clips were produced. Try lowering --min-score or --num-clips.")
            sys.exit(1)

    except Exception as e:
        logger.error(f"Pipeline failed: {e}", exc_info=args.verbose)
        print(f"\n❌  Error: {e}")
        print("   Run with --verbose for full traceback.")
        sys.exit(1)


if __name__ == "__main__":
    main()
