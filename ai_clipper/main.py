"""
AI Video Clipper — Extract viral moments from long videos.

Usage:
    python -m ai_clipper.main video.mp4
    python -m ai_clipper.main "https://youtube.com/watch?v=..."
    python -m ai_clipper.main video.mp4 --top-n 3 --min-score 7.0
    python -m ai_clipper.main video.mp4 --whisper large-v3 --llm llama3.1:70b
"""

import argparse
import logging
import sys
from pathlib import Path
from uuid import uuid4

from .config import ClipperConfig
from .downloader import DEFAULT_DOWNLOAD_ROOT, prepare_download_dir, is_url, download_video, get_video_info
from .pipeline import AIClipper

logger = logging.getLogger(__name__)


def main():
    # Windows consoles default to cp1252, which can't encode the emoji in our
    # output. Force UTF-8 so status lines don't crash the run.
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8")
        except Exception:
            pass

    parser = argparse.ArgumentParser(
        description="AI Clipper — Extract viral moments from long videos",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Local file
  python -m ai_clipper.main my_video.mp4

  # YouTube / TikTok / any URL
  python -m ai_clipper.main "https://youtube.com/watch?v=dQw4w9WgXcQ"
  python -m ai_clipper.main "https://www.tiktok.com/@user/video/123"

  # Custom settings
  python -m ai_clipper.main video.mp4 --top-n 5 --min-score 7.0
  python -m ai_clipper.main video.mp4 --whisper medium --llm mistral:7b
  python -m ai_clipper.main video.mp4 --llm-provider openai --cloud-model gpt-4o
        """,
    )

    parser.add_argument(
        "video", type=str,
        help="Path to a local video file OR a URL (YouTube, TikTok, etc.)"
    )

    # Model options
    parser.add_argument(
        "--whisper", type=str, default="medium",
        choices=["tiny", "base", "small", "medium", "large-v2", "large-v3"],
        help="Whisper model size (default: medium)"
    )
    parser.add_argument(
        "--device", type=str, default="auto",
        choices=["auto", "cpu", "cuda"],
        help="Compute device (default: auto)"
    )
    parser.add_argument(
        "--llm-provider", type=str, default="bionic",
        choices=["bionic", "ollama", "openai"],
        help="LLM provider: bionic (LM Studio), ollama, or openai (cloud)"
    )
    parser.add_argument(
        "--llm", type=str, default="local-model",
        help="Model name. For Bionic: the name shown in LM Studio. For Ollama: e.g. llama3.1:8b"
    )
    parser.add_argument(
        "--cloud-model", type=str, default="gpt-4o-mini",
        help="Cloud fallback model (default: gpt-4o-mini)"
    )

    # Clip options
    parser.add_argument(
        "--top-n", type=int, default=5,
        help="Number of clips to extract (default: 5)"
    )
    parser.add_argument(
        "--min-score", type=float, default=60.0,
        help="Minimum virality score 0-100 (default: 60)"
    )
    parser.add_argument(
        "--min-duration", type=float, default=15.0,
        help="Minimum clip duration in seconds (default: 15)"
    )
    parser.add_argument(
        "--max-duration", type=float, default=90.0,
        help="Maximum clip duration in seconds (default: 90)"
    )

    # Download options
    parser.add_argument(
        "--max-resolution", type=int, default=1080,
        choices=[360, 480, 720, 1080, 2160],
        help="Max resolution when downloading from URL (default: 1080)"
    )
    parser.add_argument(
        "--keep-download", action="store_true",
        help="Keep the downloaded full video after clipping"
    )

    # Output
    parser.add_argument(
        "--output", type=str, default="./output_clips",
        help="Output directory (default: ./output_clips)"
    )
    parser.add_argument(
        "--caption-font-size", type=int, default=48,
        help="Caption font size (default: 48)"
    )
    parser.add_argument(
        "--caption-position", type=str, default="center",
        choices=["center", "bottom"],
        help="Caption position (default: center)"
    )

    # Vertical cropping (9:16 for TikTok/Reels/Shorts)
    parser.add_argument(
        "--vertical", action="store_true",
        help="Crop clips to vertical 9:16 format (TikTok/Reels/Shorts)"
    )
    parser.add_argument(
        "--vertical-resolution", type=str, default="1080x1920",
        help="Vertical output resolution WxH (default: 1080x1920)"
    )

    # Hashtag generation
    parser.add_argument(
        "--no-hashtags", action="store_true",
        help="Disable AI hashtag generation"
    )
    parser.add_argument(
        "--max-hashtags", type=int, default=10,
        help="Maximum hashtags per clip (default: 10)"
    )

    # Debug
    parser.add_argument(
        "--verbose", "-v", action="store_true",
        help="Enable verbose logging"
    )

    args = parser.parse_args()

    # Setup logging
    level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )

    # ---- Handle URL vs local file ----
    video_path = args.video
    download_dir = None
    should_cleanup = False

    if is_url(args.video):
        print(f"🔗 Detected URL. Fetching video info...")
        try:
            info = get_video_info(args.video)
            print(f"   📺 {info['title']}")
            print(f"   ⏱️  {info['duration'] // 60}m {info['duration'] % 60}s")
            print(f"   👤 {info['uploader']}")
        except Exception as e:
            logger.warning(f"Could not fetch info: {e}")

        download_dir = prepare_download_dir(f"aiclip_{uuid4().hex}")
        print(f"⬇️  Downloading video to E: drive: {download_dir}")
        video_path = str(download_video(
            args.video,
            download_dir,
            max_resolution=args.max_resolution,
        ))
        if not args.keep_download:
            should_cleanup = True

    # ---- Parse vertical resolution ----
    vert_parts = args.vertical_resolution.lower().split("x")
    if len(vert_parts) == 2:
        vert_w, vert_h = int(vert_parts[0]), int(vert_parts[1])
    else:
        vert_w, vert_h = 1080, 1920

    # ---- Build config ----
    config = ClipperConfig(
        whisper_model=args.whisper,
        device=args.device,
        llm_provider=args.llm_provider,
        llm_model=args.llm,
        cloud_model=args.cloud_model,
        num_clips=args.top_n,
        min_viral_score=args.min_score,
        min_clip_duration=args.min_duration,
        max_clip_duration=args.max_duration,
        output_dir=Path(args.output),
        caption_font_size=args.caption_font_size,
        caption_position=args.caption_position,
        target_aspect="9:16" if args.vertical else "original",
        vertical_width=vert_w,
        vertical_height=vert_h,
        generate_hashtags=not args.no_hashtags,
        max_hashtags=args.max_hashtags,
    )

    # ---- Run pipeline ----
    clipper = AIClipper(config)
    try:
        result = clipper.run(video_path)
        clips = result.clips
        print(f"\n✅ Produced {len(clips)} clips:")
        for c in clips:
            size_mb = c.stat().st_size / 1e6
            print(f"   📁 {c.name} ({size_mb:.1f} MB)")

        print(f"\n📂 All clips saved to: {config.output_dir.resolve()}")
        if result.gallery_path:
            print(f"🌐 Gallery: {result.gallery_path.resolve()}")
    except FileNotFoundError as e:
        print(f"❌ {e}", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"❌ Error: {e}", file=sys.stderr)
        if args.verbose:
            raise
        sys.exit(1)
    finally:
        # Cleanup downloaded video
        if should_cleanup and download_dir:
            import shutil
            try:
                shutil.rmtree(download_dir)
            except Exception:
                pass


if __name__ == "__main__":
    main()
