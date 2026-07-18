"""
AI Clipper — Web UI (Master Prompt Edition)
Paste a video link or upload a file → AI finds the best clips and story series.

Run:  streamlit run ai_clipper/web_app.py
"""

from __future__ import annotations

import logging
import io
import os
import sys
import tempfile
import time
import zipfile
from pathlib import Path
from urllib.error import URLError
from urllib.request import urlopen
from uuid import uuid4

# Ensure the repo root is importable when launched via `streamlit run
# ai_clipper/web_app.py`, which puts the script's own directory (not the repo
# root) on sys.path and would otherwise break `import ai_clipper`.
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

import streamlit as st

from ai_clipper.config import ClipperConfig
from ai_clipper.downloader import DEFAULT_DOWNLOAD_ROOT, prepare_download_dir, is_url, download_video, get_video_info
from ai_clipper.pipeline import AIClipper

# --- Page config ---
st.set_page_config(
    page_title="AI Clipper — Master Engine",
    page_icon="✂️",
    layout="centered",
)

st.title("✂️ AI Video Clipper — Master Engine")
st.caption("Paste a video link → AI finds the most viral moments + story series → Download clips with captions")


def _lm_studio_is_running() -> bool:
    """Return whether LM Studio's local OpenAI-compatible server is reachable."""
    try:
        with urlopen("http://127.0.0.1:1234/v1/models", timeout=1):
            return True
    except (URLError, OSError):
        return False


def _download_all_clips(clips: list[Path], key: str) -> None:
    """Offer all generated clips and their hashtag files as one ZIP download."""
    archive = io.BytesIO()
    with zipfile.ZipFile(archive, "w", zipfile.ZIP_DEFLATED) as zip_file:
        for clip in clips:
            zip_file.write(clip, arcname=clip.name)
            hashtags = clip.with_suffix(".txt")
            if hashtags.exists():
                zip_file.write(hashtags, arcname=hashtags.name)

    st.download_button(
        "Download all clips (.zip)",
        data=archive.getvalue(),
        file_name="ai_clipper_results.zip",
        mime="application/zip",
        key=key,
        use_container_width=True,
    )


# --- Sidebar settings ---
with st.sidebar:
    st.header("⚙️ Settings")

    whisper_model = st.selectbox(
        "Whisper model",
        ["tiny", "base", "small", "medium", "large-v2", "large-v3", "large-v3-turbo"],
        index=6,
        help="Larger = better transcription, slower.",
    )

    llm_provider = st.selectbox(
        "LLM provider",
        ["local (llama-cpp)", "openai"],
        index=0,
        help="Choose local llama-cpp or OpenAI provider.",
    )

    if "local" in llm_provider:
        llm_provider_key = "local"
        llm_model = st.text_input(
            "Local model path",
            os.environ.get("LLAMA_MODEL_PATH", "C:\\models\\orca-mini-3b.gguf"),
            help="Path to a local GGUF model file for llama-cpp-python.",
        )
        cloud_model = ""
        st.caption("Local model path must point to a downloaded GGUF model.")
    else:
        llm_provider_key = "openai"
        llm_model = st.text_input("OpenAI model", "gpt-4o-mini")
        cloud_model = st.text_input("Cloud model", "gpt-4o-mini")
        st.warning("⚠️ Requires OPENAI_API_KEY environment variable")

    st.divider()

    st.subheader("🎬 Standalone Clips")
    top_n = st.slider("Number of standalone clips", 1, 12, 8)
    target_duration = st.slider("Target clip duration (sec)", 60, 240, 148, 5)
    tolerance = st.slider("Duration tolerance (sec)", 0, 10, 3)
    min_score = st.slider("Minimum overall score", 0, 100, 60, 5)

    st.subheader("📖 Story Series")
    enable_story_series = st.checkbox("Generate story series", value=True)
    num_story_series = st.slider(
        "Number of story series", 0, 5, 1,
        help="Each series = 3 chronological episodes that together tell one full story.",
    )
    episode_duration = st.slider("Episode target duration (sec)", 60, 240, 148, 5)

    st.divider()

    st.subheader("🧠 AI Engine")
    min_candidates = st.slider("Minimum candidates", 100, 1000, 300, 50)
    iterative_rounds = st.slider("Optimization rounds", 0, 5, 3)
    skip_multimodal = st.checkbox("Skip multimodal analysis (faster)", value=False)

    st.divider()

    st.subheader("📱 Output format")
    caption_style = st.selectbox(
        "Caption style", ["center (TikTok-style)", "bottom (YouTube-style)"], index=0
    )
    caption_position = "center" if "center" in caption_style else "bottom"

    vertical_crop = st.checkbox(
        "Crop to 9:16 vertical (TikTok/Reels/Shorts)",
        value=True,
        help="Crops horizontal video to vertical portrait format"
    )

    max_resolution = st.selectbox(
        "Max resolution", [360, 480, 720, 1080], index=3
    )

    st.divider()
    st.subheader("🔑 Download cookies")
    st.caption(
        "Needed for YouTube 'sign in to confirm you're not a bot' errors and "
        "private/members-only videos."
    )
    cookies_from_browser = st.selectbox(
        "Use cookies from browser",
        ["(none)", "chrome", "edge", "firefox", "brave", "chromium", "opera", "vivaldi", "safari"],
        index=0,
        help="Reads cookies from a browser you're signed into. Requires that browser installed locally.",
    )
    cookies_from_browser = "" if cookies_from_browser == "(none)" else cookies_from_browser
    cookiefile = st.text_input(
        "Or cookies.txt path",
        os.environ.get("AICLIPPER_COOKIEFILE", ""),
        help="Path to an exported Netscape-format cookies.txt (takes priority over the browser option).",
    )

    st.divider()
    st.subheader("🏷️ Hashtags")
    generate_hashtags = st.checkbox(
        "Generate AI hashtags", value=True,
        help="LLM generates optimized viral hashtags for each clip"
    )
    max_hashtags = st.slider("Max hashtags per clip", 3, 20, 10)


def _build_config() -> ClipperConfig:
    """Build ClipperConfig from sidebar settings."""
    return ClipperConfig(
        whisper_model=whisper_model,
        llm_provider=llm_provider_key,
        llm_model=llm_model,
        cloud_model=cloud_model,
        num_clips=top_n,
        target_clip_duration=float(target_duration),
        clip_duration_tolerance=float(tolerance),
        min_viral_score=float(min_score),
        num_story_series=num_story_series if enable_story_series else 0,
        episode_target_duration=float(episode_duration),
        skip_story_series=not enable_story_series,
        min_candidates=min_candidates,
        iterative_rounds=iterative_rounds,
        skip_multimodal=skip_multimodal,
        output_dir=Path("./output_clips"),
        caption_position=caption_position,
        target_aspect="9:16" if vertical_crop else "original",
        generate_hashtags=generate_hashtags,
        max_hashtags=max_hashtags,
        max_resolution=max_resolution,
        cookies_from_browser=cookies_from_browser,
        cookiefile=cookiefile,
    )


def _display_results(result, progress):
    """Display clips and story series in the Streamlit UI."""
    clips = result.clips
    report = result.report
    story_series = result.story_series

    progress.update(label="✅ Complete!", state="complete")

    if not clips:
        st.warning("No clips produced. Try lowering the minimum score.")
        return

    st.success(f"🎉 Found {len(clips)} clips!")

    # Standalone clips
    standalone = [c for c in report.get("candidates", []) if c.get("type") == "standalone"]
    if standalone:
        st.subheader("🎬 Standalone Clips")
        clip_files = [c for c in clips if "standalone" in c.name]
        for i, meta in enumerate(standalone):
            if i >= len(clip_files):
                continue
            clip_path = clip_files[i]

            with st.container(border=True):
                col1, col2 = st.columns([3, 1])
                with col1:
                    st.video(str(clip_path))
                with col2:
                    st.metric("Clip", f"#{meta.get('rank', i+1)}")
                    st.metric("Score", f"{meta.get('score', 0):.0f}/100")
                    size_mb = clip_path.stat().st_size / 1e6
                    st.metric("Size", f"{size_mb:.1f} MB")
                    if meta.get("title"):
                        st.caption(f"📝 {meta['title']}")
                    if meta.get("retention"):
                        r = meta["retention"]
                        st.caption(
                            f"▶ {r.get('completion_rate', 0)*100:.0f}% completion · "
                            f"↗ {r.get('share_likelihood', 0)*100:.0f}% share · "
                            f"💬 {r.get('comment_likelihood', 0)*100:.0f}% comment"
                        )
                    hashtag_path = clip_path.with_suffix(".txt")
                    if hashtag_path.exists():
                        ht_content = hashtag_path.read_text(encoding="utf-8")
                        lines = ht_content.strip().split("\n")
                        if len(lines) > 2:
                            st.code(lines[-1], language=None)
                    with open(clip_path, "rb") as f:
                        st.download_button(
                            "⬇️ Download",
                            f.read(),
                            file_name=clip_path.name,
                            mime="video/mp4",
                            key=f"dl_{i}",
                            use_container_width=True,
                        )

    # Story series
    if story_series:
        st.subheader("📖 Story Series")
        for series_idx, series in enumerate(story_series, 1):
            with st.expander(f"{series.title} — {series.overall_story_score:.0f}/100"):
                for ep_idx, key in enumerate(["episode_1", "episode_2", "episode_3"], 1):
                    ep = getattr(series, key)
                    # Find matching clip
                    clip_path = next((c for c in clips if f"series{series_idx}_ep{ep_idx}" in c.name), None)
                    cols = st.columns([2, 1])
                    with cols[0]:
                        st.markdown(f"**{ep_idx}. {ep.title}**")
                        st.caption(f"{format_time(ep.start_time)} – {format_time(ep.end_time)} · {format_time(ep.duration)}")
                        if ep_idx < 3 and ep.cliffhanger:
                            st.caption(f"🪝 {ep.cliffhanger}")
                        elif ep.ending:
                            st.caption(f"🏁 {ep.ending}")
                    with cols[1]:
                        if clip_path:
                            with open(clip_path, "rb") as f:
                                st.download_button(
                                    "⬇️",
                                    f.read(),
                                    file_name=clip_path.name,
                                    mime="video/mp4",
                                    key=f"dl_s{series_idx}_e{ep_idx}",
                                )
                    if clip_path:
                        st.video(str(clip_path))

    _download_all_clips(clips, key="download_all")


def format_time(seconds: float) -> str:
    """Format seconds as M:SS."""
    m = int(seconds // 60)
    s = int(seconds % 60)
    return f"{m}:{s:02d}"


def _process_url(url: str):
    """Download from URL and run pipeline."""
    progress = st.status("Processing...", expanded=True)

    config = _build_config()
    try:
        progress.write("🔍 Fetching video info...")
        info = get_video_info(
            url,
            cookies_from_browser=config.cookies_from_browser,
            cookiefile=config.cookiefile,
        )
        st.info(
            f"**{info['title']}**  \n"
            f"⏱️ {info['duration'] // 60}m {info['duration'] % 60}s  ·  "
            f"👤 {info['uploader']}"
        )

        progress.write("⬇️ Downloading video to E: drive...")
        download_dir = prepare_download_dir(f"aiclip_{uuid4().hex}")
        video_path = download_video(
            url,
            download_dir,
            max_resolution=max_resolution,
            cookies_from_browser=config.cookies_from_browser,
            cookiefile=config.cookiefile,
        )
        progress.write(f"✅ Downloaded: {video_path.name}")

        progress.write("🧠 Running Master Prompt AI engine...")
        config.output_dir.mkdir(parents=True, exist_ok=True)

        clipper = AIClipper(config)
        result = clipper.run(str(video_path))

        _display_results(result, progress)

        import shutil
        shutil.rmtree(download_dir, ignore_errors=True)

    except Exception as e:
        progress.update(label="❌ Failed", state="error")
        error_msg = str(e).lower()
        if "ollama" in error_msg or "connection" in error_msg:
            st.error(f"**LLM connection error.** Make sure your LLM server is running.\n\nDetails: {e}")
        elif "yt-dlp" in error_msg or "download" in error_msg:
            st.error(f"**Download failed.** The video may be private, age-restricted, or unavailable.\n\nDetails: {e}")
        elif "ffmpeg" in error_msg:
            st.error(f"**FFmpeg not found.** Install FFmpeg and make sure it's on your PATH.\n\nDetails: {e}")
        else:
            st.error(f"**Error:** {e}")
        import traceback
        st.code(traceback.format_exc())


def _process_upload(uploaded_file):
    """Process an uploaded file."""
    progress = st.status("Processing...", expanded=True)

    try:
        temp_dir = prepare_download_dir(f"aiclip_upload_{uuid4().hex}")
        video_path = temp_dir / uploaded_file.name
        video_path.write_bytes(uploaded_file.read())
        progress.write(f"✅ Saved: {video_path.name} ({video_path.stat().st_size / 1e6:.1f} MB)")

        progress.write("🧠 Running Master Prompt AI engine...")
        config = _build_config()
        config.output_dir.mkdir(parents=True, exist_ok=True)

        clipper = AIClipper(config)
        result = clipper.run(str(video_path))

        _display_results(result, progress)

        import shutil
        shutil.rmtree(temp_dir, ignore_errors=True)

    except Exception as e:
        progress.update(label="❌ Failed", state="error")
        st.error(f"**Error:** {e}")
        import traceback
        st.code(traceback.format_exc())


# --- Main UI ---
tab1, tab2 = st.tabs(["🔗 From URL", "📁 Local File"])

with tab1:
    url = st.text_input(
        "Paste video URL",
        placeholder="https://www.youtube.com/watch?v=... or TikTok, Instagram, etc.",
    )

    if url and st.button("🔍 Analyze & Clip", type="primary", use_container_width=True):
        if not is_url(url):
            st.error("Please enter a valid URL starting with http:// or https://")
        else:
            _process_url(url)

with tab2:
    uploaded_file = st.file_uploader(
        "Upload a video file",
        type=["mp4", "mov", "avi", "mkv", "webm"],
        help="Max 500MB (Streamlit limit)",
    )

    if uploaded_file and st.button("🔍 Analyze & Clip", type="primary", use_container_width=True, key="local_btn"):
        _process_upload(uploaded_file)


# --- Footer ---
st.divider()
st.caption("🧠 Uses Whisper for transcription + local/cloud LLM for the Master Prompt engine. Everything runs on your machine.")
