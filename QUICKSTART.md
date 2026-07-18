# AI Clipper — Quick Start

Turn any long video (YouTube URL or local file) into the most viral vertical
clips with burned captions, hook titles, face-tracking, thumbnails and a
chronological story series.

## One-click launch (Windows)

1. **Double-click `Start AI Clipper.bat`.**
   - On the first run it automatically creates a private Python environment,
     installs everything, and (if needed) installs FFmpeg via `winget`.
   - This first setup takes a few minutes. Later launches are instant.
2. Your browser opens at `http://localhost:8501`.
3. Paste a YouTube URL (or upload a file), pick your options, click **Run**.

> Requirements the launcher checks for you: **Python 3.10+** (install from
> https://www.python.org/downloads/ and tick *Add Python to PATH*) and
> **FFmpeg** (auto-installed via winget when possible).

## Local LLM (no API key needed)

Scoring which moments are most viral runs on a **local** model:

- **LM Studio** (recommended): open LM Studio → Developer → **Start Server**,
  load a model, then in the app sidebar pick **LM Studio (local)**.
- **Ollama**: `ollama serve` + `ollama pull llama3.1`, then pick **Ollama (local)**.

## Downloading private / age-restricted / bot-checked YouTube videos

In the sidebar under **Download cookies**, either:

- choose the browser you're signed into YouTube with (reads its cookies), or
- point to an exported `cookies.txt`.

## What you get per run

- Standalone viral clips (~2:28 each) — 9:16, burned word-level captions with
  highlighted keywords + emojis, a bold hook title, face-tracked framing,
  loudness-normalized audio, and an auto thumbnail.
- One chronological 3-episode story series that tells the full story.
- A report + gallery, plus a one-click "Download all clips (.zip)".

## Viral polish toggles (sidebar → ✨ Viral polish)

Face-tracking reframe, dead-air trim (hook-first), keyword/emoji captions,
per-clip thumbnails, loudness normalization, optional background-music bed, and
platform presets (TikTok / Reels / Shorts / Square / YouTube).
