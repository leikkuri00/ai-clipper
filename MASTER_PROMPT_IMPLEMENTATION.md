# Master Prompt Implementation — AI Clipper v1.0

The full Master Prompt pipeline has been implemented into the existing `ai_clipper` codebase.

## What Changed

### New Modules

| File | Purpose |
|---|---|
| `ai_clipper/semantic_segmenter.py` | **Step 3**: Splits transcript by meaning, not by time. LLM-powered with sentence-boundary fallback. |
| `ai_clipper/multimodal_analyzer.py` | **Step 2**: Audio (RMS, pitch, pauses, excitement) + video (motion, scene changes, brightness) analysis. |
| `ai_clipper/candidate_generator.py` | **Step 5**: Generates 300+ clip candidates from semantic segments using combinatorial boundary expansion. |
| `ai_clipper/retention_predictor.py` | **Step 7**: Predicts watch time, completion rate, drop-off, share/save/comment/subscribe likelihood. |
| `ai_clipper/clip_optimizer.py` | **Step 6**: Iteratively refines clip boundaries using LLM feedback. |
| `ai_clipper/story_series_builder.py` | **Output B**: Builds 3 continuous story series × 3 episodes each. |

### Overhauled Modules

| File | Changes |
|---|---|
| `ai_clipper/scorer.py` | Expanded from 6 to **26 scoring dimensions** using the Master Prompt rubric and final weighted formula. |
| `ai_clipper/pipeline.py` | Full rewrite to orchestrate the 9-step Master Prompt flow. |
| `ai_clipper/config.py` | New fields for target duration, tolerance, candidate count, story series, multimodal toggles. |
| `ai_clipper/html_gallery.py` | Now displays standalone clips and story series separately with retention metrics. |
| `ai_clipper/web_app.py` | Updated Streamlit UI with story series, target duration, and optimization controls. |
| `clip_tool.py` | Updated CLI with Master Prompt flags. |
| `ai_clipper/__init__.py` | Exposes all new public classes. |

## Pipeline Flow

```
1. Source resolution (URL or local file)
2. Complete transcription (Whisper, word-level timestamps)
3. Content type classification
4. Semantic segmentation (by meaning)
5. Multimodal analysis (audio + video)
6. 26-dimension scoring of every segment
7. Generate 300+ candidates
8. Fast heuristic scoring of candidates
9. Iterative optimization of top candidates
10. Select 8 diverse standalone clips (~2:28 each)
11. Build 3 story series (3 episodes each, continuous)
12. Moderation, cutting, captioning, gallery
```

## Master Prompt Weighted Score

```
Overall =
  0.30 × Viewer Retention +
  0.20 × Story Quality +
  0.15 × Hook Strength +
  0.10 × Emotional Intensity +
  0.10 × Educational Value +
  0.05 × Visual Quality +
  0.05 × Audio Quality +
  0.05 × Virality
```

## CLI Usage

```bash
# Default: 8 standalone clips + 3 story series
python clip_tool.py --url "https://youtube.com/watch?v=..."

# Skip story series for faster processing
python clip_tool.py --url "..." --skip-story-series

# Skip multimodal analysis (audio/video features) for speed
python clip_tool.py --url "..." --skip-multimodal

# Custom target duration
python clip_tool.py --url "..." --target-duration 148 --tolerance 3
```

## Web UI Usage

Run:
```bash
"Open AI Clipper.bat"
# or
py -3 -m streamlit run ai_clipper/web_app.py
```

The UI now includes:
- Number of standalone clips
- Target clip duration and tolerance
- Story series toggle and count
- Minimum candidates and optimization rounds
- Multimodal analysis toggle

## Output

All outputs go to `./output` (CLI) or `./output_clips` (web UI):

- `clips/standalone_01_scoreXX.mp4` … `clips/standalone_08_scoreXX.mp4`
- `clips/series1_ep1_scoreXX.mp4` … `clips/series3_ep3_scoreXX.mp4`
- `index.html` — gallery with standalone + story series sections
- `{video}_report.json` — full report with all scores and metadata

## Notes

- The engine is designed for local LLM use (LM Studio / Ollama). Cloud providers work but will be expensive due to the number of LLM calls.
- Multimodal analysis requires `soundfile` and `opencv-python` (already in `requirements.txt`).
- If the LLM fails at any step, robust fallbacks (sentence boundaries, continuous arcs, heuristic scoring) keep the pipeline running.
