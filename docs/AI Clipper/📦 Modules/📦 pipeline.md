# pipeline.py

> Main orchestration module. Implements the full Master Prompt flow.

## Entry class

`AIClipper(config: ClipperConfig)`

## Main method

`run(video_path=None) -> ClipResult`

## Pipeline steps

1. Source resolution (URL or local file)
2. Audio extraction
3. Whisper transcription
4. Content type classification
5. Semantic segmentation
6. Multimodal analysis
7. 26-dimension segment scoring
8. Candidate generation (300+)
9. Fast heuristic candidate scoring
10. Iterative optimization
11. Standalone clip selection (8 clips)
12. Story series generation (3 series × 3 episodes)
13. Moderation
14. Clip cutting & captioning
15. Report & gallery generation

## Key methods

- `_resolve_source()` — download or validate local file
- `_analyze_multimodal()` — audio/video features per segment
- `_apply_multimodal_scores()` — boost/dampen scores from signals
- `_score_candidates()` — fast overlap-based candidate scoring
- `_select_standalone_clips()` — diverse 8-clip selection
- `_cut_clips()` — render all outputs
- `_build_report()` — JSON report

## Output

`ClipResult` contains:

- `clips: List[Path]`
- `gallery_path: Path`
- `report: dict`
- `story_series: List[StorySeries]`

---

#module #pipeline #ai-clipper
