# config.py

> Central configuration. Every tunable value is a field — nothing hardcoded.

## Key config groups

### Source

- `source_url`
- `source_path`

### Models

- `transcribe_provider`: `"groq" | "local"`
- `whisper_model`: e.g. `"large-v3-turbo"`
- `llm_provider`: `"bionic" | "ollama" | "openai" | "router"`
- `llm_model`

### Master Prompt outputs

- `num_clips`: 8
- `target_clip_duration`: 148.0 (2:28)
- `clip_duration_tolerance`: 3.0
- `num_story_series`: 3
- `episodes_per_series`: 3
- `episode_target_duration`: 120.0

### Candidate generation

- `min_candidates`: 300
- `iterative_rounds`: 3

### Toggles

- `skip_multimodal`
- `skip_story_series`
- `story_only`
- `offline_mode`

### Backward-compatible aliases

- `vertical_crop` — true for 9:16 or 4:5
- `crop_aspect_w` / `crop_aspect_h`
- `video_output_size()`

---

#module #config #ai-clipper
