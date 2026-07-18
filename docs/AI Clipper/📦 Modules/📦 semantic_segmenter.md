# semantic_segmenter.py

> Step 3 of the Master Prompt: split transcript by meaning, not by time.

## Class

`SemanticSegmenter(llm_call_fn)`

## Main method

`segment(words, max_words_per_chunk=800) -> List[SemanticSegment]`

## Rules

- Split by meaning: one complete idea per segment
- Boundaries at sentence endings, topic transitions, breath pauses, story beats, emotional changes
- Never cut mid-sentence, mid-joke, mid-explanation, mid-emotion, mid-reveal

## Output

`SemanticSegment` contains:

- `id`
- `start_time`, `end_time`
- `text`
- `words`
- `topic`
- `segment_type` (hook, story, payoff, advice, etc.)
- `emotion`
- `is_complete`

## Fallback

If LLM segmentation fails, falls back to sentence-boundary segmentation.

---

#module #segmentation #ai-clipper
