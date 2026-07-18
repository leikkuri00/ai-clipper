# Master Prompt

> The full specification that drives AI Clipper v1.0.

## Objective

Find the absolute best clips from any long-form video. Optimize only for:

- Maximum viewer retention
- Storytelling quality
- Emotional impact
- Virality
- Educational value
- Shareability
- Narrative coherence
- Professional editing quality

## Pipeline steps

### Step 1 — Complete Transcription

Generate a complete transcript with:

- Word-level timestamps
- Speaker diarization
- Proper punctuation
- Paragraphs
- Emphasis
- Pauses
- Laughter
- Tone
- Emotion
- Non-verbal audio events

### Step 2 — Multimodal Understanding

Analyze simultaneously:

**Audio:** voice excitement, energy, volume, pitch, pauses, silence, laughter, emotion, music, audience reaction.

**Video:** scene changes, camera movement, facial expressions, eye contact, gestures, body language, visual motion, on-screen text, slides, graphics, objects, lighting, framing.

**Language:** context, meaning, intent, humor, sarcasm, emotion, narrative, story arcs, arguments, examples, analogies, quotes, advice, personal stories, statistics, controversial statements, memorable moments.

### Step 3 — Semantic Segmentation

Never split by time. Split only by meaning. Boundaries at:

- Sentence endings
- Topic transitions
- Breath pauses
- Story beats
- Emotional changes

Never cut mid-sentence, mid-joke, mid-explanation, mid-emotion, or mid-reveal.

### Step 4 — Score Every Segment

26 dimensions (0–100):

1. viewer_retention
2. hook_strength
3. emotional_intensity
4. curiosity
5. information_density
6. educational_value
7. entertainment_value
8. story_quality
9. authenticity
10. virality
11. surprise
12. humor
13. practical_value
14. quote_quality
15. headline_potential
16. thumbnail_potential
17. replay_probability
18. share_probability
19. comment_potential
20. completion_probability
21. subscriber_conversion_probability
22. emotional_payoff
23. resolution_quality
24. context_independence
25. visual_quality
26. audio_quality

### Step 5 — Generate Candidates

Minimum 300 candidates. Each may begin/end at different semantic boundaries. Never fixed intervals.

### Step 6 — Iterative Optimization

Repeat until convergence:

Generate → Predict retention → Adjust boundaries → Rescore → Compare → Keep strongest.

### Step 7 — Retention Prediction

Predict for every candidate:

- Average watch time
- Completion rate
- Viewer drop-off
- Replay likelihood
- Share likelihood
- Save likelihood
- Comment likelihood
- Subscription likelihood

## Outputs

### Output A — 8 Standalone Clips

- Duration: 2 minutes 28 seconds ± 3 seconds
- Begin with strong hook within 5 seconds
- Complete context, understandable alone
- Clear beginning, middle, satisfying ending
- Maximize topic diversity

### Output B — Story Series

- 3 series
- 3 episodes per series
- Each episode ~2 minutes
- Episode 2 starts exactly where Episode 1 ends
- Episode 3 starts exactly where Episode 2 ends
- No skipped footage, no overlap, no repeated scenes

**Episode 1:** hook → story → conflict → cliffhanger  
**Episode 2:** escalation → twist → discoveries → stronger cliffhanger  
**Episode 3:** resolution → lesson → payoff → emotional ending

## Final ranking formula

```text
Overall Score =
  0.30 × Predicted Retention +
  0.20 × Story Quality +
  0.15 × Hook Strength +
  0.10 × Emotional Impact +
  0.10 × Educational Value +
  0.05 × Visual Quality +
  0.05 × Audio Quality +
  0.05 × Virality
```

---

#master-prompt #specification #ai-clipper
