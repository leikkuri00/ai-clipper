"""
Viral Moment Scoring Engine — Step 4 of the Master Prompt.

Scores every semantic segment across 26+ independent dimensions (0-100):
  - Viewer retention
  - Hook strength
  - Emotional intensity
  - Curiosity
  - Information density
  - Educational value
  - Entertainment value
  - Story quality
  - Authenticity
  - Virality
  - Surprise
  - Humor
  - Practical value
  - Quote quality
  - Headline potential
  - Thumbnail potential
  - Replay probability
  - Share probability
  - Comment potential
  - Completion probability
  - Subscriber conversion probability
  - Emotional payoff
  - Resolution quality
  - Context independence
  - Visual quality
  - Audio quality

Then computes the master prompt weighted overall score:
  Overall = 0.30*Retention + 0.20*Story + 0.15*Hook + 0.10*Emotion + 0.10*Education + 0.05*Visual + 0.05*Audio + 0.05*Virality
"""

from __future__ import annotations

import json
import logging
import os
import re
import time
from typing import List, Optional, Literal, Callable

from pydantic import BaseModel

from .config import ClipperConfig

logger = logging.getLogger(__name__)


# ── Data models ────────────────────────────────────────────

class ScoredSegment(BaseModel):
    """A transcript segment scored across all master-prompt dimensions."""
    start: float
    end: float
    text: str
    score: float = 0.0               # master-prompt overall weighted score 0-100
    
    # Core identification
    hook_line: str = ""              # First line that stops the scroll
    reasoning: str = ""              # Why it scored this way
    content_type: str = "unknown"    # podcast/interview/tutorial/vlog/stream
    suggested_title: str = ""        # Suggested clip title
    hashtags: List[str] = []
    
    # Master Prompt: 26 dimension scores
    viewer_retention: float = 0
    hook_strength: float = 0
    emotional_intensity: float = 0
    curiosity: float = 0
    information_density: float = 0
    educational_value: float = 0
    entertainment_value: float = 0
    story_quality: float = 0
    authenticity: float = 0
    virality: float = 0
    surprise: float = 0
    humor: float = 0
    practical_value: float = 0
    quote_quality: float = 0
    headline_potential: float = 0
    thumbnail_potential: float = 0
    replay_probability: float = 0
    share_probability: float = 0
    comment_potential: float = 0
    completion_probability: float = 0
    subscriber_conversion_probability: float = 0
    emotional_payoff: float = 0
    resolution_quality: float = 0
    context_independence: float = 0
    visual_quality: float = 0
    audio_quality: float = 0
    
    # Legacy aliases for backward compatibility
    emotional_peak: float = 0
    standalone_clarity: float = 0
    payoff: float = 0
    quotability: float = 0
    
    # Multimodal signals (populated externally)
    audio_excitement: float = 0.0
    video_motion: float = 0.0
    scene_changes: int = 0
    
    # Retention prediction
    predicted_completion: float = 0.0
    predicted_share_rate: float = 0.0
    predicted_comment_rate: float = 0.0
    
    # Semantic info
    topic: str = ""
    segment_type: str = ""
    emotion: str = ""
    is_complete: bool = True


# ── Content type classification prompt ─────────────────────

CONTENT_TYPE_PROMPT = """Classify the type of content this transcript comes from. Choose exactly one:

- podcast: conversational, multiple speakers, long-form discussion
- interview: Q&A format, one person asking questions
- tutorial: teaching, step-by-step instructions, "how to"
- vlog: personal, single speaker talking to camera, casual
- stream: live streaming, audience interaction, gaming
- presentation: slides or prepared talk, formal delivery
- comedy: stand-up, skits, intentionally funny
- review: product or media review/analysis
- documentary: narrated, informational, edited

TRANSCRIPT SAMPLE (first 1000 chars):
{text}

Respond ONLY with a JSON object:
{{"content_type": "<one of the above>", "confidence": <0-1 float>}}"""


# ── Master Prompt 26-dimension scoring rubric ───────────────

MASTER_SCORING_PROMPT = """You are the world's best AI video editor, documentary editor, YouTube strategist, and TikTok/Reels expert.

Your only objective: score this video segment for maximum viewer retention, storytelling quality, emotional impact, virality, educational value, shareability, narrative coherence, and professional editing quality.

CONTENT TYPE: {content_type}
{context}

Analyze this transcript segment and score it 0-100 across these 26 dimensions. Be critical and realistic — most content should score 40-70. Only truly exceptional moments score 90+.

1. **viewer_retention**: Likelihood viewers keep watching past the first 5 seconds and continue to the end.
2. **hook_strength**: Does the opening 1-5 seconds stop the scroll? Unexpected claim, big question, controversy, emotion, surprise.
3. **emotional_intensity**: Raw emotion — laughter, anger, surprise, vulnerability, excitement, tension.
4. **curiosity**: Does it create a gap in knowledge that viewers urgently want closed?
5. **information_density**: How much valuable information per unit time? High density without being overwhelming.
6. **educational_value**: Concrete learning, insights, "I wish I knew this sooner" value.
7. **entertainment_value**: Pure enjoyment, engagement, watchability.
8. **story_quality**: Clear narrative arc — setup, tension, resolution. Beginning, middle, end.
9. **authenticity**: Feels genuine, unscripted, human, not performative or fake.
10. **virality**: Likelihood to spread — shareable, timely, relatable, controversial-but-fair.
11. **surprise**: Unexpected twist, counterintuitive claim, reveal.
12. **humor**: Funny moments, wit, comedic timing, punchlines.
13. **practical_value**: Actionable advice, tips, steps viewers can apply.
14. **quote_quality**: Memorable one-liner, hot take, screenshot-worthy statement.
15. **headline_potential**: How clickable is the core idea as a title/thumbnail text?
16. **thumbnail_potential**: Visual or verbal moment that would work as a thumbnail hook.
17. **replay_probability**: Would viewers rewatch this?
18. **share_probability**: Would viewers send this to someone?
19. **comment_potential**: Would viewers feel compelled to comment?
20. **completion_probability**: Likelihood of watching to the very end.
21. **subscriber_conversion_probability**: Would this make someone subscribe?
22. **emotional_payoff**: Satisfying emotional resolution at the end.
23. **resolution_quality**: The ending resolves, lands, answers — no abrupt cuts.
24. **context_independence**: Understandable with zero outside context.
25. **visual_quality**: Assumed visual interest based on content (motion, emotion, demonstration).
26. **audio_quality**: Assumed audio clarity, energy, no dead air based on transcript.

Also provide:
- **hook_line**: The exact first sentence or phrase that hooks (max 80 chars)
- **reasoning**: Concise 2-3 sentence explanation of why this works
- **suggested_title**: Clickable title for this clip (max 60 chars, emoji OK)
- **hashtags**: 3-5 relevant hashtags (without # symbol)
- **overall_score**: Weighted average using: 0.30*viewer_retention + 0.20*story_quality + 0.15*hook_strength + 0.10*emotional_intensity + 0.10*educational_value + 0.05*visual_quality + 0.05*audio_quality + 0.05*virality

REJECT clips containing dead air, long silence, bad audio, interruptions, low energy, confusing context, or mid-thought endings. Prefer strong emotion, dynamic delivery, complete ideas, and satisfying payoffs.

TRANSCRIPT SEGMENT [{start_time:.1f}s - {end_time:.1f}s]:
{text}

Respond ONLY with a valid JSON object, no markdown, no code fences:

{{
  "viewer_retention": <0-100>,
  "hook_strength": <0-100>,
  "emotional_intensity": <0-100>,
  "curiosity": <0-100>,
  "information_density": <0-100>,
  "educational_value": <0-100>,
  "entertainment_value": <0-100>,
  "story_quality": <0-100>,
  "authenticity": <0-100>,
  "virality": <0-100>,
  "surprise": <0-100>,
  "humor": <0-100>,
  "practical_value": <0-100>,
  "quote_quality": <0-100>,
  "headline_potential": <0-100>,
  "thumbnail_potential": <0-100>,
  "replay_probability": <0-100>,
  "share_probability": <0-100>,
  "comment_potential": <0-100>,
  "completion_probability": <0-100>,
  "subscriber_conversion_probability": <0-100>,
  "emotional_payoff": <0-100>,
  "resolution_quality": <0-100>,
  "context_independence": <0-100>,
  "visual_quality": <0-100>,
  "audio_quality": <0-100>,
  "overall_score": <0-100>,
  "hook_line": "<text>",
  "reasoning": "<text>",
  "suggested_title": "<text>",
  "hashtags": ["tag1", "tag2", "tag3"]
}}"""


# ── Content-type-specific context hints ───────────────────

TYPE_CONTEXT = {
    "podcast": "Podcast content: conversational, multiple speakers. Weight emotional_intensity, quotability/quote_quality, and context_independence highest. Look for hot takes and relatable moments.",
    "interview": "Interview content: Q&A format. Weight hook_strength, surprising answers, quote_quality, and headline_potential. A bold guest answer is gold.",
    "tutorial": "Tutorial content: step-by-step teaching. Weight educational_value, practical_value, and information_density highest. A concrete, actionable tip wins.",
    "vlog": "Vlog content: personal, single speaker. Weight authenticity, emotional_intensity, and story_quality. Relatable personal stories win.",
    "stream": "Stream content: live, audience interaction. Weight emotional_intensity, humor, surprise, and hook_strength. High-energy reactions and clutch moments.",
    "presentation": "Presentation content: prepared talk. Weight educational_value, quote_quality, and headline_potential. Key insights and memorable slides.",
    "comedy": "Comedy content: intentionally funny. Weight humor, payoff, hook_strength, and replay_probability. The punchline must land.",
    "review": "Review content: product/media analysis. Weight practical_value, context_independence, and headline_potential. The verdict or hot take.",
    "documentary": "Documentary content: narrated, informational. Weight story_quality, curiosity, and emotional_payoff. Fascinating facts or reveals.",
    "unknown": "Balance all dimensions. Prioritize segments that stand alone well with strong hooks and payoffs.",
}


class ViralScorer:
    """Scores transcript segments for viral potential using the master prompt rubric."""

    WEIGHTS = {
        "viewer_retention": 0.30,
        "story_quality": 0.20,
        "hook_strength": 0.15,
        "emotional_intensity": 0.10,
        "educational_value": 0.10,
        "visual_quality": 0.05,
        "audio_quality": 0.05,
        "virality": 0.05,
    }

    def __init__(self, config: ClipperConfig):
        self.config = config
        self.provider = config.llm_provider
        self.model = config.llm_model
        self._content_type: Optional[str] = None

    # ── LLM routing (kept for backward compatibility) ─────────

    def _get_ollama_response(self, prompt: str) -> str:
        import ollama
        response = ollama.chat(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            options={"temperature": 0.3, "num_predict": 2048},
        )
        return response["message"]["content"]

    def _get_openai_compatible_response(
        self, prompt: str, base_url: str | None = None
    ) -> str:
        from openai import OpenAI
        url = base_url or self.config.cloud_api_base
        # Local OpenAI-compatible servers (LM Studio, Ollama, etc.) don't need a
        # real key, but the OpenAI client requires one to be set.
        api_key = os.environ.get("OPENAI_API_KEY") or "not-needed"
        client = OpenAI(base_url=url, api_key=api_key)
        response = client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3,
            max_tokens=2048,
        )
        return response.choices[0].message.content or ""

    def _get_local_response(self, prompt: str) -> str:
        try:
            from llama_cpp import Llama
        except ImportError as e:
            raise RuntimeError(
                "llama-cpp-python is not installed. Install it with `pip install llama-cpp-python sentencepiece`."
            ) from e

        model_path = self.config.llm_model
        if not model_path:
            raise RuntimeError(
                "Local LLM model path is not configured. Set LLAMA_MODEL_PATH or provide llm_model path in config."
            )

        llama = Llama(model_path=model_path)
        response = llama(
            prompt,
            max_tokens=1024,
            temperature=0.3,
            stop=["\n\n"],
        )
        if not response or not getattr(response, "choices", None):
            raise RuntimeError("Local LLM produced no response.")
        return response.choices[0].text.strip()

    def _call_router(self, prompt: str) -> str:
        """Route through external llm-router.ps1."""
        import subprocess
        import tempfile
        from pathlib import Path
        router = self.config.llm_router_script
        if not Path(router).exists():
            logger.warning(f"Router script not found: {router}. Falling back to local provider.")
            return self._get_local_response(prompt)

        tf = Path(tempfile.mktemp(suffix=".json"))
        tf.write_text(json.dumps({"prompt": prompt}), encoding="utf-8")
        result = subprocess.run(
            ["powershell", "-File", router, "-InputFile", str(tf)],
            capture_output=True, text=True, timeout=120,
        )
        tf.unlink(missing_ok=True)
        if result.returncode == 0:
            return result.stdout.strip()
        raise RuntimeError(f"Router failed: {result.stderr[:300]}")

    def call_llm(self, prompt: str) -> str:
        """Route to correct provider with fallback chain."""
        try:
            if self.provider == "router":
                return self._call_router(prompt)
        except Exception as e:
            logger.warning(f"Router failed: {e}")

        if self.provider == "local":
            return self._get_local_response(prompt)

        try:
            if self.provider == "ollama":
                return self._get_ollama_response(prompt)
        except Exception as e:
            logger.warning(f"Ollama failed: {e}")

        if self.provider == "lmstudio":
            try:
                return self._get_openai_compatible_response(
                    prompt, base_url=self.config.lmstudio_api_base
                )
            except Exception as e:
                logger.error(f"LM Studio failed: {e}")
                raise RuntimeError(f"LLM call failed: {e}")

        if self.provider == "openai":
            try:
                return self._get_openai_compatible_response(prompt)
            except Exception as e:
                logger.error(f"OpenAI failed: {e}")
                raise RuntimeError(f"LLM call failed: {e}")

        raise RuntimeError(f"Unsupported LLM provider: {self.provider}")

    # ── Content type classification ─────────────────────────

    def classify_content_type(self, transcript_text: str) -> str:
        """Classify the content type from the first portion of transcript."""
        if self._content_type:
            return self._content_type

        sample = transcript_text[:2000]
        prompt = CONTENT_TYPE_PROMPT.format(text=sample)

        try:
            response = self.call_llm(prompt)
            parsed = self._parse_json(response)
            ct = parsed.get("content_type", "unknown") if parsed else "unknown"
            self._content_type = ct
            logger.info(f"Content type classified as: {ct}")
            return ct
        except Exception as e:
            logger.warning(f"Content classification failed: {e}")
            self._content_type = "unknown"
            return "unknown"

    # ── Segment scoring ─────────────────────────────────────

    def score_segment(
        self,
        start_time: float,
        end_time: float,
        text: str,
        content_type: str = "unknown",
    ) -> Optional[ScoredSegment]:
        """Score a single transcript segment with the full 26-dimension rubric."""
        if not text.strip() or len(text.strip()) < 20:
            return None

        # Give the model the full clip transcript (2:28 clips ~ 2-3k chars);
        # only truncate pathologically long inputs.
        max_chars = 6000
        if len(text) > max_chars:
            text = text[:max_chars] + "..."

        context_hint = TYPE_CONTEXT.get(content_type, TYPE_CONTEXT["unknown"])

        prompt = MASTER_SCORING_PROMPT.format(
            content_type=content_type,
            context=context_hint,
            start_time=start_time,
            end_time=end_time,
            text=text,
        )

        try:
            response = self.call_llm(prompt)
            parsed = self._parse_json(response)

            if not parsed:
                logger.warning(f"Failed to parse score. Raw: {response[:200]}")
                return None

            # Compute overall score from dimensions if not provided
            overall = self._compute_overall(parsed)

            return ScoredSegment(
                start=start_time,
                end=end_time,
                text=text[:500],
                score=overall,
                hook_line=parsed.get("hook_line", ""),
                reasoning=parsed.get("reasoning", ""),
                content_type=content_type,
                suggested_title=parsed.get("suggested_title", ""),
                hashtags=parsed.get("hashtags", []),
                # Master prompt dimensions
                viewer_retention=float(parsed.get("viewer_retention", 0)),
                hook_strength=float(parsed.get("hook_strength", 0)),
                emotional_intensity=float(parsed.get("emotional_intensity", 0)),
                curiosity=float(parsed.get("curiosity", 0)),
                information_density=float(parsed.get("information_density", 0)),
                educational_value=float(parsed.get("educational_value", 0)),
                entertainment_value=float(parsed.get("entertainment_value", 0)),
                story_quality=float(parsed.get("story_quality", 0)),
                authenticity=float(parsed.get("authenticity", 0)),
                virality=float(parsed.get("virality", 0)),
                surprise=float(parsed.get("surprise", 0)),
                humor=float(parsed.get("humor", 0)),
                practical_value=float(parsed.get("practical_value", 0)),
                quote_quality=float(parsed.get("quote_quality", 0)),
                headline_potential=float(parsed.get("headline_potential", 0)),
                thumbnail_potential=float(parsed.get("thumbnail_potential", 0)),
                replay_probability=float(parsed.get("replay_probability", 0)),
                share_probability=float(parsed.get("share_probability", 0)),
                comment_potential=float(parsed.get("comment_potential", 0)),
                completion_probability=float(parsed.get("completion_probability", 0)),
                subscriber_conversion_probability=float(parsed.get("subscriber_conversion_probability", 0)),
                emotional_payoff=float(parsed.get("emotional_payoff", 0)),
                resolution_quality=float(parsed.get("resolution_quality", 0)),
                context_independence=float(parsed.get("context_independence", 0)),
                visual_quality=float(parsed.get("visual_quality", 0)),
                audio_quality=float(parsed.get("audio_quality", 0)),
                # Legacy aliases
                emotional_peak=float(parsed.get("emotional_intensity", 0)),
                standalone_clarity=float(parsed.get("context_independence", 0)),
                payoff=float(parsed.get("emotional_payoff", 0)),
                quotability=float(parsed.get("quote_quality", 0)),
            )

        except Exception as e:
            logger.error(f"Scoring failed [{start_time:.1f}-{end_time:.1f}]: {e}")
            return None

    def _compute_overall(self, parsed: dict) -> float:
        """Compute master-prompt weighted overall score from dimensions."""
        retention = float(parsed.get("viewer_retention", 50))
        story = float(parsed.get("story_quality", 50))
        hook = float(parsed.get("hook_strength", 50))
        emotion = float(parsed.get("emotional_intensity", 50))
        education = float(parsed.get("educational_value", 50))
        visual = float(parsed.get("visual_quality", 50))
        audio = float(parsed.get("audio_quality", 50))
        virality = float(parsed.get("virality", 50))

        overall = (
            self.WEIGHTS["viewer_retention"] * retention +
            self.WEIGHTS["story_quality"] * story +
            self.WEIGHTS["hook_strength"] * hook +
            self.WEIGHTS["emotional_intensity"] * emotion +
            self.WEIGHTS["educational_value"] * education +
            self.WEIGHTS["visual_quality"] * visual +
            self.WEIGHTS["audio_quality"] * audio +
            self.WEIGHTS["virality"] * virality
        )

        # Use LLM-provided overall if available and reasonable
        llm_overall = float(parsed.get("overall_score", 0))
        if 0 < llm_overall <= 100:
            # Blend: 70% formula, 30% LLM to prevent drift
            overall = 0.7 * overall + 0.3 * llm_overall

        return round(overall, 2)

    # ── Batch scoring for semantic segments ─────────────────

    def score_semantic_segments(
        self,
        segments: List,
        content_type: Optional[str] = None,
    ) -> List[ScoredSegment]:
        """
        Score semantic segments with the full 26-dimension rubric.

        Args:
            segments: List of SemanticSegment objects
            content_type: Optional content type override

        Returns:
            List of ScoredSegment sorted by score descending
        """
        ct = content_type or "unknown"
        scored: List[ScoredSegment] = []

        logger.info(f"Scoring {len(segments)} semantic segments (type={ct})")

        for i, seg in enumerate(segments):
            result = self.score_segment(seg.start_time, seg.end_time, seg.text, ct)
            if result:
                # Copy semantic metadata
                result.topic = getattr(seg, "topic", "")
                result.segment_type = getattr(seg, "segment_type", "")
                result.emotion = getattr(seg, "emotion", "")
                result.is_complete = getattr(seg, "is_complete", True)
                scored.append(result)

            if i % 10 == 0 and i > 0:
                logger.debug(f"  Scoring progress: {i}/{len(segments)}")

        scored.sort(key=lambda s: s.score, reverse=True)

        logger.info(f"Scored {len(scored)} segments. Top 5:")
        for s in scored[:5]:
            logger.info(f"  [{s.score:.0f}] {s.hook_line[:60]}... ({s.start:.1f}-{s.end:.1f}s)")

        return scored

    # ── Legacy sliding-window scoring (kept for compatibility) ─

    def score_transcript(
        self,
        words: list,
        words_per_segment: int = 100,
        overlap: int = 25,
        content_type: Optional[str] = None,
    ) -> List[ScoredSegment]:
        """Legacy sliding-window scoring."""
        if not words:
            return []

        ct = content_type or "unknown"
        scored: List[ScoredSegment] = []
        step = max(1, words_per_segment - overlap)
        total_windows = max(1, (len(words) - overlap) // step)

        logger.info(f"Legacy scoring {len(words)} words in ~{total_windows} windows (type={ct})")

        for idx in range(0, len(words), step):
            window = words[idx: idx + words_per_segment]
            if len(window) < 10:
                continue

            start_time = window[0].start
            end_time = window[-1].end
            text = " ".join(w.word for w in window)

            result = self.score_segment(start_time, end_time, text, content_type=ct)
            if result:
                scored.append(result)

            if idx // step % 20 == 0:
                logger.debug(f"  Scoring progress: {idx // step}/{total_windows}")

        scored.sort(key=lambda s: s.score, reverse=True)
        return scored

    # ── Hashtag generation ─────────────────────────────────

    def generate_hashtags(
        self,
        text: str,
        suggested_title: str = "",
        reasoning: str = "",
    ) -> dict:
        """Generate viral hashtags for a clip."""
        prompt = f"""Generate 3-5 optimized hashtags for this short video clip.

Clip title: {suggested_title}
Clip content: {text[:300]}
Why it's engaging: {reasoning[:200]}

Rules:
- Mix broad and niche tags
- No spaces, use CamelCase
- No # symbols in the list
- Order from most to least important

Respond ONLY with: {{"hashtags": ["tag1", "tag2", ...]}}"""

        try:
            response = self.call_llm(prompt)
            parsed = self._parse_json(response)
            if parsed and "hashtags" in parsed:
                return parsed
        except Exception as e:
            logger.warning(f"Hashtag generation failed: {e}")

        return {"hashtags": ["viral", "mustwatch", "trending"]}

    # ── JSON parsing ───────────────────────────────────────

    @staticmethod
    def _parse_json(text: str) -> Optional[dict]:
        """
        Extract a JSON object from an LLM response.

        Tolerant of markdown fences and of responses that were cut off before the
        closing brace (common with smaller models and long rubrics): first tries
        strict parsing, then brace-repair, then a key/value regex fallback so a
        truncated score is still usable instead of being dropped entirely.
        """
        if not text:
            return None
        cleaned = re.sub(r'^```(?:json)?\s*', '', text.strip(), flags=re.MULTILINE)
        cleaned = re.sub(r'\s*```$', '', cleaned.strip(), flags=re.MULTILINE)

        # 1) Strict: largest {...} span, then the whole string.
        match = re.search(r'\{.*\}', cleaned, re.DOTALL)
        for candidate in ([match.group()] if match else []) + [cleaned]:
            try:
                return json.loads(candidate)
            except json.JSONDecodeError:
                pass

        # 2) Repair a truncated object: drop the trailing incomplete field and
        #    balance braces.
        start = cleaned.find("{")
        if start != -1:
            frag = cleaned[start:]
            frag = re.sub(r',\s*"[^"]*"\s*:?\s*[^,{}\[\]]*$', '', frag)  # partial last pair
            frag = frag.rstrip().rstrip(",")
            frag += "}" * max(0, frag.count("{") - frag.count("}"))
            try:
                return json.loads(frag)
            except json.JSONDecodeError:
                pass

        # 3) Last resort: pull out "key": value pairs directly.
        result: dict = {}
        for key, num in re.findall(r'"([\w ]+)"\s*:\s*(-?\d+(?:\.\d+)?)', cleaned):
            result[key] = float(num)
        for key, sval in re.findall(r'"([\w ]+)"\s*:\s*"([^"]*)"', cleaned):
            result.setdefault(key, sval)
        return result or None
