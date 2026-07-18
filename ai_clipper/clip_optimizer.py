"""
Iterative Clip Optimization Engine — Step 6 of the Master Prompt.

Repeats until no meaningful improvement exists:
  Generate candidates → Predict retention → Adjust boundaries → Rescore → Compare → Keep strongest

Uses LLM feedback to refine clip boundaries toward:
  - Stronger hooks within 5 seconds
  - More satisfying endings
  - Better narrative coherence
  - Higher predicted retention
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass, field
from typing import List, Callable, Optional

from .candidate_generator import ClipCandidate
from .scorer import ScoredSegment
from .retention_predictor import RetentionPredictor, RetentionPrediction

logger = logging.getLogger(__name__)


@dataclass
class OptimizedClip:
    """A clip candidate after iterative optimization."""
    start_time: float
    end_time: float
    text: str
    score: float
    retention: RetentionPrediction = field(default_factory=RetentionPrediction)
    improvements: List[str] = field(default_factory=list)
    iteration: int = 0


BOUNDARY_OPTIMIZATION_PROMPT = """You are an elite video editor optimizing clip boundaries.

CURRENT CLIP:
- Start: {start:.1f}s
- End: {end:.1f}s
- Duration: {duration:.1f}s
- Current overall score: {score:.1f}/100

CLIP TEXT:
{text}

DIMENSION SCORES:
{scores}

RETENTION PREDICTION:
{retention}

Your task: suggest better start/end times to maximize the clip.

Rules:
1. Start must have a strong hook within 5 seconds.
2. End must land after a resolution, punchline, lesson, reveal, or emotional payoff.
3. Never cut mid-sentence, mid-joke, mid-explanation, or mid-emotion.
4. Keep the clip between {min_duration:.0f} and {max_duration:.0f} seconds.
5. Target duration: {target_duration:.0f}s ± {tolerance:.0f}s.

Respond with a JSON object:
{{
  "new_start": <float seconds>,
  "new_end": <float seconds>,
  "reasoning": "<what you improved>",
  "expected_score": <0-100>
}}

Respond ONLY with valid JSON, no markdown, no code fences."""


class ClipOptimizer:
    """Iteratively refines clip candidates using LLM feedback."""

    def __init__(
        self,
        scorer: Callable[[float, float, str], Optional[ScoredSegment]],
        retention_predictor: RetentionPredictor,
        target_duration: float = 148.0,
        tolerance: float = 3.0,
        min_duration: float = 30.0,
        max_duration: float = 180.0,
        max_rounds: int = 3,
    ):
        self._scorer = scorer
        self._retention = retention_predictor
        self.target_duration = target_duration
        self.tolerance = tolerance
        self.min_duration = min_duration
        self.max_duration = max_duration
        self.max_rounds = max_rounds

    def optimize(
        self,
        candidates: List[ClipCandidate],
        all_words: List,
        top_k: int = 50,
    ) -> List[OptimizedClip]:
        """
        Optimize top candidates iteratively.

        Args:
            candidates: Generated candidates
            all_words: Full word-level transcript
            top_k: Optimize only the top K candidates

        Returns:
            List of OptimizedClip with refined boundaries
        """
        if not candidates:
            return []

        # Sort by current score and take top_k
        top_candidates = candidates[:top_k]
        optimized: List[OptimizedClip] = []

        logger.info(f"Optimizing top {len(top_candidates)} candidates for {self.max_rounds} rounds")

        for i, cand in enumerate(top_candidates):
            opt = self._optimize_one(cand, all_words, i)
            if opt:
                optimized.append(opt)

        # Sort by final score descending
        optimized.sort(key=lambda c: c.score, reverse=True)
        return optimized

    def _optimize_one(
        self,
        candidate: ClipCandidate,
        all_words: List,
        index: int,
    ) -> Optional[OptimizedClip]:
        """Run iterative optimization on a single candidate."""
        current_start = candidate.start_time
        current_end = candidate.end_time
        current_text = candidate.text
        current_score = 0.0
        current_scored: Optional[ScoredSegment] = None
        current_retention = RetentionPrediction()
        improvements = []

        # Initial score
        scored = self._scorer(current_start, current_end, current_text)
        if scored:
            current_score = scored.score
            current_scored = scored
            current_retention = self._retention.predict(
                current_text, current_start, current_end, self._scores_to_dict(scored)
            )

        for round_idx in range(self.max_rounds):
            # Build prompt
            prompt = BOUNDARY_OPTIMIZATION_PROMPT.format(
                start=current_start,
                end=current_end,
                duration=current_end - current_start,
                score=current_score,
                text=current_text[:2000],
                scores=self._format_scores(current_scored),
                retention=self._format_retention(current_retention),
                min_duration=self.min_duration,
                max_duration=self.max_duration,
                target_duration=self.target_duration,
                tolerance=self.tolerance,
            )

            try:
                response = self._retention._call_llm(prompt)
                parsed = self._parse_json(response)
                if not parsed:
                    break

                new_start = float(parsed.get("new_start", current_start))
                new_end = float(parsed.get("new_end", current_end))
                expected_score = float(parsed.get("expected_score", current_score))
                reasoning = parsed.get("reasoning", "")

                # Validate boundaries
                new_start, new_end = self._validate_boundaries(new_start, new_end)
                if new_start is None:
                    break

                # Get text for new boundaries
                new_text = self._get_text_between(all_words, new_start, new_end)
                if len(new_text.strip()) < 20:
                    break

                # Rescore
                new_scored = self._scorer(new_start, new_end, new_text)
                if not new_scored:
                    break

                new_retention = self._retention.predict(
                    new_text, new_start, new_end, self._scores_to_dict(new_scored)
                )

                # Accept if meaningful improvement (>2 points)
                if new_scored.score > current_score + 2.0:
                    current_start = new_start
                    current_end = new_end
                    current_text = new_text
                    current_score = new_scored.score
                    current_scored = new_scored
                    current_retention = new_retention
                    if reasoning:
                        improvements.append(f"Round {round_idx+1}: {reasoning}")
                    logger.debug(
                        f"  Candidate {index}: improved to {current_score:.1f} "
                        f"[{current_start:.1f}-{current_end:.1f}]"
                    )
                else:
                    # No meaningful improvement
                    break

            except Exception as e:
                logger.debug(f"Optimization round failed for candidate {index}: {e}")
                break

        if current_scored is None:
            return None

        return OptimizedClip(
            start_time=current_start,
            end_time=current_end,
            text=current_text,
            score=current_score,
            retention=current_retention,
            improvements=improvements,
            iteration=len(improvements),
        )

    def _validate_boundaries(
        self,
        start: float,
        end: float,
    ) -> tuple[Optional[float], Optional[float]]:
        """Ensure optimized boundaries are valid."""
        if start < 0 or end <= start:
            return None, None
        duration = end - start
        if duration < self.min_duration or duration > self.max_duration:
            return None, None
        return start, end

    def _get_text_between(
        self,
        all_words: List,
        start: float,
        end: float,
    ) -> str:
        """Extract transcript text between timestamps."""
        words_in_range = [
            w for w in all_words
            if w.end > start and w.start < end
        ]
        return " ".join(w.word for w in words_in_range)

    def _scores_to_dict(self, scored: Optional[ScoredSegment]) -> dict:
        """Convert ScoredSegment to dimension score dict."""
        if scored is None:
            return {}
        return {
            "viewer_retention": scored.viewer_retention,
            "hook_strength": scored.hook_strength,
            "emotional_intensity": scored.emotional_intensity,
            "curiosity": scored.curiosity,
            "information_density": scored.information_density,
            "educational_value": scored.educational_value,
            "entertainment_value": scored.entertainment_value,
            "story_quality": scored.story_quality,
            "authenticity": scored.authenticity,
            "virality": scored.virality,
            "surprise": scored.surprise,
            "humor": scored.humor,
            "practical_value": scored.practical_value,
            "quote_quality": scored.quote_quality,
            "headline_potential": scored.headline_potential,
            "thumbnail_potential": scored.thumbnail_potential,
            "replay_probability": scored.replay_probability,
            "share_probability": scored.share_probability,
            "comment_potential": scored.comment_potential,
            "completion_probability": scored.completion_probability,
            "subscriber_conversion_probability": scored.subscriber_conversion_probability,
            "emotional_payoff": scored.emotional_payoff,
            "resolution_quality": scored.resolution_quality,
            "context_independence": scored.context_independence,
            "visual_quality": scored.visual_quality,
            "audio_quality": scored.audio_quality,
        }

    def _format_scores(self, scored: Optional[ScoredSegment]) -> str:
        """Format dimension scores for prompt."""
        if scored is None:
            return "  No scores available"
        scores = self._scores_to_dict(scored)
        lines = [f"  {k}: {v:.1f}" for k, v in scores.items()]
        return "\n".join(lines[:20])

    def _format_retention(self, retention: RetentionPrediction) -> str:
        """Format retention prediction for prompt."""
        return (
            f"  avg_watch_time: {retention.avg_watch_time_sec:.1f}s\n"
            f"  completion_rate: {retention.completion_rate:.2f}\n"
            f"  drop_off_rate: {retention.drop_off_rate:.2f}\n"
            f"  share_likelihood: {retention.share_likelihood:.2f}\n"
            f"  comment_likelihood: {retention.comment_likelihood:.2f}\n"
            f"  engagement_rate: {retention.engagement_rate:.2f}"
        )

    @staticmethod
    def _parse_json(text: str) -> Optional[dict]:
        if not text:
            return None
        text = re.sub(r'^```(?:json)?\s*', '', text.strip(), flags=re.MULTILINE)
        text = re.sub(r'\s*```$', '', text.strip(), flags=re.MULTILINE)
        match = re.search(r'\{.*\}', text, re.DOTALL)
        if match:
            try:
                return json.loads(match.group())
            except json.JSONDecodeError:
                pass
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            pass
        return None
