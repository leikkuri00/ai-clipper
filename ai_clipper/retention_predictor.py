"""
Retention Prediction Engine — Step 7 of the Master Prompt.

Predicts for every clip candidate:
  - Average watch time
  - Completion rate
  - Viewer drop-off points
  - Replay likelihood
  - Share likelihood
  - Save likelihood
  - Comment likelihood
  - Subscription likelihood

Combines the 26 dimension scores with the master prompt's weighted formula:
  Overall = 0.30*Retention + 0.20*Story + 0.15*Hook + 0.10*Emotion + 0.10*Education + 0.05*Visual + 0.05*Audio + 0.05*Virality
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass
from typing import Callable, Optional

from pydantic import BaseModel

logger = logging.getLogger(__name__)


class RetentionPrediction(BaseModel):
    """Predicted performance metrics for a clip candidate."""
    avg_watch_time_sec: float = 0.0
    completion_rate: float = 0.0      # 0-1
    drop_off_rate: float = 0.0        # 0-1
    replay_likelihood: float = 0.0    # 0-1
    share_likelihood: float = 0.0     # 0-1
    save_likelihood: float = 0.0      # 0-1
    comment_likelihood: float = 0.0   # 0-1
    subscribe_likelihood: float = 0.0 # 0-1
    
    # Normalized 0-100
    retention_score: float = 0.0
    
    # Drop-off timestamps (seconds from clip start)
    drop_off_points: list[float] = []
    
    # Estimated engagement rate
    engagement_rate: float = 0.0        # composite 0-1


class RetentionPredictor:
    """Predicts viewer retention and engagement for clip candidates."""

    # Master prompt final ranking weights
    WEIGHTS = {
        "retention": 0.30,
        "story": 0.20,
        "hook": 0.15,
        "emotion": 0.10,
        "education": 0.10,
        "visual": 0.05,
        "audio": 0.05,
        "virality": 0.05,
    }

    def __init__(self, llm_call_fn: Callable[[str], str]):
        self._call_llm = llm_call_fn

    def predict(
        self,
        transcript_text: str,
        clip_start: float,
        clip_end: float,
        dimension_scores: dict,
    ) -> RetentionPrediction:
        """
        Predict retention metrics for a clip candidate.
        
        Args:
            transcript_text: Full text of the clip
            clip_start: Start time in source video
            clip_end: End time in source video
            dimension_scores: The 26 dimension scores (0-100)
        
        Returns:
            RetentionPrediction with estimated metrics
        """
        # First compute formula-based score
        formula_score = self._compute_formula_score(dimension_scores)
        
        # Build prompt for LLM refinement
        prompt = self._build_prediction_prompt(
            transcript_text, clip_start, clip_end, dimension_scores, formula_score
        )

        try:
            response = self._call_llm(prompt)
            parsed = self._parse_json(response)
            if parsed:
                return self._build_prediction(parsed, formula_score)
        except Exception as e:
            logger.warning(f"LLM retention prediction failed: {e}")

        # Fallback: derive from dimension scores
        return self._fallback_prediction(dimension_scores, clip_end - clip_start)

    def _compute_formula_score(self, scores: dict) -> float:
        """Compute master prompt weighted overall score."""
        # Map dimension scores to the 8 weighted categories
        retention_inputs = [
            scores.get("viewer_retention", 50),
            scores.get("completion_probability", 50),
            scores.get("replay_probability", 50),
        ]
        retention = float(sum(retention_inputs) / len(retention_inputs)) if retention_inputs else 50

        story_inputs = [
            scores.get("story_quality", 50),
            scores.get("emotional_payoff", 50),
            scores.get("resolution_quality", 50),
            scores.get("narrative_coherence", 50),
        ]
        story = float(sum(story_inputs) / len(story_inputs)) if story_inputs else 50

        hook = scores.get("hook_strength", 50)

        emotion_inputs = [
            scores.get("emotional_impact", 50),
            scores.get("emotional_intensity", 50),
            scores.get("surprise", 50),
        ]
        emotion = float(sum(emotion_inputs) / len(emotion_inputs)) if emotion_inputs else 50

        education = scores.get("educational_value", 50)

        visual = scores.get("visual_quality", 50)
        audio = scores.get("audio_quality", 50)

        virality_inputs = [
            scores.get("virality", 50),
            scores.get("share_probability", 50),
            scores.get("comment_potential", 50),
        ]
        virality = float(sum(virality_inputs) / len(virality_inputs)) if virality_inputs else 50

        overall = (
            self.WEIGHTS["retention"] * retention +
            self.WEIGHTS["story"] * story +
            self.WEIGHTS["hook"] * hook +
            self.WEIGHTS["emotion"] * emotion +
            self.WEIGHTS["education"] * education +
            self.WEIGHTS["visual"] * visual +
            self.WEIGHTS["audio"] * audio +
            self.WEIGHTS["virality"] * virality
        )
        return round(overall, 2)

    def _build_prediction_prompt(
        self,
        text: str,
        start: float,
        end: float,
        scores: dict,
        formula_score: float,
    ) -> str:
        """Build LLM prompt for retention prediction."""
        # Truncate text
        text = text[:2000]
        
        # Build score summary
        score_lines = []
        for k, v in scores.items():
            score_lines.append(f"  {k}: {v:.1f}")
        score_block = "\n".join(score_lines[:30])

        return f"""You are a YouTube/TikTok retention analyst. Predict how viewers will engage with this clip.

CLIP TEXT:
{text}

CLIP TIME: {start:.1f}s - {end:.1f}s
DURATION: {end-start:.1f}s
FORMULA OVERALL SCORE: {formula_score:.1f}/100

DIMENSION SCORES (0-100):
{score_block}

Predict the following metrics as a JSON object:
{{
  "avg_watch_time_sec": <estimated average seconds watched>,
  "completion_rate": <0-1 probability>,
  "drop_off_rate": <0-1 probability of early drop-off>,
  "drop_off_points": [<list of seconds from clip start where major drop-offs occur>],
  "replay_likelihood": <0-1>,
  "share_likelihood": <0-1>,
  "save_likelihood": <0-1>,
  "comment_likelihood": <0-1>,
  "subscribe_likelihood": <0-1>,
  "retention_score": <0-100 overall retention score>,
  "engagement_rate": <0-1 composite engagement>,
  "explanation": "<1-2 sentence reasoning>"
}}

Respond ONLY with valid JSON, no markdown, no code fences."""

    def _build_prediction(self, parsed: dict, formula_score: float) -> RetentionPrediction:
        """Build RetentionPrediction from parsed LLM response."""
        retention = parsed.get("retention_score", formula_score)
        
        return RetentionPrediction(
            avg_watch_time_sec=float(parsed.get("avg_watch_time_sec", 0)),
            completion_rate=float(parsed.get("completion_rate", 0)),
            drop_off_rate=float(parsed.get("drop_off_rate", 0)),
            drop_off_points=[float(x) for x in parsed.get("drop_off_points", [])],
            replay_likelihood=float(parsed.get("replay_likelihood", 0)),
            share_likelihood=float(parsed.get("share_likelihood", 0)),
            save_likelihood=float(parsed.get("save_likelihood", 0)),
            comment_likelihood=float(parsed.get("comment_likelihood", 0)),
            subscribe_likelihood=float(parsed.get("subscribe_likelihood", 0)),
            retention_score=float(retention),
            engagement_rate=float(parsed.get("engagement_rate", 0)),
        )

    def _fallback_prediction(
        self,
        scores: dict,
        duration: float,
    ) -> RetentionPrediction:
        """Derive retention prediction from dimension scores without LLM."""
        formula_score = self._compute_formula_score(scores)
        
        # Convert 0-100 score to completion probability
        completion = min(1.0, max(0.0, formula_score / 100 * 0.85))
        
        # Average watch time estimate
        avg_watch = duration * completion
        
        # Engagement composite
        engagement = (
            scores.get("share_probability", 50) * 0.25 +
            scores.get("comment_potential", 50) * 0.25 +
            scores.get("replay_probability", 50) * 0.20 +
            scores.get("subscribe_conversion_probability", 50) * 0.15 +
            scores.get("virality", 50) * 0.15
        ) / 100

        return RetentionPrediction(
            avg_watch_time_sec=avg_watch,
            completion_rate=completion,
            drop_off_rate=1.0 - completion,
            replay_likelihood=scores.get("replay_probability", 50) / 100,
            share_likelihood=scores.get("share_probability", 50) / 100,
            save_likelihood=scores.get("educational_value", 50) / 100,
            comment_likelihood=scores.get("comment_potential", 50) / 100,
            subscribe_likelihood=scores.get("subscribe_conversion_probability", 50) / 100,
            retention_score=formula_score,
            engagement_rate=engagement,
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
