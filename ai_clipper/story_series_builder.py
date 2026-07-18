"""
Story Series Builder — Output B of the Master Prompt.

Creates exactly 3 Story Series.
Each series contains 3 episodes (~2 min each).
Episode 2 begins exactly where Episode 1 ends.
Episode 3 begins exactly where Episode 2 ends.
No skipped footage, no overlap, no repeated scenes.

Episode Structure:
  Episode 1: Hook → Introduce story → Introduce conflict → Cliffhanger
  Episode 2: Escalation → Twist → New discoveries → Build suspense → Stronger cliffhanger
  Episode 3: Resolution → Lesson → Payoff → Strong emotional ending
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass, field
from typing import List, Optional, Callable

from pydantic import BaseModel

from .semantic_segmenter import SemanticSegment
from .scorer import ScoredSegment, ViralScorer

logger = logging.getLogger(__name__)


class Episode(BaseModel):
    title: str = ""
    start_time: float = 0.0
    end_time: float = 0.0
    duration: float = 0.0
    cliffhanger: str = ""
    ending: str = ""
    text: str = ""
    score: float = 0.0


class StorySeries(BaseModel):
    title: str = ""
    overall_story_score: float = 0.0
    episode_1: Episode = field(default_factory=Episode)
    episode_2: Episode = field(default_factory=Episode)
    episode_3: Episode = field(default_factory=Episode)


STORY_SERIES_PROMPT = """You are an elite documentary and YouTube series editor.

Analyze these semantic segments and identify the 3 BEST narrative arcs that can be split into 3 continuous episodes (~2 minutes each).

SEGMENTS:
{segments_json}

Rules for each story series:
1. Episode 1 (hook + setup + conflict + cliffhanger): ~120 seconds
2. Episode 2 (escalation + twist + discoveries + stronger cliffhanger): ~120 seconds, begins exactly where Episode 1 ends
3. Episode 3 (resolution + lesson + payoff + emotional ending): ~120 seconds, begins exactly where Episode 2 ends
4. No skipped footage, no overlap, no repeated scenes between episodes.
5. Together, all 3 episodes must tell one complete story.
6. Each episode should be understandable on its own but stronger as part of the series.
7. Choose arcs with strong emotional progression and a satisfying payoff.

For each of the 3 story series, return:
- series title
- overall_story_score (0-100)
- episode_1: title, start_segment_id, end_segment_id, cliffhanger text
- episode_2: title, start_segment_id, end_segment_id, cliffhanger text
- episode_3: title, start_segment_id, end_segment_id, ending text

Respond ONLY with a JSON array of 3 objects. No markdown, no code fences.

Example format:
[
  {{
    "title": "The Betrayal and Redemption",
    "overall_story_score": 94.5,
    "episode_1": {{"title": "The Setup", "start_segment_id": 5, "end_segment_id": 8, "cliffhanger": "..."}},
    "episode_2": {{"title": "The Twist", "start_segment_id": 9, "end_segment_id": 12, "cliffhanger": "..."}},
    "episode_3": {{"title": "The Payoff", "start_segment_id": 13, "end_segment_id": 16, "ending": "..."}}
  }}
]"""


class StorySeriesBuilder:
    """Builds 3 continuous story series from semantic segments."""

    def __init__(
        self,
        scorer: ViralScorer,
        episode_target_duration: float = 120.0,
        tolerance: float = 10.0,
    ):
        self._scorer = scorer
        self.episode_target_duration = episode_target_duration
        self.tolerance = tolerance

    def build_series(
        self,
        segments: List[SemanticSegment],
        all_words: List,
    ) -> List[StorySeries]:
        """
        Build 3 story series from semantic segments.

        Args:
            segments: Semantic segments
            all_words: Full word-level transcript

        Returns:
            List of 3 StorySeries
        """
        if len(segments) < 9:
            logger.warning("Not enough semantic segments for story series; need at least 9")
            return []

        # Build segment JSON for LLM
        segments_json = json.dumps(
            [
                {
                    "id": s.id,
                    "start": s.start_time,
                    "end": s.end_time,
                    "topic": s.topic,
                    "type": s.segment_type,
                    "emotion": s.emotion,
                    "is_complete": s.is_complete,
                    "text_preview": s.text[:200],
                }
                for s in segments
            ],
            ensure_ascii=False,
        )

        # Truncate if too long
        if len(segments_json) > 12000:
            segments_json = segments_json[:12000] + "\n...[truncated]"

        prompt = STORY_SERIES_PROMPT.format(segments_json=segments_json)

        try:
            response = self._scorer.call_llm(prompt)
            parsed = self._parse_json(response)
        except Exception as e:
            logger.error(f"Story series LLM call failed: {e}")
            return self._fallback_series(segments, all_words)

        if not parsed or not isinstance(parsed, list):
            logger.warning("Story series response invalid, using fallback")
            return self._fallback_series(segments, all_words)

        series_list = []
        for series_data in parsed[:3]:
            series = self._build_single_series(series_data, segments, all_words)
            if series:
                series_list.append(series)

        # If we got fewer than 3, fill with fallback
        while len(series_list) < 3:
            fallback = self._fallback_single_series(segments, all_words, series_list)
            if fallback:
                series_list.append(fallback)
            else:
                break

        return series_list

    def _build_single_series(
        self,
        series_data: dict,
        segments: List[SemanticSegment],
        all_words: List,
    ) -> Optional[StorySeries]:
        """Build a single StorySeries from LLM output."""
        segment_map = {s.id: s for s in segments}
        
        episodes = []
        episode_keys = ["episode_1", "episode_2", "episode_3"]
        prev_end_id = None
        
        for key in episode_keys:
            ep_data = series_data.get(key, {})
            start_id = ep_data.get("start_segment_id")
            end_id = ep_data.get("end_segment_id")
            
            if start_id not in segment_map or end_id not in segment_map:
                return None
            
            # Ensure continuity
            if prev_end_id is not None and start_id != prev_end_id:
                # LLM may have small gaps; force continuity
                start_id = prev_end_id
            
            start_idx = next((i for i, s in enumerate(segments) if s.id == start_id), -1)
            end_idx = next((i for i, s in enumerate(segments) if s.id == end_id), -1)
            
            if start_idx < 0 or end_idx < start_idx:
                return None
            
            combo = segments[start_idx: end_idx + 1]
            start_time = combo[0].start_time
            end_time = combo[-1].end_time
            text = self._get_text_between(all_words, start_time, end_time)
            
            # Score the episode
            scored = self._scorer.score_segment(start_time, end_time, text)
            score = scored.score if scored else 50.0
            
            episode = Episode(
                title=ep_data.get("title", ""),
                start_time=start_time,
                end_time=end_time,
                duration=end_time - start_time,
                cliffhanger=ep_data.get("cliffhanger", ""),
                ending=ep_data.get("ending", "") if key == "episode_3" else "",
                text=text,
                score=score,
            )
            episodes.append(episode)
            prev_end_id = end_id

        if len(episodes) != 3:
            return None

        overall = sum(ep.score for ep in episodes) / 3
        
        return StorySeries(
            title=series_data.get("title", "Story Series"),
            overall_story_score=float(series_data.get("overall_story_score", overall)),
            episode_1=episodes[0],
            episode_2=episodes[1],
            episode_3=episodes[2],
        )

    def _fallback_series(
        self,
        segments: List[SemanticSegment],
        all_words: List,
    ) -> List[StorySeries]:
        """Build story series using simple heuristics when LLM fails."""
        series_list = []
        
        # Try 3 different starting points spread across the video
        start_indices = [0, len(segments) // 4, len(segments) // 2]
        
        for start_idx in start_indices:
            series = self._build_continuous_series(segments, all_words, start_idx)
            if series:
                series_list.append(series)
            if len(series_list) >= 3:
                break

        return series_list

    def _fallback_single_series(
        self,
        segments: List[SemanticSegment],
        all_words: List,
        existing: List[StorySeries],
    ) -> Optional[StorySeries]:
        """Create one additional fallback series avoiding used segments."""
        used_ids = set()
        for s in existing:
            # Find segments covered by this series
            for seg in segments:
                if (seg.start_time >= s.episode_1.start_time and 
                    seg.end_time <= s.episode_3.end_time):
                    used_ids.add(seg.id)
        
        available = [s for s in segments if s.id not in used_ids]
        if len(available) < 9:
            return None
        
        return self._build_continuous_series(available, all_words, 0)

    def _build_continuous_series(
        self,
        segments: List[SemanticSegment],
        all_words: List,
        start_idx: int,
    ) -> Optional[StorySeries]:
        """Build a continuous 3-episode series from consecutive segments."""
        n = len(segments)
        if start_idx + 9 > n:
            return None
        
        # Build episodes by grouping segments to reach target duration
        episodes = []
        current_idx = start_idx
        
        for ep_num in range(3):
            combo = []
            duration = 0.0
            
            while current_idx < n and duration < self.episode_target_duration - self.tolerance:
                combo.append(segments[current_idx])
                duration = combo[-1].end_time - combo[0].start_time
                current_idx += 1
                
                if duration >= self.episode_target_duration + self.tolerance:
                    break
            
            if not combo:
                return None
            
            start_time = combo[0].start_time
            end_time = combo[-1].end_time
            text = self._get_text_between(all_words, start_time, end_time)
            
            scored = self._scorer.score_segment(start_time, end_time, text)
            score = scored.score if scored else 50.0
            
            episode = Episode(
                title=f"Episode {ep_num + 1}",
                start_time=start_time,
                end_time=end_time,
                duration=end_time - start_time,
                cliffhanger="Continues in the next episode..." if ep_num < 2 else "",
                ending="The story concludes." if ep_num == 2 else "",
                text=text,
                score=score,
            )
            episodes.append(episode)

        overall = sum(ep.score for ep in episodes) / 3
        
        return StorySeries(
            title=f"Story Arc {start_idx + 1}",
            overall_story_score=overall,
            episode_1=episodes[0],
            episode_2=episodes[1],
            episode_3=episodes[2],
        )

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

    @staticmethod
    def _parse_json(text: str) -> Optional[list]:
        if not text:
            return None
        text = re.sub(r'^```(?:json)?\s*', '', text.strip(), flags=re.MULTILINE)
        text = re.sub(r'\s*```$', '', text.strip(), flags=re.MULTILINE)
        match = re.search(r'\[.*\]', text, re.DOTALL)
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
