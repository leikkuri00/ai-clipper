"""
Candidate Generation Engine — Step 5 of the Master Prompt.

Generates hundreds of possible clip candidates by combining semantic segments.

Strategy:
  1. Start from high-quality semantic segment boundaries.
  2. Build candidates by combining 1-3 consecutive segments to reach target duration.
  3. Extend/trim boundaries to natural sentence endings.
  4. Vary start/end points around strong hooks and payoffs.
  5. Generate at least 300 candidates for the optimizer to choose from.

Never uses fixed intervals. Every candidate begins and ends at semantic boundaries.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import List, Tuple

from .semantic_segmenter import SemanticSegment

logger = logging.getLogger(__name__)


@dataclass
class ClipCandidate:
    """A potential clip with semantic boundaries."""
    start_time: float
    end_time: float
    text: str
    segment_ids: List[int] = field(default_factory=list)
    duration: float = 0.0
    hook_segment_id: int = -1
    payoff_segment_id: int = -1
    estimated_score: float = 0.0  # fast heuristic score from segment overlap


def find_natural_word_boundary(
    words: List,
    target_time: float,
    direction: str = "backward",
) -> float:
    """
    Find the nearest natural word boundary to a target time.
    Prefers sentence-ending punctuation and pauses.
    """
    if not words:
        return target_time

    best_time = target_time
    best_score = -1

    candidates = []
    if direction == "backward":
        candidates = reversed(words)
    else:
        candidates = words

    for w in candidates:
        # Only consider boundaries within 5 seconds
        if abs(w.end - target_time) > 5.0:
            continue

        score = 10.0  # base for being a word boundary
        word_text = w.word.strip()
        
        # Strong preference for sentence endings
        if word_text and word_text[-1] in ".!?":
            score += 50
        elif word_text and word_text[-1] in ",;:":
            score += 20
        
        # Prefer boundaries closer to target
        distance = abs(w.end - target_time)
        score -= distance * 3  # penalty for distance

        if score > best_score:
            best_score = score
            best_time = w.end

    return best_time


class CandidateGenerator:
    """Generates clip candidates from semantic segments."""

    def __init__(
        self,
        target_duration: float = 148.0,  # 2:28
        tolerance: float = 3.0,
        min_duration: float = 30.0,
        max_duration: float = 180.0,
    ):
        self.target_duration = target_duration
        self.tolerance = tolerance
        self.min_duration = min_duration
        self.max_duration = max_duration

    def generate(
        self,
        segments: List[SemanticSegment],
        all_words: List,
        min_candidates: int = 300,
    ) -> List[ClipCandidate]:
        """
        Generate clip candidates from semantic segments.

        Args:
            segments: Semantic segments (each one complete idea)
            all_words: Full word-level transcript
            min_candidates: Minimum number of candidates to produce

        Returns:
            List of ClipCandidate with natural boundaries
        """
        if not segments:
            return []

        candidates: List[ClipCandidate] = []

        # Strategy 1: Single strong segments extended to target duration
        candidates.extend(self._single_segment_candidates(segments, all_words))

        # Strategy 2: Combine 2-3 consecutive segments
        candidates.extend(self._combined_segment_candidates(segments, all_words))

        # Strategy 3: Variations around strong hooks/payoffs
        candidates.extend(self._hook_payoff_variations(segments, all_words))

        # Strategy 4: Longer narrative arcs (4-6 segments)
        candidates.extend(self._narrative_arc_candidates(segments, all_words))

        # Deduplicate and filter
        candidates = self._deduplicate(candidates)
        candidates = self._filter_by_duration(candidates)

        # If we still need more, generate trimmed variants
        if len(candidates) < min_candidates:
            extra = self._generate_trimmed_variants(candidates, segments, all_words, min_candidates - len(candidates))
            candidates.extend(extra)
            candidates = self._deduplicate(candidates)
            candidates = self._filter_by_duration(candidates)

        logger.info(f"Generated {len(candidates)} clip candidates")
        return candidates

    def _single_segment_candidates(
        self,
        segments: List[SemanticSegment],
        all_words: List,
    ) -> List[ClipCandidate]:
        """Candidates built around a single strong segment, extended to target duration."""
        candidates = []
        
        for seg in segments:
            duration = seg.end_time - seg.start_time
            if duration < 10:
                continue

            if self._in_target_range(duration):
                # Already good length
                candidates.append(self._build_candidate([seg], all_words))
            elif duration < self.target_duration - self.tolerance:
                # Try extending forward with next segments
                extended = self._extend_forward(seg, segments, all_words)
                if extended:
                    candidates.append(extended)
            elif duration > self.target_duration + self.tolerance:
                # Try trimming to a natural boundary
                trimmed = self._trim_to_target(seg, all_words)
                if trimmed:
                    candidates.append(trimmed)

        return candidates

    def _combined_segment_candidates(
        self,
        segments: List[SemanticSegment],
        all_words: List,
    ) -> List[ClipCandidate]:
        """Candidates combining 2-6 consecutive segments."""
        candidates = []
        n = len(segments)
        
        for i in range(n):
            for length in range(2, min(7, n - i + 1)):
                combo = segments[i: i + length]
                duration = combo[-1].end_time - combo[0].start_time
                
                if duration < self.min_duration or duration > self.max_duration:
                    continue
                
                # Only keep combinations that form a coherent narrative arc
                # Heuristic: first segment is a hook/story/setup, last has payoff/conclusion
                has_hook = any(s.segment_type in ("hook", "story", "question", "setup") for s in combo[:2])
                has_payoff = any(s.segment_type in ("payoff", "conclusion", "revelation", "advice") for s in combo[-2:])
                
                if has_hook or has_payoff or length <= 3:
                    candidates.append(self._build_candidate(combo, all_words))

        return candidates

    def _hook_payoff_variations(
        self,
        segments: List[SemanticSegment],
        all_words: List,
    ) -> List[ClipCandidate]:
        """Generate candidates that start at hooks and end at payoffs."""
        candidates = []
        n = len(segments)
        
        hook_indices = [i for i, s in enumerate(segments) if s.segment_type in ("hook", "question", "setup", "story")]
        payoff_indices = [i for i, s in enumerate(segments) if s.segment_type in ("payoff", "conclusion", "revelation", "advice")]
        
        for hook_idx in hook_indices:
            for payoff_idx in payoff_indices:
                if payoff_idx <= hook_idx:
                    continue
                
                duration = segments[payoff_idx].end_time - segments[hook_idx].start_time
                if duration < self.min_duration or duration > self.max_duration:
                    continue
                
                combo = segments[hook_idx: payoff_idx + 1]
                candidates.append(self._build_candidate(combo, all_words))

        return candidates

    def _narrative_arc_candidates(
        self,
        segments: List[SemanticSegment],
        all_words: List,
    ) -> List[ClipCandidate]:
        """Generate longer narrative arc candidates (4-8 segments)."""
        candidates = []
        n = len(segments)
        
        for i in range(n):
            for length in range(4, min(9, n - i + 1)):
                combo = segments[i: i + length]
                duration = combo[-1].end_time - combo[0].start_time
                
                if duration < self.target_duration - self.tolerance or duration > self.max_duration:
                    continue
                
                candidates.append(self._build_candidate(combo, all_words))

        return candidates

    def _extend_forward(
        self,
        seg: SemanticSegment,
        segments: List[SemanticSegment],
        all_words: List,
    ) -> Optional[ClipCandidate]:
        """Extend a segment forward with following segments until target duration."""
        idx = next((i for i, s in enumerate(segments) if s.id == seg.id), -1)
        if idx < 0:
            return None
        
        combo = [seg]
        current_duration = seg.end_time - seg.start_time
        
        for next_seg in segments[idx + 1:]:
            new_duration = next_seg.end_time - combo[0].start_time
            if new_duration > self.max_duration:
                break
            combo.append(next_seg)
            current_duration = new_duration
            if self._in_target_range(current_duration):
                break
        
        if current_duration < self.min_duration:
            return None
        
        return self._build_candidate(combo, all_words)

    def _trim_to_target(
        self,
        seg: SemanticSegment,
        all_words: List,
    ) -> Optional[ClipCandidate]:
        """Trim a long segment to target duration at natural boundaries."""
        duration = seg.end_time - seg.start_time
        if duration < self.target_duration - self.tolerance:
            return None
        
        # Find a natural end point near target duration
        target_end = seg.start_time + self.target_duration
        natural_end = find_natural_word_boundary(all_words, target_end, direction="backward")
        
        # Ensure we don't cut before minimum duration
        if natural_end - seg.start_time < self.min_duration:
            natural_end = seg.start_time + self.min_duration
        
        if natural_end > seg.end_time:
            natural_end = seg.end_time
        
        return ClipCandidate(
            start_time=seg.start_time,
            end_time=natural_end,
            text=seg.text[:2000],
            segment_ids=[seg.id],
            duration=natural_end - seg.start_time,
        )

    def _generate_trimmed_variants(
        self,
        existing: List[ClipCandidate],
        segments: List[SemanticSegment],
        all_words: List,
        needed: int,
    ) -> List[ClipCandidate]:
        """Generate additional candidates by trimming existing ones at word boundaries."""
        variants = []
        
        for cand in existing:
            if len(variants) >= needed:
                break
            
            # Try +/- 2 second trims at start and end
            for delta in [-2.0, -1.0, 1.0, 2.0]:
                new_start = find_natural_word_boundary(all_words, cand.start_time + delta, direction="forward")
                new_end = find_natural_word_boundary(all_words, cand.end_time + delta, direction="backward")
                
                duration = new_end - new_start
                if duration < self.min_duration or duration > self.max_duration:
                    continue
                
                variants.append(ClipCandidate(
                    start_time=new_start,
                    end_time=new_end,
                    text=cand.text,
                    segment_ids=list(cand.segment_ids),
                    duration=duration,
                ))
                
                if len(variants) >= needed:
                    break

        return variants

    def _build_candidate(
        self,
        segments: List[SemanticSegment],
        all_words: List,
    ) -> ClipCandidate:
        """Build a ClipCandidate from a list of segments."""
        start = segments[0].start_time
        end = segments[-1].end_time
        
        # Refine boundaries to natural word boundaries
        start = find_natural_word_boundary(all_words, start, direction="forward")
        end = find_natural_word_boundary(all_words, end, direction="backward")
        
        # Get text from all_words within refined boundaries
        candidate_words = [w for w in all_words if w.start >= start and w.end <= end]
        text = " ".join(w.word for w in candidate_words)
        
        # Identify hook and payoff segments
        hook_id = -1
        payoff_id = -1
        for s in segments:
            if hook_id < 0 and s.segment_type in ("hook", "question", "setup"):
                hook_id = s.id
            if s.segment_type in ("payoff", "conclusion", "revelation", "advice"):
                payoff_id = s.id
        
        return ClipCandidate(
            start_time=start,
            end_time=end,
            text=text,
            segment_ids=[s.id for s in segments],
            duration=end - start,
            hook_segment_id=hook_id,
            payoff_segment_id=payoff_id,
        )

    def _in_target_range(self, duration: float) -> bool:
        """Check if duration is within target ± tolerance."""
        return abs(duration - self.target_duration) <= self.tolerance

    def _filter_by_duration(self, candidates: List[ClipCandidate]) -> List[ClipCandidate]:
        """Keep only candidates within acceptable duration range."""
        return [c for c in candidates if self.min_duration <= c.duration <= self.max_duration]

    def _deduplicate(self, candidates: List[ClipCandidate]) -> List[ClipCandidate]:
        """Remove candidates with nearly identical boundaries."""
        if not candidates:
            return candidates
        
        # Sort by start time
        candidates.sort(key=lambda c: (c.start_time, c.end_time))
        unique = []
        
        for cand in candidates:
            is_dup = False
            for existing in unique:
                # Consider duplicate if boundaries within 1 second
                if (abs(cand.start_time - existing.start_time) < 1.0 and
                    abs(cand.end_time - existing.end_time) < 1.0):
                    is_dup = True
                    break
            if not is_dup:
                unique.append(cand)
        
        return unique
