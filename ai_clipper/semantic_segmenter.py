"""
Semantic Segmentation Engine — Step 3 of the Master Prompt.

Splits transcript by meaning, not by time. Every segment represents exactly
one complete idea. Boundaries land at natural locations:
  - Sentence endings
  - Topic transitions
  - Breath pauses
  - Story beats
  - Emotional changes

Never cuts mid-sentence, mid-joke, mid-explanation, mid-emotion, or mid-reveal.
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass, field
from typing import List, Optional, Callable

from .transcriber import WordTimestamp

logger = logging.getLogger(__name__)


@dataclass
class SemanticSegment:
    """One complete idea from the transcript."""
    id: int
    start_time: float           # seconds from video start
    end_time: float             # seconds from video start
    text: str                   # full text of this segment
    words: List[WordTimestamp] = field(default_factory=list)
    
    # Metadata
    topic: str = ""             # what this segment is about
    speaker: str = ""           # who's speaking (if diarized)
    emotion: str = ""           # dominant emotion
    segment_type: str = ""      # "hook", "story", "argument", "example", "advice", "joke", "revelation", etc.
    is_complete: bool = True    # does this contain a complete idea?


SEMANTIC_SEGMENT_PROMPT = """You are an elite video editor analyzing a transcript. Your job is to split this transcript into SEMANTIC SEGMENTS — each one representing exactly ONE complete idea.

RULES:
1. Split by MEANING, never by time or word count.
2. Every segment must contain ONE complete idea: one point, one story beat, one argument, one piece of advice, one joke, one revelation.
3. Place boundaries at:
   - Natural sentence endings
   - Topic transitions
   - Speaker changes
   - Emotional shifts
   - Story beat transitions
4. NEVER split:
   - Mid-sentence
   - Mid-joke (before the punchline)
   - Mid-explanation
   - Mid-emotional moment
   - Mid-reveal
5. A segment can be 2 sentences or 20 — what matters is that it's ONE complete idea.
6. For each segment, identify:
   - The dominant TOPIC (3-5 words)
   - The segment TYPE: hook, story, argument, example, advice, humor, revelation, question, setup, payoff, transition, introduction, conclusion
   - The dominant EMOTION: excited, serious, funny, emotional, informative, controversial, vulnerable, energetic, calm, suspenseful
   - Whether it's COMPLETE (has a clear beginning, middle, and end/resolution)

TRANSCRIPT:
{text}

Respond with a JSON array of segments. Each segment must have:
- "start_idx": word index where segment begins (0-based from the transcript)
- "end_idx": word index where segment ends (inclusive)
- "topic": short topic description
- "segment_type": one of the types above
- "emotion": dominant emotion
- "is_complete": true/false

Example:
[
  {{"start_idx": 0, "end_idx": 45, "topic": "introducing the main argument", "segment_type": "hook", "emotion": "energetic", "is_complete": true}},
  {{"start_idx": 46, "end_idx": 120, "topic": "personal story about failure", "segment_type": "story", "emotion": "vulnerable", "is_complete": true}}
]

Respond ONLY with the JSON array, no markdown, no code fences."""


class SemanticSegmenter:
    """LLM-powered semantic segmentation of transcripts."""

    def __init__(self, llm_call_fn: Callable[[str], str]):
        self._call_llm = llm_call_fn

    def segment(
        self,
        words: List[WordTimestamp],
        max_words_per_chunk: int = 800,
    ) -> List[SemanticSegment]:
        """
        Split a transcript into semantic segments.

        For long transcripts, processes in overlapping chunks to keep
        within LLM context limits, then merges results.

        Args:
            words: Word-level timestamp list from transcription
            max_words_per_chunk: Max words per LLM call

        Returns:
            List of SemanticSegments with complete idea boundaries
        """
        if not words:
            return []

        total_words = len(words)
        logger.info(f"Semantic segmentation: {total_words} words")

        if total_words <= max_words_per_chunk:
            return self._segment_chunk(words, 0)

        # Chunk with overlap for continuity
        all_segments: List[SemanticSegment] = []
        chunk_size = max_words_per_chunk
        overlap = chunk_size // 4  # 25% overlap
        
        for offset in range(0, total_words, chunk_size - overlap):
            chunk_words = words[offset: offset + chunk_size]
            if len(chunk_words) < 20:
                break
            
            chunk_segments = self._segment_chunk(chunk_words, offset)
            all_segments.extend(chunk_segments)

        # Merge overlapping segments
        all_segments = self._merge_overlapping_segments(all_segments)
        
        logger.info(f"Semantic segmentation complete: {len(all_segments)} segments")
        return all_segments

    def _segment_chunk(
        self,
        words: List[WordTimestamp],
        global_offset: int,
    ) -> List[SemanticSegment]:
        """Segment a chunk of words using the LLM."""
        # Build indexed transcript for the LLM
        indexed_lines = []
        for i, w in enumerate(words):
            idx = global_offset + i
            indexed_lines.append(f"[{idx}] {w.word}")
        
        # Truncate if needed
        text_block = "\n".join(indexed_lines)
        if len(text_block) > 15000:
            text_block = text_block[:15000] + "\n...[truncated]"

        prompt = SEMANTIC_SEGMENT_PROMPT.format(text=text_block)

        try:
            response = self._call_llm(prompt)
            segments_data = self._parse_segments_json(response)

            if not segments_data:
                # Fallback: create segments by sentence boundaries
                logger.warning("LLM segmentation failed, using sentence-based fallback")
                return self._sentence_fallback(words, global_offset)

            segments = []
            for i, seg_data in enumerate(segments_data):
                start_idx = seg_data.get("start_idx", 0)
                end_idx = seg_data.get("end_idx", len(words) - 1)
                
                # Clamp to valid range
                start_idx = max(0, min(start_idx, len(words) - 1))
                end_idx = max(start_idx, min(end_idx, len(words) - 1))
                
                seg_words = words[start_idx: end_idx + 1]
                if not seg_words:
                    continue

                segment = SemanticSegment(
                    id=global_offset + i,
                    start_time=seg_words[0].start,
                    end_time=seg_words[-1].end,
                    text=" ".join(w.word for w in seg_words),
                    words=seg_words,
                    topic=seg_data.get("topic", ""),
                    speaker="",
                    emotion=seg_data.get("emotion", ""),
                    segment_type=seg_data.get("segment_type", ""),
                    is_complete=seg_data.get("is_complete", True),
                )
                segments.append(segment)

            return segments

        except Exception as e:
            logger.error(f"Semantic segmentation error: {e}")
            return self._sentence_fallback(words, global_offset)

    def _sentence_fallback(
        self,
        words: List[WordTimestamp],
        global_offset: int,
    ) -> List[SemanticSegment]:
        """Fallback: segment by sentence-ending punctuation."""
        segments = []
        current_words = []
        
        for i, w in enumerate(words):
            current_words.append(w)
            # Sentence boundary: . ! ? followed by pause or end
            word_text = w.word.strip()
            if word_text and word_text[-1] in ".!?":
                if len(current_words) >= 5:  # minimum meaningful sentence
                    segments.append(SemanticSegment(
                        id=global_offset + len(segments),
                        start_time=current_words[0].start,
                        end_time=current_words[-1].end,
                        text=" ".join(w.word for w in current_words),
                        words=list(current_words),
                        topic="",
                        is_complete=True,
                    ))
                    current_words = []

        # Don't lose trailing words
        if current_words and len(current_words) >= 3:
            segments.append(SemanticSegment(
                id=global_offset + len(segments),
                start_time=current_words[0].start,
                end_time=current_words[-1].end,
                text=" ".join(w.word for w in current_words),
                words=list(current_words),
                topic="",
                is_complete=True,
            ))

        return segments

    def _merge_overlapping_segments(
        self,
        segments: List[SemanticSegment],
    ) -> List[SemanticSegment]:
        """Remove duplicate/overlapping segments from chunked processing."""
        if not segments:
            return segments

        segments.sort(key=lambda s: s.start_time)
        merged = []
        
        for seg in segments:
            if not merged:
                merged.append(seg)
                continue
            
            last = merged[-1]
            # If this segment substantially overlaps with the last, skip or merge
            overlap = min(last.end_time, seg.end_time) - max(last.start_time, seg.start_time)
            seg_duration = seg.end_time - seg.start_time
            
            if overlap > 0 and seg_duration > 0:
                overlap_ratio = overlap / seg_duration
                if overlap_ratio > 0.7:
                    # This segment is mostly covered; skip
                    continue
                elif overlap_ratio > 0.3:
                    # Partial overlap — prefer the one that's marked complete
                    if seg.is_complete and not last.is_complete:
                        merged[-1] = seg
                    continue
            
            merged.append(seg)

        # Re-index
        for i, seg in enumerate(merged):
            seg.id = i

        return merged

    @staticmethod
    def _parse_segments_json(text: str) -> Optional[List[dict]]:
        """Parse JSON array from LLM response."""
        if not text:
            return None
        text = re.sub(r'^```(?:json)?\s*', '', text.strip(), flags=re.MULTILINE)
        text = re.sub(r'\s*```$', '', text.strip(), flags=re.MULTILINE)
        
        # Try to find JSON array
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
