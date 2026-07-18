"""
Main pipeline: orchestrates the full Master Prompt flow.

Flow:
  1. Source resolution (download or local file)
  2. Complete transcription (word-level timestamps + optional diarization)
  3. Multimodal analysis (audio + video features)
  4. Semantic segmentation (by meaning, not time)
  5. 26-dimension scoring of every segment
  6. Generate 300+ clip candidates
  7. Retention prediction for every candidate
  8. Iterative optimization of top candidates
  9. Select 8 standalone clips (2:28 each, diverse topics)
  10. Build 3 story series (3 episodes each, continuous)
  11. Moderation, clipping, captioning, gallery generation
"""

from __future__ import annotations

import json
import logging
import re
import shutil
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional
from uuid import uuid4

from .config import ClipperConfig
from .transcriber import transcribe_audio
from .scorer import ViralScorer, ScoredSegment
from .clip_cutter import extract_audio, cut_and_caption_clip
from .audio_analyzer import analyze_audio_energy, detect_high_energy_regions
from .moderation import moderate_segment
from .downloader import DEFAULT_DOWNLOAD_ROOT, prepare_download_dir, is_url, download_video, get_video_info
from .html_gallery import generate_gallery

# New master-prompt modules
from .semantic_segmenter import SemanticSegmenter, SemanticSegment
from .multimodal_analyzer import compute_multimodal_profile, MultimodalSegment
from .candidate_generator import CandidateGenerator, ClipCandidate
from .retention_predictor import RetentionPredictor
from .clip_optimizer import ClipOptimizer, OptimizedClip
from .story_series_builder import StorySeriesBuilder, StorySeries

logger = logging.getLogger(__name__)


@dataclass
class ClipResult:
    """Result of a clipping run."""
    clips: List[Path] = field(default_factory=list)
    gallery_path: Optional[Path] = None
    report: dict = field(default_factory=dict)
    story_series: List[StorySeries] = field(default_factory=list)


class AIClipper:
    """
    AI-powered video clipping tool implementing the Master Prompt.
    """

    def __init__(self, config: ClipperConfig | None = None):
        self.config = config or ClipperConfig()
        self.config.apply_platform_preset()
        self.config.output_dir.mkdir(parents=True, exist_ok=True)
        self.clips_dir = self.config.output_dir / "clips"
        self.clips_dir.mkdir(parents=True, exist_ok=True)
        
        self.scorer = ViralScorer(self.config)
        self.segmenter = SemanticSegmenter(self.scorer.call_llm)
        self.retention_predictor = RetentionPredictor(self.scorer.call_llm)
        self.candidate_generator = CandidateGenerator(
            target_duration=self.config.target_clip_duration,
            tolerance=self.config.clip_duration_tolerance,
            min_duration=self.config.min_clip_duration,
            max_duration=self.config.max_clip_duration,
        )
        self.clip_optimizer = ClipOptimizer(
            scorer=self._score_candidate,
            retention_predictor=self.retention_predictor,
            target_duration=self.config.target_clip_duration,
            tolerance=self.config.clip_duration_tolerance,
            min_duration=self.config.min_clip_duration,
            max_duration=self.config.max_clip_duration,
            max_rounds=self.config.iterative_rounds,
        )
        self.story_builder = StorySeriesBuilder(
            scorer=self.scorer,
            episode_target_duration=self.config.episode_target_duration,
            num_series=self.config.num_story_series,
        )
        self._scored_segments_cache: List[ScoredSegment] = []

    def _score_candidate(
        self,
        start: float,
        end: float,
        text: str,
    ) -> Optional[ScoredSegment]:
        """Score a clip candidate. Uses cached content type."""
        return self.scorer.score_segment(
            start, end, text,
            content_type=self.scorer._content_type or "unknown"
        )

    # ── Source resolution ───────────────────────────────────

    def _resolve_source(self, video_path: str | Path | None) -> tuple[Path, Optional[Path]]:
        """Resolve the input to a local video file."""
        if video_path is not None:
            src = str(video_path)
        elif self.config.source_path is not None:
            src = str(self.config.source_path)
        elif self.config.source_url:
            src = self.config.source_url
        else:
            raise ValueError("No source provided: pass a path/URL or set source_url/source_path.")

        if is_url(src):
            try:
                info = get_video_info(
                    src,
                    cookies_from_browser=self.config.cookies_from_browser,
                    cookiefile=self.config.cookiefile,
                )
                logger.info(f"Source: {info['title']} ({info['duration']}s) by {info['uploader']}")
            except Exception as e:
                logger.warning(f"Could not fetch video info: {e}")
            download_dir = prepare_download_dir(f"aiclip_{uuid4().hex}")
            logger.info(f"Downloading video to E: drive: {download_dir}")
            path = download_video(
                src,
                download_dir,
                max_resolution=self.config.max_resolution,
                cookies_from_browser=self.config.cookies_from_browser,
                cookiefile=self.config.cookiefile,
            )
            return path, download_dir

        path = Path(src)
        if not path.exists():
            raise FileNotFoundError(f"Video not found: {path}")
        return path, None

    # ── Main entry point ─────────────────────────────────────

    def run(self, video_path: str | Path | None = None) -> ClipResult:
        """Process a video end-to-end using the master prompt pipeline."""
        video_path, download_dir = self._resolve_source(video_path)

        logger.info(f"{'='*60}")
        logger.info(f"AI Clipper Master Engine — Processing: {video_path.name}")
        logger.info(f"{'='*60}")

        try:
            # Step 1: Extract audio
            logger.info("[1/9] Extracting audio...")
            audio_path = extract_audio(video_path)

            # Step 2: Transcribe
            logger.info("[2/9] Transcribing with Whisper...")
            transcript = transcribe_audio(
                audio_path,
                provider=self.config.transcribe_provider,
                model_size=self.config.whisper_model,
                device=self.config.device,
                compute_type=self.config.compute_type,
                language=self.config.language,
                api_key=self.config.groq_api_key,
                offline_mode=self.config.offline_mode,
            )

            # Step 3: Classify content type
            logger.info("[3/9] Classifying content type...")
            content_type = self.scorer.classify_content_type(transcript.full_text)

            # Step 4: Semantic segmentation
            logger.info("[4/9] Semantic segmentation (by meaning, not time)...")
            semantic_segments = self.segmenter.segment(
                transcript.all_words,
                max_words_per_chunk=self.config.max_segment_words * 3,
            )

            # Step 5: Multimodal analysis
            logger.info("[5/9] Multimodal analysis (audio + video)...")
            multimodal_profile = self._analyze_multimodal(
                audio_path, video_path, semantic_segments
            )

            # Step 6: 26-dimension scoring
            logger.info("[6/9] Scoring segments across 26 dimensions...")
            scored_segments = self.scorer.score_semantic_segments(
                semantic_segments, content_type=content_type
            )
            scored_segments = self._apply_multimodal_scores(scored_segments, multimodal_profile)
            self._scored_segments_cache = scored_segments

            # Step 7: Generate candidates
            logger.info("[7/9] Generating 300+ clip candidates...")
            candidates = self.candidate_generator.generate(
                semantic_segments,
                transcript.all_words,
                min_candidates=self.config.min_candidates,
            )

            # Step 8: Score and predict retention for candidates
            logger.info("[8/9] Scoring candidates and predicting retention...")
            scored_candidates = self._score_candidates(candidates)

            # Step 9: Iterative optimization
            logger.info("[9/9] Iteratively optimizing top candidates...")
            optimized = self.clip_optimizer.optimize(
                scored_candidates,
                transcript.all_words,
                top_k=max(100, self.config.num_clips * 15),
            )

            # Step 10: Select standalone clips with diversity
            standalone: List[OptimizedClip] = []
            if not self.config.story_only:
                logger.info(f"Selecting {self.config.num_clips} diverse standalone clips...")
                standalone = self._select_standalone_clips(optimized)

            # Step 11: Build story series
            story_series: List[StorySeries] = []
            if not self.config.skip_story_series:
                logger.info(f"Building {self.config.num_story_series} story series...")
                story_series = self.story_builder.build_series(
                    semantic_segments, transcript.all_words,
                    num_series=self.config.num_story_series,
                )

            # Step 12: Moderate and cut clips
            logger.info("Cutting clips with captions...")
            output_clips, clip_meta = self._cut_clips(
                video_path, standalone, story_series, transcript.all_words
            )

            # Step 13: Report + gallery
            logger.info("Building report and gallery...")
            report = self._build_report(
                video_path, scored_segments, optimized, standalone, story_series, clip_meta
            )
            gallery_path = generate_gallery(
                self.config.output_dir, output_clips, report, video_title=video_path.stem
            )

            # Cleanup audio
            try:
                audio_path.unlink()
            except Exception:
                pass

        finally:
            if download_dir and not self.config.keep_download:
                shutil.rmtree(download_dir, ignore_errors=True)

        logger.info(f"{'='*60}")
        logger.info(f"Done! {len(output_clips)} clips saved to {self.clips_dir}")
        for clip in output_clips:
            logger.info(f"  → {clip.name}")
        logger.info(f"{'='*60}")

        return ClipResult(
            clips=output_clips,
            gallery_path=gallery_path,
            report=report,
            story_series=story_series,
        )

    # ── Multimodal analysis ──────────────────────────────────

    def _analyze_multimodal(
        self,
        audio_path: Path,
        video_path: Path,
        segments: List[SemanticSegment],
    ) -> List[MultimodalSegment]:
        """Compute audio+video features for each semantic segment."""
        if self.config.skip_multimodal:
            return []

        segment_times = [(s.start_time, s.end_time) for s in segments]
        
        try:
            profile = compute_multimodal_profile(
                audio_path,
                video_path,
                segment_times,
                skip_video=not self.config.enable_video_analysis,
            )
            logger.info(f"Multimodal profile computed for {len(profile)} segments")
            return profile
        except Exception as e:
            logger.warning(f"Multimodal analysis failed: {e}")
            return []

    def _apply_multimodal_scores(
        self,
        scored: List[ScoredSegment],
        profile: List[MultimodalSegment],
    ) -> List[ScoredSegment]:
        """Boost/dampen scores based on audio/video signals."""
        if not profile or len(profile) != len(scored):
            return scored

        for seg, multi in zip(scored, profile):
            # Audio excitement boost
            if multi.audio.voice_excitement > 0.6:
                seg.emotional_intensity = min(100, seg.emotional_intensity + 5)
                seg.audio_quality = min(100, seg.audio_quality + 3)
                seg.audio_excitement = multi.audio.voice_excitement
            
            # Low audio quality penalty
            if multi.audio.speech_ratio < 0.3:
                seg.audio_quality = max(0, seg.audio_quality - 15)
            
            # Video motion boost
            if multi.video.mean_motion > 5:
                seg.visual_quality = min(100, seg.visual_quality + 5)
                seg.video_motion = multi.video.mean_motion
            
            # Dark / low visual quality penalty
            if multi.video.is_dark:
                seg.visual_quality = max(0, seg.visual_quality - 10)
            
            # Scene changes can indicate dynamic content
            seg.scene_changes = multi.video.scene_change_count
            
            # Recalculate overall score
            seg.score = self.scorer._compute_overall({
                "viewer_retention": seg.viewer_retention,
                "story_quality": seg.story_quality,
                "hook_strength": seg.hook_strength,
                "emotional_intensity": seg.emotional_intensity,
                "educational_value": seg.educational_value,
                "visual_quality": seg.visual_quality,
                "audio_quality": seg.audio_quality,
                "virality": seg.virality,
            })

        scored.sort(key=lambda s: s.score, reverse=True)
        return scored

    # ── Candidate scoring ────────────────────────────────────

    def _score_candidates(
        self,
        candidates: List[ClipCandidate],
    ) -> List[ClipCandidate]:
        """
        Score candidates using a FAST heuristic based on overlapping semantic segments.
        We avoid calling the LLM for every candidate; only the top candidates will be
        rescored with the full LLM rubric during optimization.
        """
        if not candidates:
            return []

        for cand in candidates:
            # Find scored segments overlapping this candidate
            overlapping = []
            for s in self._scored_segments_cache:
                overlap_start = max(cand.start_time, s.start)
                overlap_end = min(cand.end_time, s.end)
                if overlap_end > overlap_start:
                    overlap_dur = overlap_end - overlap_start
                    seg_dur = s.end - s.start
                    if seg_dur > 0:
                        weight = overlap_dur / seg_dur
                        overlapping.append((s, weight))

            if not overlapping:
                cand.estimated_score = 50.0
                continue

            # Weighted average of segment scores, with bonuses for hook/payoff coverage
            total_weight = sum(w for _, w in overlapping)
            weighted_score = sum(s.score * w for s, w in overlapping) / total_weight

            # Bonus if candidate contains high-scoring hook/payoff segments
            bonus = 0.0
            for s, w in overlapping:
                if s.segment_type in ("hook", "question", "setup") and w > 0.3:
                    bonus += 3.0
                if s.segment_type in ("payoff", "conclusion", "revelation", "advice") and w > 0.3:
                    bonus += 3.0
                if s.is_complete:
                    bonus += 1.0

            # Penalty if duration far from target
            duration_penalty = abs(cand.duration - self.config.target_clip_duration) * 0.5

            cand.estimated_score = min(100.0, max(0.0, weighted_score + bonus - duration_penalty))

        # Sort by estimated score
        candidates.sort(key=lambda c: c.estimated_score, reverse=True)
        return candidates

    # ── Standalone clip selection ────────────────────────────

    def _select_standalone_clips(
        self,
        optimized: List[OptimizedClip],
    ) -> List[OptimizedClip]:
        """
        Select exactly 8 diverse standalone clips.
        Maximizes topic/emotion diversity while keeping top scores.
        """
        if not optimized:
            return []

        selected: List[OptimizedClip] = []
        used_topics: set = set()
        used_emotions: set = set()

        # First pass: select top clips with diverse topics
        for opt in optimized:
            if len(selected) >= self.config.num_clips:
                break
            
            # Check duration is within target ± tolerance
            duration = opt.end_time - opt.start_time
            if duration < self.config.target_clip_duration - self.config.clip_duration_tolerance:
                continue
            if duration > self.config.target_clip_duration + self.config.clip_duration_tolerance:
                continue
            
            # Check overlap with already selected
            if self._overlaps_existing(opt, selected, threshold=0.3):
                continue
            
            # Moderation check
            if self.config.moderate_content:
                result = moderate_segment(opt.text, self.scorer.call_llm, self.config.moderation_threshold)
                if result.flagged:
                    logger.info(f"  Moderation dropped standalone [{opt.start_time:.1f}-{opt.end_time:.1f}]")
                    continue
            
            # Diversity preference
            topic = self._extract_topic(opt.text)
            emotion = self._extract_emotion(opt.text)
            
            if topic not in used_topics or emotion not in used_emotions:
                selected.append(opt)
                used_topics.add(topic)
                used_emotions.add(emotion)

        # Second pass: fill remaining slots with highest scores regardless of diversity
        for opt in optimized:
            if len(selected) >= self.config.num_clips:
                break
            if opt in selected:
                continue
            duration = opt.end_time - opt.start_time
            if not (self.config.target_clip_duration - self.config.clip_duration_tolerance <= duration <= self.config.target_clip_duration + self.config.clip_duration_tolerance):
                continue
            if self._overlaps_existing(opt, selected, threshold=0.3):
                continue
            if self.config.moderate_content:
                result = moderate_segment(opt.text, self.scorer.call_llm, self.config.moderation_threshold)
                if result.flagged:
                    continue
            selected.append(opt)

        # Sort by start time for logical ordering
        selected.sort(key=lambda o: o.start_time)
        return selected

    def _overlaps_existing(
        self,
        opt: OptimizedClip,
        selected: List[OptimizedClip],
        threshold: float = 0.3,
    ) -> bool:
        """Check if a clip overlaps too much with already selected clips."""
        for existing in selected:
            overlap_start = max(opt.start_time, existing.start_time)
            overlap_end = min(opt.end_time, existing.end_time)
            if overlap_end > overlap_start:
                overlap_dur = overlap_end - overlap_start
                opt_dur = opt.end_time - opt.start_time
                if opt_dur > 0 and overlap_dur / opt_dur > threshold:
                    return True
        return False

    def _extract_topic(self, text: str) -> str:
        """Simple topic extraction from first few nouns."""
        words = text.lower().split()[:20]
        # Very simple: use first meaningful word
        stopwords = {"the", "a", "an", "and", "or", "but", "in", "on", "at", "to", "for", "of", "with", "by", "is", "are", "was", "were", "be", "been", "being", "have", "has", "had", "do", "does", "did", "will", "would", "could", "should", "may", "might", "must", "can", "this", "that", "these", "those", "i", "you", "he", "she", "it", "we", "they", "me", "him", "her", "us", "them", "my", "your", "his", "its", "our", "their"}
        for w in words:
            clean = re.sub(r'[^a-z]', '', w)
            if clean and clean not in stopwords and len(clean) > 2:
                return clean
        return "general"

    def _extract_emotion(self, text: str) -> str:
        """Simple emotion detection from text."""
        text_lower = text.lower()
        emotion_keywords = {
            "funny": ["laugh", "joke", "funny", "hilarious", "haha", "lol"],
            "angry": ["angry", "mad", "furious", "outraged", "hate", "pissed"],
            "sad": ["sad", "cry", "crying", "tears", "depressed", "heartbroken"],
            "excited": ["excited", "amazing", "incredible", "wow", "awesome", "unbelievable"],
            "serious": ["serious", "important", "problem", "issue", "concern"],
            "inspiring": ["inspire", "motivation", "believe", "dream", "achieve", "never give up"],
            "controversial": ["controversial", "debate", "argue", "disagree", "wrong"],
        }
        for emotion, keywords in emotion_keywords.items():
            if any(kw in text_lower for kw in keywords):
                return emotion
        return "neutral"

    # ── Clip cutting ─────────────────────────────────────────

    def _cut_clips(
        self,
        video_path: Path,
        standalone: List[OptimizedClip],
        story_series: List[StorySeries],
        all_words: list,
    ) -> tuple[List[Path], List[dict]]:
        """Cut all standalone clips and story series episodes."""
        output_clips: List[Path] = []
        clip_meta: List[dict] = []
        clip_index = 0

        # Cut standalone clips
        for i, opt in enumerate(standalone, 1):
            output_path = self._make_output_path(
                video_path, opt, f"standalone_{i:02d}", clip_index
            )
            meta = self._optimized_to_meta(opt, "standalone", i, clip_index)
            clip = cut_and_caption_clip(
                video_path, output_path, opt.start_time, opt.end_time, all_words,
                self.config, hook_title=meta["title"],
            )
            output_clips.append(clip)

            clip_meta.append(meta)
            self._write_hashtags_file(clip, meta["title"], meta["hashtags"])
            clip_index += 1

        # Cut story series episodes
        for series_idx, series in enumerate(story_series, 1):
            for ep_idx, episode in enumerate([series.episode_1, series.episode_2, series.episode_3], 1):
                output_path = self._make_output_path(
                    video_path,
                    episode,
                    f"series{series_idx}_ep{ep_idx}",
                    clip_index,
                    title=episode.title,
                )
                clip = cut_and_caption_clip(
                    video_path, output_path, episode.start_time, episode.end_time,
                    all_words, self.config, hook_title=episode.title,
                )
                output_clips.append(clip)
                
                meta = {
                    "_clip_index": clip_index,
                    "type": "story_series",
                    "series_index": series_idx,
                    "episode_index": ep_idx,
                    "series_title": series.title,
                    "title": episode.title,
                    "start": episode.start_time,
                    "end": episode.end_time,
                    "duration": episode.duration,
                    "score": episode.score,
                    "text": episode.text[:300],
                    "cliffhanger": episode.cliffhanger,
                    "ending": episode.ending,
                }
                clip_meta.append(meta)
                clip_index += 1

        return output_clips, clip_meta

    def _make_output_path(
        self,
        video_path: Path,
        opt,
        suffix: str,
        clip_index: int,
        title: str = "",
    ) -> Path:
        """Generate output filename for a clip."""
        safe_name = video_path.stem.replace(" ", "_")
        score = getattr(opt, "score", 50.0)
        output_name = f"{safe_name}_{suffix}_score{score:.0f}.mp4"
        return self.clips_dir / _sanitize_filename(output_name)

    def _optimized_to_meta(
        self,
        opt: OptimizedClip,
        clip_type: str,
        rank: int,
        clip_index: int,
    ) -> dict:
        """Convert OptimizedClip to report metadata."""
        # Score the final optimized clip to get all dimensions
        scored = self._score_candidate(opt.start_time, opt.end_time, opt.text)
        if scored is None:
            scored = ScoredSegment(start=opt.start_time, end=opt.end_time, text=opt.text[:500])
            scored.score = opt.score

        hashtags = self.scorer.generate_hashtags(
            text=opt.text,
            suggested_title=scored.suggested_title,
            reasoning=scored.reasoning,
        ).get("hashtags", [])
        hashtags = hashtags[:self.config.max_hashtags]

        return {
            "_clip_index": clip_index,
            "type": clip_type,
            "rank": rank,
            "start": opt.start_time,
            "end": opt.end_time,
            "duration": opt.end_time - opt.start_time,
            "score": opt.score,
            "title": scored.suggested_title or scored.hook_line or opt.text[:60],
            "hook_line": scored.hook_line,
            "reasoning": scored.reasoning,
            "text": opt.text[:500],
            "hashtags": hashtags,
            "retention": {
                "completion_rate": opt.retention.completion_rate,
                "share_likelihood": opt.retention.share_likelihood,
                "comment_likelihood": opt.retention.comment_likelihood,
                "engagement_rate": opt.retention.engagement_rate,
            },
            "sub_scores": {
                "viewer_retention": scored.viewer_retention,
                "hook_strength": scored.hook_strength,
                "emotional_intensity": scored.emotional_intensity,
                "story_quality": scored.story_quality,
                "educational_value": scored.educational_value,
                "visual_quality": scored.visual_quality,
                "audio_quality": scored.audio_quality,
                "virality": scored.virality,
            },
        }

    @staticmethod
    def _write_hashtags_file(clip_path: Path, title: str, hashtags: list) -> None:
        """Save title + hashtags as a text file next to the clip."""
        tags = " ".join(f"#{t.lstrip('#')}" for t in hashtags)
        content = f"{title}\n\n{tags}\n"
        clip_path.with_suffix(".txt").write_text(content, encoding="utf-8")

    # ── Report ───────────────────────────────────────────────

    def _build_report(
        self,
        video_path: Path,
        scored_segments: List[ScoredSegment],
        optimized: List[OptimizedClip],
        standalone: List[OptimizedClip],
        story_series: List[StorySeries],
        clip_meta: List[dict],
    ) -> dict:
        """Build the final JSON report."""
        report = {
            "source_video": str(video_path),
            "num_segments_scored": len(scored_segments),
            "num_candidates_generated": len(optimized) * 3 if optimized else 0,
            "num_optimized_candidates": len(optimized),
            "clips_produced": len(clip_meta),
            "candidates": clip_meta,
            "top_segments": [
                {
                    "start": s.start,
                    "end": s.end,
                    "score": s.score,
                    "content_type": s.content_type,
                    "topic": s.topic,
                    "segment_type": s.segment_type,
                    "emotion": s.emotion,
                    "reasoning": s.reasoning,
                    "text_preview": s.text[:200],
                }
                for s in scored_segments[:20]
            ],
            "story_series": [
                {
                    "title": s.title,
                    "overall_story_score": s.overall_story_score,
                    "episode_1": {
                        "title": s.episode_1.title,
                        "start": s.episode_1.start_time,
                        "end": s.episode_1.end_time,
                        "duration": s.episode_1.duration,
                        "cliffhanger": s.episode_1.cliffhanger,
                    },
                    "episode_2": {
                        "title": s.episode_2.title,
                        "start": s.episode_2.start_time,
                        "end": s.episode_2.end_time,
                        "duration": s.episode_2.duration,
                        "cliffhanger": s.episode_2.cliffhanger,
                    },
                    "episode_3": {
                        "title": s.episode_3.title,
                        "start": s.episode_3.start_time,
                        "end": s.episode_3.end_time,
                        "duration": s.episode_3.duration,
                        "ending": s.episode_3.ending,
                    },
                }
                for s in story_series
            ],
        }

        report_path = self.config.output_dir / f"{video_path.stem}_report.json"
        report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
        logger.info(f"Report saved: {report_path}")
        return report


def _sanitize_filename(name: str) -> str:
    """Strip characters that are invalid in Windows filenames."""
    invalid = '<>:"/\\|?*'
    for ch in invalid:
        name = name.replace(ch, "_")
    return name
