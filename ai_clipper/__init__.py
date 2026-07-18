"""AI Clipper — Ultimate AI Video Clip Selection & Story Engine."""

__version__ = "1.0.0"

from .config import ClipperConfig
from .pipeline import AIClipper, ClipResult
from .scorer import ViralScorer, ScoredSegment
from .semantic_segmenter import SemanticSegmenter, SemanticSegment
from .candidate_generator import CandidateGenerator, ClipCandidate
from .retention_predictor import RetentionPredictor, RetentionPrediction
from .clip_optimizer import ClipOptimizer, OptimizedClip
from .story_series_builder import StorySeriesBuilder, StorySeries, Episode
from .multimodal_analyzer import (
    analyze_audio_segment,
    analyze_video_segment,
    compute_multimodal_profile,
    AudioFeatures,
    VideoFeatures,
    MultimodalSegment,
)

__all__ = [
    "ClipperConfig",
    "AIClipper",
    "ClipResult",
    "ViralScorer",
    "ScoredSegment",
    "SemanticSegmenter",
    "SemanticSegment",
    "CandidateGenerator",
    "ClipCandidate",
    "RetentionPredictor",
    "RetentionPrediction",
    "ClipOptimizer",
    "OptimizedClip",
    "StorySeriesBuilder",
    "StorySeries",
    "Episode",
    "analyze_audio_segment",
    "analyze_video_segment",
    "compute_multimodal_profile",
    "AudioFeatures",
    "VideoFeatures",
    "MultimodalSegment",
]
