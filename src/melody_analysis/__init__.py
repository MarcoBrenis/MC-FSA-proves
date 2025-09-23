"""Melody analysis and segmentation package."""

from .features import MelodyFeatures, extract_melody_features
from .segmenter import MelodySegment, MelodySegmenter
from .classifier import MelodySegmentAnnotation, MelodyClassifier
from .pipeline import MelodyAnalyzer, analyze_melody

__all__ = [
    "MelodyFeatures",
    "MelodySegment",
    "MelodySegmentAnnotation",
    "MelodySegmenter",
    "MelodyClassifier",
    "MelodyAnalyzer",
    "extract_melody_features",
    "analyze_melody",
]
