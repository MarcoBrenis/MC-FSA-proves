"""Clon experimental del paquete de análisis y segmentación melódica."""

from .features import MelodyFeatures, extract_melody_features
from .segmenter import MelodySegment, MelodySegmenter
from .classifier import MelodySegmentAnnotation, MelodyClassifier
from .pipeline import MelodyAnalyzer, analyze_melody
from .visualization import (
    plot_f0_only,
    plot_f0_no_segments,
    plot_melody_contour,
    plot_melody_only,
    plot_spectrogram_with_segments,
)

__all__ = [
    "MelodyFeatures",
    "MelodySegment",
    "MelodySegmentAnnotation",
    "MelodySegmenter",
    "MelodyClassifier",
    "MelodyAnalyzer",
    "extract_melody_features",
    "analyze_melody",
    "plot_f0_no_segments",
    "plot_f0_only",
    "plot_melody_only",
    "plot_melody_contour",
    "plot_spectrogram_with_segments",
]
