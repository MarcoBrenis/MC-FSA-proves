"""High level pipeline for melody segmentation and classification."""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

import numpy as np

from .classifier import MelodyClassifier, MelodySegmentAnnotation
from .features import MelodyFeatures, extract_melody_features
from .segmenter import MelodySegmenter

try:  # pragma: no cover - optional dependency
    import librosa
except Exception:  # pragma: no cover
    librosa = None  # type: ignore


@dataclass
class MelodyAnalysisResult:
    """Structured result returned by :class:`MelodyAnalyzer`."""

    features: MelodyFeatures
    segments: List[MelodySegmentAnnotation]

    def to_dict(self) -> dict:
        """Serialize the analysis to a JSON-compatible dictionary."""

        return {
            "segments": [
                {
                    "start_time": ann.segment.start_time,
                    "end_time": ann.segment.end_time,
                    "label": ann.label,
                    "confidence": ann.confidence,
                    "descriptor": ann.descriptor,
                }
                for ann in self.segments
            ],
            "times": self.features.times.tolist(),
            "pitch_midi": self.features.pitch_midi.tolist(),
            "confidence": self.features.confidence.tolist(),
            "energy": self.features.energy.tolist(),
        }


class MelodyAnalyzer:
    """Encapsulates feature extraction, segmentation and classification."""

    def __init__(
        self,
        *,
        segmenter: Optional[MelodySegmenter] = None,
        classifier: Optional[MelodyClassifier] = None,
        hop_length: int = 512,
        sample_rate: int = 22050,
    ) -> None:
        self.segmenter = segmenter or MelodySegmenter()
        self.classifier = classifier or MelodyClassifier()
        self.hop_length = hop_length
        self.sample_rate = sample_rate

    def analyze_features(self, features: MelodyFeatures) -> MelodyAnalysisResult:
        segments = self.segmenter.segment(features)
        annotations = self.classifier.classify(features, segments)
        return MelodyAnalysisResult(features=features, segments=annotations)

    def analyze_audio(self, audio: np.ndarray, sample_rate: int) -> MelodyAnalysisResult:
        features = extract_melody_features(
            audio,
            sample_rate,
            hop_length=self.hop_length,
        )
        return self.analyze_features(features)

    def analyze_file(self, path: str) -> MelodyAnalysisResult:
        if librosa is None:
            raise ImportError("librosa is required to load audio files")
        audio, sr = librosa.load(path, sr=self.sample_rate)
        return self.analyze_audio(audio, sr)


def analyze_melody(path: str) -> MelodyAnalysisResult:
    """Convenience function for analyzing a melody from a file path."""

    analyzer = MelodyAnalyzer()
    return analyzer.analyze_file(path)


__all__ = [
    "MelodyAnalyzer",
    "MelodyAnalysisResult",
    "analyze_melody",
]
