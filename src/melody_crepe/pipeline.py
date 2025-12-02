"""Pipeline de alto nivel para extraer contornos melódicos con CREPE."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np

try:  # pragma: no cover - dependencia opcional
    import librosa
except Exception:  # pragma: no cover
    librosa = None  # type: ignore

from melody_analysis.classifier import MelodyClassifier, MelodySegmentAnnotation
from melody_analysis.features import MelodyFeatures
from melody_analysis.pipeline import MelodyAnalysisResult
from melody_analysis.segmenter import MelodySegmenter

from .features import CrepeConfig, extract_crepe_melody_features


@dataclass
class CrepeAnalysisResult(MelodyAnalysisResult):
    """El resultado mantiene compatibilidad con el pipeline original."""

    features: MelodyFeatures
    segments: list[MelodySegmentAnnotation]


class CrepeAnalyzer:
    """Analizador que emplea CREPE para estimar el contorno melódico."""

    def __init__(
        self,
        *,
        segmenter: Optional[MelodySegmenter] = None,
        classifier: Optional[MelodyClassifier] = None,
        sample_rate: int = 16000,
        crepe_config: Optional[CrepeConfig] = None,
    ) -> None:
        self.segmenter = segmenter or MelodySegmenter()
        self.classifier = classifier or MelodyClassifier()
        self.sample_rate = sample_rate
        self.crepe_config = crepe_config or CrepeConfig()

    def analyze_features(self, features: MelodyFeatures) -> CrepeAnalysisResult:
        segments = self.segmenter.segment(features)
        annotations = self.classifier.classify(features, segments)
        return CrepeAnalysisResult(features=features, segments=annotations)

    def analyze_audio(self, audio: np.ndarray, sample_rate: int) -> CrepeAnalysisResult:
        features = extract_crepe_melody_features(
            audio,
            sample_rate,
            config=self.crepe_config,
        )
        return self.analyze_features(features)

    def analyze_file(self, path: str) -> CrepeAnalysisResult:
        if librosa is None:
            raise ImportError("librosa es requerida para cargar audio desde archivo")
        audio, sr = librosa.load(path, sr=self.sample_rate, mono=True)
        return self.analyze_audio(audio, sr)


def analyze_melody_with_crepe(path: str) -> CrepeAnalysisResult:
    """Función de conveniencia que analiza un archivo usando CREPE."""

    analyzer = CrepeAnalyzer()
    return analyzer.analyze_file(path)


__all__ = [
    "CrepeAnalyzer",
    "CrepeAnalysisResult",
    "analyze_melody_with_crepe",
]
