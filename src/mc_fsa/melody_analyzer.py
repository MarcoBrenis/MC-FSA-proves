"""Herramientas de alto nivel para análisis melódico con CREPE."""

from __future__ import annotations

import numpy as np

from .audio_io import load_audio
from .models import MelodyAnalysisResult
from .pitch_crepe import extract_f0_crepe_from_audio


class MelodyAnalyzer:
    """Analizador que usa CREPE para extraer el contorno f0."""

    def __init__(
        self,
        *,
        step_size_ms: float = 10.0,
        model_capacity: str = "full",
        conf_threshold: float = 0.4,
        sample_rate: int = 16000,
    ) -> None:
        self.step_size_ms = step_size_ms
        self.model_capacity = model_capacity
        self.conf_threshold = conf_threshold
        self.sample_rate = sample_rate

    def analyze_audio(
        self, audio: np.ndarray, sample_rate: int, *, audio_path: str = ""
    ) -> MelodyAnalysisResult:
        """Ejecuta la extracción de f0 sobre un array de audio."""

        times, f0_hz, confidence = extract_f0_crepe_from_audio(
            audio,
            sample_rate,
            step_size_ms=self.step_size_ms,
            model_capacity=self.model_capacity,
            conf_threshold=self.conf_threshold,
        )

        return MelodyAnalysisResult(
            audio_path=audio_path,
            sr=sample_rate,
            audio=audio,
            f0_hz=f0_hz,
            f0_times=times,
            f0_confidence=confidence,
        )

    def analyze_file(self, path: str, *, mono: bool = True, normalize: bool = True) -> MelodyAnalysisResult:
        """Carga un archivo de audio y devuelve el contorno melódico estimado."""

        audio, sr = load_audio(path, sr=self.sample_rate, mono=mono, normalize=normalize)
        return self.analyze_audio(audio, sr, audio_path=path)


__all__ = ["MelodyAnalyzer"]
