"""Data models for MC-FSA outputs."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np


@dataclass
class MelodyAnalysisResult:
    """Contenedor de resultados de análisis melódico."""

    audio_path: str
    sr: int
    audio: np.ndarray
    f0_hz: Optional[np.ndarray] = None
    f0_times: Optional[np.ndarray] = None
    f0_confidence: Optional[np.ndarray] = None

    def to_dict(self) -> dict:
        """Serializa el resultado en un diccionario JSON-friendly."""

        return {
            "audio_path": self.audio_path,
            "sr": self.sr,
            "f0_hz": None if self.f0_hz is None else self.f0_hz.tolist(),
            "f0_times": None if self.f0_times is None else self.f0_times.tolist(),
            "f0_confidence": None
            if self.f0_confidence is None
            else self.f0_confidence.tolist(),
        }


__all__ = ["MelodyAnalysisResult"]
