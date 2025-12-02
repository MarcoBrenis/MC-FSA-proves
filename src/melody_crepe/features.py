"""Extracción de características melódicas utilizando CREPE."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np

try:  # pragma: no cover - dependencia opcional
    import crepe
except Exception:  # pragma: no cover
    crepe = None  # type: ignore

try:  # pragma: no cover - usada para energía y utilidades
    import librosa
except Exception:  # pragma: no cover
    librosa = None  # type: ignore

from melody_analysis.features import MelodyFeatures, _interpolate_nans


CrepeCapacity = Literal["tiny", "small", "medium", "large", "full"]


@dataclass
class CrepeConfig:
    """Parámetros de extracción específicos de CREPE."""

    model_capacity: CrepeCapacity = "medium"
    step_size_ms: int = 10
    center: bool = True


def _normalize_audio(audio: np.ndarray) -> np.ndarray:
    audio = np.asarray(audio, dtype=float)
    if audio.ndim > 1:
        audio = np.mean(audio, axis=0)
    if np.max(np.abs(audio)) > 0:
        audio = audio / np.max(np.abs(audio))
    return audio.astype(np.float32)


def _estimate_energy(
    audio: np.ndarray,
    sample_rate: int,
    step_size_ms: int,
    center: bool,
) -> np.ndarray:
    if librosa is None:
        raise ImportError("librosa es necesaria para calcular la energía de la señal")

    hop_length = max(int(round(step_size_ms * sample_rate / 1000.0)), 1)
    frame_length = hop_length * 4
    energy = librosa.feature.rms(
        y=audio,
        frame_length=frame_length,
        hop_length=hop_length,
        center=center,
    ).flatten()
    energy = _interpolate_nans(energy)
    if energy.max() > 0:
        energy = energy / energy.max()
    return energy


def extract_crepe_melody_features(
    audio: np.ndarray,
    sample_rate: int,
    *,
    config: CrepeConfig | None = None,
) -> MelodyFeatures:
    """Obtener contorno melódico usando el modelo CREPE.

    El resultado es compatible con el pipeline existente, por lo que puede
    segmentarse y visualizarse con las utilidades del paquete principal.
    """

    if crepe is None:
        raise ImportError(
            "El paquete 'crepe' es necesario para extraer el contorno melódico. "
            "Instálalo con `pip install .[crepe]` o `pip install crepe`."
        )
    if librosa is None:
        raise ImportError("librosa es necesaria para procesar el audio de entrada")

    cfg = config or CrepeConfig()
    audio = _normalize_audio(audio)

    time, frequency, confidence, _ = crepe.predict(
        audio,
        sample_rate,
        model_capacity=cfg.model_capacity,
        step_size=cfg.step_size_ms,
        center=cfg.center,
        viterbi=True,
    )

    frequency = np.asarray(frequency, dtype=float)
    frequency[frequency <= 0] = np.nan
    pitch_midi = librosa.hz_to_midi(frequency)
    pitch_midi = _interpolate_nans(pitch_midi)

    confidence = np.asarray(confidence, dtype=float)
    confidence = np.clip(confidence, 0.0, 1.0)

    energy = _estimate_energy(audio, sample_rate, cfg.step_size_ms, cfg.center)
    if energy.size != pitch_midi.size:
        energy = librosa.util.fix_length(energy, size=pitch_midi.size)

    return MelodyFeatures(
        times=np.asarray(time, dtype=float),
        pitch_midi=pitch_midi,
        confidence=confidence,
        energy=energy,
    )


__all__ = [
    "CrepeConfig",
    "extract_crepe_melody_features",
]
