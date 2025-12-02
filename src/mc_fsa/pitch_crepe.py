"""Estimación de f0 usando CREPE."""

from __future__ import annotations

from typing import Tuple

import crepe
import librosa
import numpy as np


def extract_f0_crepe_from_audio(
    y: np.ndarray,
    sr: int,
    step_size_ms: float = 10.0,
    model_capacity: str = "full",
    conf_threshold: float = 0.4,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Extrae el contorno f0 de una señal de audio usando CREPE.

    Parameters
    ----------
    y : np.ndarray
        Señal de audio (mono).
    sr : int
        Frecuencia de muestreo de `y`.
    step_size_ms : float
        Tamaño de paso en milisegundos para CREPE.
    model_capacity : str
        Capacidad del modelo CREPE: "tiny", "small", "medium", "large", "full".
    conf_threshold : float
        Umbral de confianza. Valores por debajo se consideran no-voz y se ponen como NaN.

    Returns
    -------
    times : np.ndarray
        Tiempos en segundos de cada frame.
    f0_hz : np.ndarray
        Frecuencia fundamental en Hz (NaN donde la confianza < conf_threshold).
    confidence : np.ndarray
        Vector de confianza de CREPE entre [0, 1].
    """

    audio = np.asarray(y, dtype=np.float32)
    if audio.ndim > 1:
        audio = librosa.to_mono(audio)

    if np.max(np.abs(audio)) > 0:
        audio = audio / np.max(np.abs(audio))

    target_sr = 16000
    if sr != target_sr:
        audio = librosa.resample(audio, orig_sr=sr, target_sr=target_sr)
        sr = target_sr

    time, frequency, confidence, _ = crepe.predict(
        audio,
        sr,
        model_capacity=model_capacity,
        step_size=step_size_ms,
        verbose=0,
    )

    f0_hz = frequency.astype(float)
    f0_hz[confidence < conf_threshold] = np.nan

    return time, f0_hz, confidence


__all__ = ["extract_f0_crepe_from_audio"]
