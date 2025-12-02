"""Audio loading utilities for MC-FSA."""

from __future__ import annotations

from typing import Tuple

import librosa
import numpy as np


def load_audio(
    path: str,
    sr: int = 16000,
    mono: bool = True,
    normalize: bool = True,
) -> Tuple[np.ndarray, int]:
    """Load an audio file using ``librosa``.

    Parameters
    ----------
    path : str
        Ruta al archivo de audio.
    sr : int
        Frecuencia de muestreo de salida deseada.
    mono : bool
        Si ``True``, convierte el audio a mono.
    normalize : bool
        Si ``True``, normaliza la señal entre [-1, 1].

    Returns
    -------
    np.ndarray
        Señal de audio cargada.
    int
        Frecuencia de muestreo de la señal retornada.
    """

    audio, loaded_sr = librosa.load(path, sr=sr, mono=mono)

    if normalize and np.max(np.abs(audio)) > 0:
        audio = audio / np.max(np.abs(audio))

    return audio, loaded_sr


__all__ = ["load_audio"]
