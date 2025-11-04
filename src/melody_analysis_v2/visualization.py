"""Utilidades de visualización para la versión experimental del analizador."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable, Optional

import numpy as np

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.figure import Figure

try:  # pragma: no cover
    import librosa
    import librosa.display  # noqa: F401
except Exception:  # pragma: no cover
    librosa = None  # type: ignore

from .pipeline import MelodyAnalysisResult


def _ensure_output_path(output_path: Optional[Path]) -> Optional[Path]:
    if output_path is None:
        return None
    output_path.parent.mkdir(parents=True, exist_ok=True)
    return output_path


def _draw_segment_overlays(ax: plt.Axes, segments: Iterable, ymax: float) -> None:
    for ann in segments:
        ax.axvspan(
            ann.segment.start_time,
            ann.segment.end_time,
            color="tab:orange",
            alpha=0.15,
        )
        ax.text(
            (ann.segment.start_time + ann.segment.end_time) / 2,
            ymax,
            ann.label,
            ha="center",
            va="bottom",
            fontsize=8,
        )


def plot_melody_contour(
    result: MelodyAnalysisResult,
    *,
    output_path: Optional[Path] = None,
    dpi: int = 300,
) -> Figure:
    times = result.features.times
    pitch = result.features.pitch_midi
    energy = result.features.energy

    fig, ax1 = plt.subplots(figsize=(10, 4))
    ax1.plot(times, pitch, color="tab:blue")
    ax1.set_xlabel("Tiempo (s)")
    ax1.set_ylabel("Pitch (MIDI)", color="tab:blue")
    ax1.tick_params(axis="y", labelcolor="tab:blue")

    ymax = float(np.nanmax(pitch)) if pitch.size else 0.0
    _draw_segment_overlays(ax1, result.segments, ymax)

    ax2 = ax1.twinx()
    ax2.plot(times, energy, color="tab:green", alpha=0.6)
    ax2.set_ylabel("Energía normalizada", color="tab:green")
    ax2.tick_params(axis="y", labelcolor="tab:green")

    ax1.set_title("Contorno melódico y segmentos detectados (v2)")
    fig.tight_layout()

    output_path = _ensure_output_path(output_path)
    if output_path is not None:
        fig.savefig(output_path, dpi=dpi, bbox_inches="tight")

    return fig


def _infer_hop_length(times: np.ndarray, sample_rate: int) -> int:
    if times.size < 2:
        return 512
    dt = float(np.median(np.diff(times)))
    hop = int(np.round(dt * sample_rate))
    return max(hop, 1)


def plot_spectrogram_with_segments(
    audio: np.ndarray,
    sample_rate: int,
    result: MelodyAnalysisResult,
    *,
    output_path: Optional[Path] = None,
    dpi: int = 300,
    cmap: str = "magma",
) -> Figure:
    if librosa is None:
        raise ImportError("librosa es necesario para generar espectrogramas")

    hop_length = _infer_hop_length(result.features.times, sample_rate)
    S = librosa.feature.melspectrogram(
        y=audio,
        sr=sample_rate,
        hop_length=hop_length,
        n_fft=2048,
        power=2.0,
    )
    S_db = librosa.power_to_db(S, ref=np.max)

    fig, ax = plt.subplots(figsize=(10, 4))
    img = librosa.display.specshow(
        S_db,
        sr=sample_rate,
        hop_length=hop_length,
        x_axis="time",
        y_axis="mel",
        cmap=cmap,
        ax=ax,
    )
    fig.colorbar(img, ax=ax, format="%.0f dB", label="Intensidad")

    ymax = S_db.shape[0]
    for ann in result.segments:
        ax.axvspan(
            ann.segment.start_time,
            ann.segment.end_time,
            color="white",
            alpha=0.15,
            linewidth=0,
        )
        ax.text(
            (ann.segment.start_time + ann.segment.end_time) / 2,
            ymax - 1,
            ann.label,
            ha="center",
            va="top",
            color="white",
            fontsize=8,
            bbox={"facecolor": "black", "alpha": 0.4, "pad": 1},
        )

    ax.set_title("Espectrograma con secciones anotadas (v2)")
    fig.tight_layout()

    output_path = _ensure_output_path(output_path)
    if output_path is not None:
        fig.savefig(output_path, dpi=dpi, bbox_inches="tight")

    return fig


__all__ = [
    "plot_melody_contour",
    "plot_spectrogram_with_segments",
]
