"""Utilidades de visualización para la versión experimental del analizador."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable, Optional

import numpy as np

import os
import sys

import matplotlib


def _configure_backend() -> None:
    """Escoger un backend de Matplotlib compatible con el entorno."""

    requested = os.environ.get("MPLBACKEND")
    if requested:
        try:
            matplotlib.use(requested)
        except Exception:
            pass
        else:
            return

    current = matplotlib.get_backend().lower()
    if "agg" not in current:
        return

    display_available = any(
        os.environ.get(var)
        for var in ("DISPLAY", "WAYLAND_DISPLAY", "MPLBACKEND")
    ) or sys.platform == "darwin" or sys.platform.startswith("win")

    if display_available:
        for candidate in ("TkAgg", "Qt5Agg", "QtAgg", "MacOSX"):
            try:
                matplotlib.use(candidate)
            except Exception:
                continue
            else:
                return

    matplotlib.use("Agg")
_configure_backend()
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.patches import Patch

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
        display_label, color, bbox = _format_segment_label(ann.label)
        ax.axvspan(
            ann.segment.start_time,
            ann.segment.end_time,
            color="tab:orange",
            alpha=0.15,
        )
        ax.text(
            (ann.segment.start_time + ann.segment.end_time) / 2,
            ymax,
            display_label,
            ha="center",
            va="bottom",
            color=color,
            fontsize=8,
            bbox=bbox,
        )


def _format_segment_label(label: str) -> tuple[str, str, dict]:
    normalized = label.strip().lower()
    highlight_map = {
        "pregunta": ("Q", "red"),
        "question": ("Q", "red"),
        "q": ("Q", "red"),
        "respuesta": ("A", "green"),
        "answer": ("A", "green"),
        "a": ("A", "green"),
    }
    display_label, color = highlight_map.get(normalized, (label, "white"))

    if normalized in highlight_map:
        bbox = {"facecolor": color, "alpha": 0.25, "pad": 1, "edgecolor": "none"}
    else:
        bbox = {"facecolor": "black", "alpha": 0.4, "pad": 1}

    return display_label, color, bbox


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

    legend_handles = [
        Patch(facecolor="red", alpha=0.25, label="rojo = question (Q)"),
        Patch(facecolor="green", alpha=0.25, label="verde = answer (A)"),
    ]
    ax1.legend(handles=legend_handles, loc="upper right")

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
        display_label, color, bbox = _format_segment_label(ann.label)
        ax.text(
            (ann.segment.start_time + ann.segment.end_time) / 2,
            ymax - 1,
            display_label,
            ha="center",
            va="top",
            color=color,
            fontsize=8,
            bbox=bbox,
        )

    legend_handles = [
        Patch(facecolor="red", alpha=0.25, label="rojo = question (Q)"),
        Patch(facecolor="green", alpha=0.25, label="verde = answer (A)"),
    ]
    ax.legend(handles=legend_handles, loc="upper right")

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
