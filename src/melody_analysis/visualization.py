"""Utility functions to visualizar resultados del análisis melódico."""

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
        # El usuario ha indicado explícitamente el backend: lo respetamos.
        try:
            matplotlib.use(requested)
        except Exception:
            # Si la selección manual falla, continuamos con la detección automática.
            pass
        else:
            return

    current = matplotlib.get_backend().lower()
    if "agg" not in current:
        # Ya tenemos un backend interactivo o inline (por ejemplo, Jupyter).
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

    # Si nada funcionó, mantenemos Agg para garantizar que se puedan guardar archivos.
    matplotlib.use("Agg")
_configure_backend()
import matplotlib.pyplot as plt
from matplotlib.figure import Figure

try:  # pragma: no cover - dependencia opcional para espectrogramas
    import librosa
    import librosa.display  # noqa: F401 - activa utilidades de visualización
except Exception:  # pragma: no cover
    librosa = None  # type: ignore

from .pipeline import MelodyAnalysisResult

LABEL_COLOR_MAP = {
    "exposicion": "tab:blue",
    "desarrollo": "tab:orange",
    "pregunta": "tab:red",
    "respuesta": "tab:green",
    "transicion": "tab:purple",
    "cadencia": "tab:brown",
    "afirmacion": "tab:gray",
}


def _label_color(label: str) -> str:
    """Asignar un color consistente a cada etiqueta funcional."""

    return LABEL_COLOR_MAP.get(label.lower(), "tab:gray")


def _ensure_output_path(output_path: Optional[Path]) -> Optional[Path]:
    """Crear la carpeta de salida si se especifica una ruta."""

    if output_path is None:
        return None
    output_path.parent.mkdir(parents=True, exist_ok=True)
    return output_path


def _draw_segment_overlays(ax: plt.Axes, segments: Iterable, ymax: float) -> None:
    """Sombrear cada segmento en el eje temporal."""

    for ann in segments:
        color = _label_color(ann.label)
        ax.axvspan(
            ann.segment.start_time,
            ann.segment.end_time,
            color=color,
            alpha=0.15,
        )
        ax.text(
            (ann.segment.start_time + ann.segment.end_time) / 2,
            ymax,
            ann.label,
            ha="center",
            va="bottom",
            fontsize=8,
            rotation=0,
            color=color,
        )


def plot_melody_only(
    result: MelodyAnalysisResult,
    *,
    output_path: Optional[Path] = None,
    dpi: int = 300,
    show_segments: bool = True,
) -> Figure:
    """Graficar únicamente el contorno melódico opcionalmente con segmentos."""

    times = result.features.times
    pitch = result.features.pitch_midi

    fig, ax = plt.subplots(figsize=(10, 3))
    ax.plot(times, pitch, label="Melodía (MIDI)", color="tab:blue")
    ax.set_xlabel("Tiempo (s)")
    ax.set_ylabel("Pitch (MIDI)")

    if show_segments:
        ymax = float(np.nanmax(pitch)) if pitch.size else 0.0
        _draw_segment_overlays(ax, result.segments, ymax)

    ax.set_title("Contorno melódico")
    ax.grid(True, alpha=0.2)
    fig.tight_layout()

    output_path = _ensure_output_path(output_path)
    if output_path is not None:
        fig.savefig(output_path, dpi=dpi, bbox_inches="tight")

    return fig


def plot_melody_contour(
    result: MelodyAnalysisResult,
    *,
    output_path: Optional[Path] = None,
    dpi: int = 300,
) -> Figure:
    """Generar un gráfico del contorno melódico con los segmentos resaltados."""

    times = result.features.times
    pitch = result.features.pitch_midi
    energy = result.features.energy

    fig, ax1 = plt.subplots(figsize=(10, 4))
    ax1.plot(times, pitch, label="Melodía (MIDI)", color="tab:blue")
    ax1.set_xlabel("Tiempo (s)")
    ax1.set_ylabel("Pitch (MIDI)", color="tab:blue")
    ax1.tick_params(axis="y", labelcolor="tab:blue")

    ymax = float(np.nanmax(pitch)) if pitch.size else 0.0
    _draw_segment_overlays(ax1, result.segments, ymax)

    ax2 = ax1.twinx()
    ax2.plot(times, energy, label="Energía", color="tab:green", alpha=0.6)
    ax2.set_ylabel("Energía normalizada", color="tab:green")
    ax2.tick_params(axis="y", labelcolor="tab:green")

    ax1.set_title("Contorno melódico y segmentos detectados")
    fig.tight_layout()

    output_path = _ensure_output_path(output_path)
    if output_path is not None:
        fig.savefig(output_path, dpi=dpi, bbox_inches="tight")

    return fig


def _infer_hop_length(times: np.ndarray, sample_rate: int) -> int:
    """Inferir el hop length a partir de la cuadrícula temporal."""

    if times.size < 2:
        return 512
    delta = np.diff(times)
    dt = float(np.median(delta))
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
    """Dibujar un espectrograma mel con los segmentos del JSON sobrepuestos."""

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
        color = _label_color(ann.label)
        ax.axvspan(
            ann.segment.start_time,
            ann.segment.end_time,
            color=color,
            alpha=0.15,
            linewidth=0,
        )
        ax.text(
            (ann.segment.start_time + ann.segment.end_time) / 2,
            ymax - 1,
            ann.label,
            ha="center",
            va="top",
            color="black",
            fontsize=8,
            bbox={"facecolor": color, "alpha": 0.35, "pad": 1},
        )

    ax.set_title("Espectrograma mel con secciones anotadas")
    fig.tight_layout()

    output_path = _ensure_output_path(output_path)
    if output_path is not None:
        fig.savefig(output_path, dpi=dpi, bbox_inches="tight")

    return fig


__all__ = [
    "plot_melody_only",
    "plot_melody_contour",
    "plot_spectrogram_with_segments",
]
