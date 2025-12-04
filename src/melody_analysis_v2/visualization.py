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

try:  # pragma: no cover
    import librosa
    import librosa.display  # noqa: F401
except Exception:  # pragma: no cover
    librosa = None  # type: ignore

from .pipeline import MelodyAnalysisResult

LABEL_COLOR_MAP = {
    "exposicion": "tab:blue",
    "desarrollo": "tab:orange",
    "q": "tab:red",
    "a": "tab:green",
    "transicion": "tab:purple",
    "cadencia": "tab:brown",
    "afirmacion": "tab:gray",
}

LABEL_ALIAS_COLOR_MAP = {
    "Q": LABEL_COLOR_MAP["q"],
    "A": LABEL_COLOR_MAP["a"],
}


def _label_color(label: str) -> str:
    label_lower = label.lower()
    if label_lower in LABEL_COLOR_MAP:
        return LABEL_COLOR_MAP[label_lower]
    if label_lower in LABEL_ALIAS_COLOR_MAP:
        return LABEL_ALIAS_COLOR_MAP[label_lower]
    return "tab:gray"


def _midi_to_hz(pitch_midi: np.ndarray) -> np.ndarray:
    """Convertir valores MIDI a frecuencia fundamental (f0) en Hz."""

    pitch_midi = np.asarray(pitch_midi, dtype=float)
    return 440.0 * np.power(2.0, (pitch_midi - 69.0) / 12.0)


def _ensure_output_path(output_path: Optional[Path]) -> Optional[Path]:
    if output_path is None:
        return None
    output_path.parent.mkdir(parents=True, exist_ok=True)
    return output_path


def _draw_segment_overlays(ax: plt.Axes, segments: Iterable, ymax: float) -> None:
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
            color=color,
        )


def plot_melody_only(
    result: MelodyAnalysisResult,
    *,
    output_path: Optional[Path] = None,
    dpi: int = 300,
    show_segments: bool = False,
) -> Figure:
    """Graficar únicamente el pitch de la melodía, con segmentos opcionales."""

    times = result.features.times
    pitch = result.features.pitch_midi
    f0_hz = _midi_to_hz(pitch)

    fig, ax = plt.subplots(figsize=(10, 3))
    ax.plot(times, pitch, color="tab:blue")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Pitch (MIDI)")

    ax_hz = ax.twinx()
    ax_hz.plot(times, f0_hz, color="tab:red", alpha=0.5)
    ax_hz.set_ylabel("f0 (Hz)", color="tab:red")
    ax_hz.tick_params(axis="y", labelcolor="tab:red")

    if show_segments:
        ymax = float(np.nanmax(pitch)) if pitch.size else 0.0
        _draw_segment_overlays(ax, result.segments, ymax)

    ax.set_title("Melodic contour")
    ax.grid(True, alpha=0.2)
    fig.tight_layout()

    output_path = _ensure_output_path(output_path)
    if output_path is not None:
        fig.savefig(output_path, dpi=dpi, bbox_inches="tight")

    return fig


def plot_f0_only(
    result: MelodyAnalysisResult,
    *,
    output_path: Optional[Path] = None,
    dpi: int = 300,
    show_segments: bool = False,
) -> Figure:
    """Graficar solamente la curva de f0 en Hz."""

    times = result.features.times
    f0_hz = _midi_to_hz(result.features.pitch_midi)

    fig, ax = plt.subplots(figsize=(10, 3))

    ax.plot(times, f0_hz, color="tab:blue")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("f0 (Hz)")

    if show_segments:
        ymax = float(np.nanmax(f0_hz)) if f0_hz.size else 0.0
        _draw_segment_overlays(ax, result.segments, ymax)

    ax.set_title("Melodic contour (f0)")
    ax.grid(True, alpha=0.2)
    fig.tight_layout()

    output_path = _ensure_output_path(output_path)
    if output_path is not None:
        fig.savefig(output_path, dpi=dpi, bbox_inches="tight")

    return fig


def plot_f0_no_segments(
    result: MelodyAnalysisResult,
    *,
    output_path: Optional[Path] = None,
    dpi: int = 300,
) -> Figure:
    """Graficar la curva de f0 en Hz sin mostrar segmentos."""

    
    times = result.features.times
    pitch = result.features.pitch_midi
    energy = result.features.energy
    f0_hz = _midi_to_hz(pitch)

    fig, ax1 = plt.subplots(figsize=(10, 4))
    ax1.plot(times, pitch, color="tab:blue")
    ax1.set_xlabel("Time (s)")
    ax1.set_ylabel("Pitch (MIDI)", color="tab:blue")
    ax1.tick_params(axis="y", labelcolor="tab:blue")


    ax2 = ax1.twinx()
    ax2.plot(times, energy, color="tab:green", alpha=0.6)
    ax2.set_ylabel("Normalized energy", color="tab:green")
    ax2.tick_params(axis="y", labelcolor="tab:green")

    ax3 = ax1.twinx()
    ax3.plot(times, f0_hz, color="tab:red", alpha=0.5)
    ax3.set_ylabel("f0 (Hz)", color="tab:red")
    ax3.tick_params(axis="y", labelcolor="tab:red")


    ax1.set_title("Melodic contour and energy normalized")
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
    times = result.features.times
    pitch = result.features.pitch_midi
    energy = result.features.energy
    f0_hz = _midi_to_hz(pitch)

    fig, ax1 = plt.subplots(figsize=(10, 4))
    ax1.plot(times, pitch, color="tab:blue")
    ax1.set_xlabel("Time (s)")
    ax1.set_ylabel("Pitch (MIDI)", color="tab:blue")
    ax1.tick_params(axis="y", labelcolor="tab:blue")

    ymax = float(np.nanmax(pitch)) if pitch.size else 0.0
    _draw_segment_overlays(ax1, result.segments, ymax)

    ax2 = ax1.twinx()
    ax2.plot(times, energy, color="tab:green", alpha=0.6)
    ax2.set_ylabel("Normalized energy", color="tab:green")
    ax2.tick_params(axis="y", labelcolor="tab:green")

    


    ax1.set_title("Melodic contour and detected segments")
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

    f0_hz = _midi_to_hz(result.features.pitch_midi)
    f0_mel = librosa.hz_to_mel(f0_hz)
    ax.plot(result.features.times, f0_mel, color="white", linewidth=1.5, alpha=0.9, label="f0")

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

    ax.set_title("Espectrograma con secciones anotadas (v2) y f0")
    ax.legend(loc="upper right")
    fig.tight_layout()

    output_path = _ensure_output_path(output_path)
    if output_path is not None:
        fig.savefig(output_path, dpi=dpi, bbox_inches="tight")

    return fig


def plot_self_similarity(
    result: MelodyAnalysisResult,
    *,
    output_path: Optional[Path] = None,
    dpi: int = 300,
    cmap: str = "viridis",
) -> Figure:
    """Mostrar la matriz de autosimilitud utilizada durante la segmentación."""

    if result.self_similarity is None:
        raise ValueError("No se encontró matriz de autosimilitud en el resultado.")

    ssm = np.asarray(result.self_similarity)
    fig, ax = plt.subplots(figsize=(6, 5))
    img = ax.imshow(ssm, origin="lower", aspect="auto", cmap=cmap)
    ax.set_xlabel("Frame index")
    ax.set_ylabel("Frame index")
    ax.set_title("Self-similarity matrix")
    fig.colorbar(img, ax=ax, label="Similarity")
    fig.tight_layout()

    output_path = _ensure_output_path(output_path)
    if output_path is not None:
        fig.savefig(output_path, dpi=dpi, bbox_inches="tight")

    return fig


def plot_boundary_detection(
    result: MelodyAnalysisResult,
    *,
    output_path: Optional[Path] = None,
    dpi: int = 300,
) -> Figure:
    """Graficar curvas de novedad usadas para detectar fronteras."""

    if result.novelty is None and result.base_novelty is None and result.ssm_novelty is None:
        raise ValueError("No hay curvas de novedad disponibles en el resultado.")

    times = result.features.times
    def _time_axis(target: np.ndarray) -> np.ndarray:
        if times.size:
            return np.linspace(times[0], times[-1], num=len(target))
        return np.arange(len(target))

    fig, ax = plt.subplots(figsize=(10, 3))
    if result.base_novelty is not None:
        ax.plot(_time_axis(result.base_novelty), result.base_novelty, label="Δ + energía", color="tab:blue")
    if result.ssm_novelty is not None:
        ax.plot(_time_axis(result.ssm_novelty), result.ssm_novelty, label="Autosimilitud", color="tab:orange")
    if result.novelty is not None:
        ax.plot(_time_axis(result.novelty), result.novelty, label="Combinada", color="tab:red", linewidth=2)

    ax.set_xlabel("Time (s)" if times.size else "Frame")
    ax.set_ylabel("Novelty")
    ax.set_title("Boundary detection (curvas de novedad)")
    ax.grid(True, alpha=0.2)
    ax.legend()
    fig.tight_layout()

    output_path = _ensure_output_path(output_path)
    if output_path is not None:
        fig.savefig(output_path, dpi=dpi, bbox_inches="tight")

    return fig


def plot_segment_extraction(
    result: MelodyAnalysisResult,
    *,
    output_path: Optional[Path] = None,
    dpi: int = 300,
) -> Figure:
    """Visualizar únicamente las franjas de segmentos detectados."""

    if not result.segments:
        raise ValueError("No hay segmentos para mostrar.")

    fig, ax = plt.subplots(figsize=(10, 1.6))
    if result.features.times.size:
        ax.set_xlim(result.features.times[0], result.features.times[-1])
    ax.set_ylim(0, 1)
    _draw_segment_overlays(ax, result.segments, ymax=0.9)
    ax.get_yaxis().set_visible(False)
    ax.set_xlabel("Time (s)")
    ax.set_title("Segment extraction")
    fig.tight_layout()

    output_path = _ensure_output_path(output_path)
    if output_path is not None:
        fig.savefig(output_path, dpi=dpi, bbox_inches="tight")

    return fig


def plot_descriptor_summary(
    result: MelodyAnalysisResult,
    *,
    output_path: Optional[Path] = None,
    dpi: int = 300,
    metrics: Optional[Iterable[str]] = None,
) -> Figure:
    """Resumir descriptores por segmento en gráficas separadas."""

    if not result.segments:
        raise ValueError("No hay anotaciones de segmentos disponibles.")

    default_metrics = ("slope", "pitch_range", "energy_mean", "tension_mean")
    metrics = tuple(metrics) if metrics is not None else default_metrics

    names = [f"seg{i}" for i in range(len(result.segments))]
    fig, axes = plt.subplots(len(metrics), 1, figsize=(10, 2.5 * len(metrics)), sharex=True)
    if not isinstance(axes, np.ndarray):
        axes = np.asarray([axes])

    for ax, key in zip(axes, metrics):
        values = [ann.descriptor.get(key, 0.0) for ann in result.segments]
        colors = [_label_color(ann.label) for ann in result.segments]
        ax.bar(names, values, color=colors)
        ax.set_ylabel(key)
        ax.grid(True, axis="y", alpha=0.2)
    axes[-1].set_xlabel("Segment")
    fig.suptitle("Descriptor computation per segment", y=0.99)
    fig.tight_layout()

    output_path = _ensure_output_path(output_path)
    if output_path is not None:
        fig.savefig(output_path, dpi=dpi, bbox_inches="tight")

    return fig


__all__ = [
    "plot_f0_no_segments",
    "plot_f0_only",
    "plot_melody_only",
    "plot_melody_contour",
    "plot_self_similarity",
    "plot_boundary_detection",
    "plot_segment_extraction",
    "plot_descriptor_summary",
    "plot_spectrogram_with_segments",
]
