"""Pipeline visual para análisis melódico."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Iterable, List

import numpy as np

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import librosa
import librosa.display

from melody_analysis.pipeline import MelodyAnalysisResult, MelodyAnalyzer


FUNCTION_COLORS: Dict[str, str] = {
    "Antecedent": "tab:blue",
    "Consequent": "tab:orange",
    "Initiation": "tab:green",
    "Continuation": "tab:red",
    "Cadence": "tab:purple",
}


def _color_for_label(label: str) -> str:
    """Devolver el color asociado a una etiqueta funcional."""

    return FUNCTION_COLORS.get(label, "tab:gray")


def _smooth_contour(values: np.ndarray, window: int = 9) -> np.ndarray:
    """Aplicar un suavizado simple por media móvil."""

    if values.size == 0:
        return values
    window = max(int(window), 1)
    pad = window // 2
    padded = np.pad(values, (pad, pad), mode="edge")
    kernel = np.ones(window) / float(window)
    smoothed = np.convolve(padded, kernel, mode="valid")
    return smoothed


def _normalize(values: np.ndarray) -> np.ndarray:
    """Normalizar restando la media y dividiendo por la desviación estándar."""

    if values.size == 0:
        return values
    mean = float(np.mean(values))
    std = float(np.std(values))
    if std <= 0:
        return values - mean
    return (values - mean) / std


def _compute_ssm(contour: np.ndarray) -> np.ndarray:
    """Calcular la matriz de auto-similitud a partir de un contorno."""

    if contour.size == 0:
        return np.zeros((0, 0))
    contour = contour.reshape(1, -1)
    return librosa.segment.recurrence_matrix(
        contour, mode="affinity", sym=True, metric="cosine"
    )


def _ensure_output_dir(output_dir: Path) -> Path:
    """Crear la carpeta de salida si no existe."""

    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir


def _plot_segment_overlays(ax: plt.Axes, segments: Iterable, ymax: float) -> None:
    """Añadir rectángulos semitransparentes para los segmentos."""

    for ann in segments:
        color = _color_for_label(ann.label)
        ax.axvspan(
            ann.segment.start_time,
            ann.segment.end_time,
            color=color,
            alpha=0.18,
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
            bbox={"facecolor": "white", "alpha": 0.4, "pad": 1},
        )


def _save_or_show(fig: plt.Figure, path: Path, show: bool) -> str:
    """Guardar la figura y mostrarla opcionalmente."""

    fig.savefig(path, dpi=300, bbox_inches="tight")
    if show:
        plt.show()
    else:
        plt.close(fig)
    return str(path)


def _plot_spectrogram(
    audio: np.ndarray,
    sample_rate: int,
    result: MelodyAnalysisResult,
    output_path: Path,
    show: bool,
) -> str:
    """Generar un espectrograma STFT con segmentos coloreados."""

    stft = librosa.stft(audio, hop_length=512, n_fft=2048)
    spec_db = librosa.amplitude_to_db(np.abs(stft), ref=np.max)

    fig, ax = plt.subplots(figsize=(10, 4))
    img = librosa.display.specshow(
        spec_db,
        sr=sample_rate,
        hop_length=512,
        x_axis="time",
        y_axis="linear",
        cmap="magma",
        ax=ax,
    )
    fig.colorbar(img, ax=ax, format="%.0f dB")
    ax.set_xlabel("Tiempo (s)")
    ax.set_ylabel("Frecuencia (Hz)")
    ax.set_title("Espectrograma (STFT)")

    ymax = float(np.nanmax(spec_db)) if spec_db.size else 0.0
    _plot_segment_overlays(ax, result.segments, ymax)
    return _save_or_show(fig, output_path, show)


def _plot_mel_spectrogram(
    audio: np.ndarray,
    sample_rate: int,
    result: MelodyAnalysisResult,
    output_path: Path,
    show: bool,
) -> str:
    """Generar un mel-espectrograma con segmentos."""

    mel = librosa.feature.melspectrogram(y=audio, sr=sample_rate, hop_length=512)
    mel_db = librosa.power_to_db(mel, ref=np.max)

    fig, ax = plt.subplots(figsize=(10, 4))
    img = librosa.display.specshow(
        mel_db,
        sr=sample_rate,
        hop_length=512,
        x_axis="time",
        y_axis="mel",
        cmap="magma",
        ax=ax,
    )
    fig.colorbar(img, ax=ax, format="%.0f dB", label="dB")
    ax.set_xlabel("Tiempo (s)")
    ax.set_title("Mel-espectrograma")

    ymax = mel_db.shape[0]
    _plot_segment_overlays(ax, result.segments, ymax)
    return _save_or_show(fig, output_path, show)


def _plot_f0_curve(
    times: np.ndarray,
    f0_hz: np.ndarray,
    smoothed: np.ndarray,
    output_path: Path,
    show: bool,
) -> str:
    """Graficar la curva de f0 cruda y suavizada."""

    fig, ax = plt.subplots(figsize=(10, 3))
    ax.plot(times, f0_hz, label="f0 cruda", alpha=0.6, color="tab:blue")
    ax.plot(times, smoothed, label="f0 suavizada", color="tab:orange")
    ax.set_xlabel("Tiempo (s)")
    ax.set_ylabel("Frecuencia (Hz)")
    ax.legend()
    ax.set_title("Curva de frecuencia fundamental")
    return _save_or_show(fig, output_path, show)


def _plot_normalized_contour(
    times: np.ndarray,
    original: np.ndarray,
    normalized: np.ndarray,
    output_path: Path,
    show: bool,
) -> str:
    """Visualizar el contorno original frente al suavizado y normalizado."""

    fig, ax = plt.subplots(figsize=(10, 3))
    ax.plot(times, original, label="Original", alpha=0.5, color="tab:gray")
    ax.plot(times, normalized, label="Suavizado + normalizado", color="tab:red")
    ax.set_xlabel("Tiempo (s)")
    ax.set_ylabel("Amplitud normalizada")
    ax.legend()
    ax.set_title("Contorno melódico suavizado y normalizado")
    return _save_or_show(fig, output_path, show)


def _plot_ssm(
    times: np.ndarray,
    ssm: np.ndarray,
    output_path: Path,
    show: bool,
) -> str:
    """Mostrar la matriz de auto-similitud."""

    extent = None
    if times.size >= 2:
        extent = [times[0], times[-1], times[0], times[-1]]

    fig, ax = plt.subplots(figsize=(5, 5))
    img = ax.imshow(
        ssm,
        origin="lower",
        aspect="auto",
        cmap="magma",
        extent=extent,
    )
    ax.set_xlabel("Tiempo (s)" if extent else "Tiempo (frames)")
    ax.set_ylabel("Tiempo (s)" if extent else "Tiempo (frames)")
    ax.set_title("Matriz de auto-similitud")
    fig.colorbar(img, ax=ax)
    return _save_or_show(fig, output_path, show)


def _plot_novelty_and_boundaries(
    times: np.ndarray,
    novelty: np.ndarray,
    boundaries: np.ndarray,
    output_path: Path,
    show: bool,
) -> str:
    """Graficar la curva de novedad y los límites detectados."""

    fig, ax = plt.subplots(figsize=(10, 3))
    ax.plot(times, novelty, label="Novedad", color="tab:blue")
    for b in boundaries:
        ax.axvline(times[b], color="tab:red", linestyle="--", alpha=0.7)
    ax.set_xlabel("Tiempo (s)")
    ax.set_ylabel("Amplitud")
    ax.set_title("Curva de novedad y límites")
    ax.legend()
    return _save_or_show(fig, output_path, show)


def _plot_segment_timeline(
    segments: List,
    output_path: Path,
    show: bool,
    title: str = "Cronología de segmentos",
) -> str:
    """Dibujar una barra temporal coloreada por etiqueta."""

    fig, ax = plt.subplots(figsize=(10, 1.5))
    for ann in segments:
        color = _color_for_label(ann.label)
        ax.broken_barh(
            [(ann.segment.start_time, ann.segment.duration())],
            (0, 1),
            facecolors=color,
        )
    ax.set_ylim(0, 1)
    ax.set_xlabel("Tiempo (s)")
    ax.set_yticks([])
    ax.set_title(title)
    legend_elements = [
        plt.Line2D([0], [0], color=color, lw=4, label=label)
        for label, color in FUNCTION_COLORS.items()
    ]
    ax.legend(handles=legend_elements, loc="upper center", ncol=3, bbox_to_anchor=(0.5, 1.35))
    return _save_or_show(fig, output_path, show)


def _plot_segment_features(
    segments: List,
    output_path: Path,
    show: bool,
) -> str:
    """Visualizar vectores de características a nivel de segmento."""

    if not segments:
        fig, ax = plt.subplots(figsize=(6, 2))
        ax.text(0.5, 0.5, "Sin segmentos", ha="center", va="center")
        ax.axis("off")
        return _save_or_show(fig, output_path, show)

    descriptor_keys = list(segments[0].descriptor.keys())
    data = np.array([[ann.descriptor[k] for k in descriptor_keys] for ann in segments])

    fig, ax = plt.subplots(figsize=(10, 3))
    im = ax.imshow(data.T, aspect="auto", cmap="coolwarm")
    ax.set_yticks(range(len(descriptor_keys)))
    ax.set_yticklabels(descriptor_keys)
    ax.set_xticks(range(len(segments)))
    ax.set_xticklabels([ann.label for ann in segments], rotation=45, ha="right")
    ax.set_title("Resumen de descriptores por segmento")
    fig.colorbar(im, ax=ax, label="Valor")
    return _save_or_show(fig, output_path, show)


def run_visual_melody_pipeline(
    audio_path: str,
    output_dir: str | Path,
    show_plots: bool = False,
) -> dict:
    """Ejecutar el pipeline visual completo para un archivo de audio."""

    output_dir = _ensure_output_dir(Path(output_dir))
    analyzer = MelodyAnalyzer()
    audio, sample_rate = librosa.load(audio_path, sr=analyzer.sample_rate)
    result = analyzer.analyze_audio(audio, sample_rate)

    times = result.features.times
    f0_hz = librosa.midi_to_hz(result.features.pitch_midi)
    f0_smooth = _smooth_contour(f0_hz)
    f0_norm = _normalize(f0_smooth)

    novelty = analyzer.segmenter.compute_novelty(result.features)
    boundaries = analyzer.segmenter.find_boundaries(novelty)
    ssm = _compute_ssm(f0_norm)

    outputs = {
        "spectrogram": _plot_spectrogram(
            audio, sample_rate, result, output_dir / "01_spectrogram.png", show_plots
        ),
        "mel_spectrogram": _plot_mel_spectrogram(
            audio, sample_rate, result, output_dir / "02_mel_spectrogram.png", show_plots
        ),
        "f0_curve": _plot_f0_curve(
            times, f0_hz, f0_smooth, output_dir / "03_f0_curve.png", show_plots
        ),
        "smoothed_normalized": _plot_normalized_contour(
            times,
            f0_hz,
            f0_norm,
            output_dir / "04_f0_smoothed_normalized.png",
            show_plots,
        ),
        "ssm": _plot_ssm(times, ssm, output_dir / "05_self_similarity.png", show_plots),
        "novelty": _plot_novelty_and_boundaries(
            times,
            novelty,
            boundaries,
            output_dir / "06_novelty_boundaries.png",
            show_plots,
        ),
        "segments_timeline": _plot_segment_timeline(
            result.segments,
            output_dir / "07_segments_timeline.png",
            show_plots,
            title="Segmentos detectados",
        ),
        "segment_features": _plot_segment_features(
            result.segments,
            output_dir / "08_segment_features.png",
            show_plots,
        ),
        "final_segmentation": _plot_segment_timeline(
            result.segments,
            output_dir / "09_final_functional_segmentation.png",
            show_plots,
            title="Segmentación funcional",
        ),
    }

    if result.segments:
        descriptor_keys = list(result.segments[0].descriptor.keys())
        print("Resumen de descriptores por segmento:")
        header = ["Label", "Inicio", "Fin", *descriptor_keys]
        print("\t".join(header))
        for ann in result.segments:
            values = [f"{ann.descriptor[k]:.3f}" for k in descriptor_keys]
            print(
                "\t".join(
                    [
                        ann.label,
                        f"{ann.segment.start_time:.2f}",
                        f"{ann.segment.end_time:.2f}",
                        *values,
                    ]
                )
            )

    return outputs


__all__ = ["run_visual_melody_pipeline", "FUNCTION_COLORS"]
