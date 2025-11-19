"""Visualize melodic contours and segmentation labels."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Sequence

import matplotlib
import numpy as np

if not os.environ.get("DISPLAY"):
    matplotlib.use("Agg")

import matplotlib.pyplot as plt

from melody_analysis import MelodySegment, MelodySegmenter, summarize_segments


_SEGMENT_COLORS = [
    "#8dd3c7",
    "#ffffb3",
    "#bebada",
    "#fb8072",
    "#80b1d3",
    "#fdb462",
    "#b3de69",
    "#fccde5",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Visualize melodic segmentation")
    parser.add_argument("audio", type=Path, help="Ruta al archivo de audio a analizar")
    parser.add_argument("--segments", type=int, default=None, help="Número aproximado de segmentos")
    parser.add_argument(
        "--min-duration",
        type=float,
        default=1.5,
        help="Duración mínima permitida para cada segmento (segundos)",
    )
    parser.add_argument(
        "--pitch-backend",
        choices=["pyin", "crepe"],
        default="pyin",
        help="Extractor de tono a utilizar (CREPE requiere instalar su paquete)",
    )
    parser.add_argument(
        "--crepe-step-ms",
        type=float,
        default=20.0,
        help="Resolución temporal (ms) al usar CREPE",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Ruta donde guardar la figura generada (PNG, PDF, etc.)",
    )
    parser.add_argument(
        "--json",
        type=Path,
        default=None,
        help="Guardar el resumen de segmentación/clasificación en formato JSON",
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=120,
        help="Resolución de guardado en puntos por pulgada",
    )
    parser.add_argument(
        "--no-show",
        action="store_true",
        help="No abrir la ventana interactiva (útil en entornos sin display)",
    )
    parser.add_argument(
        "--no-print",
        action="store_true",
        help="No mostrar el resumen en consola (útil cuando solo se exporta JSON)",
    )
    return parser.parse_args()


def _plot_segments(ax: plt.Axes, segments: Sequence[MelodySegment], ymax: float) -> None:
    for idx, segment in enumerate(segments):
        color = _SEGMENT_COLORS[idx % len(_SEGMENT_COLORS)]
        ax.axvspan(segment.start_time, segment.end_time, color=color, alpha=0.15)
        midpoint = 0.5 * (segment.start_time + segment.end_time)
        ax.text(
            midpoint,
            ymax,
            segment.label,
            ha="center",
            va="bottom",
            fontsize=9,
            color=color,
        )


def main() -> None:
    args = parse_args()
    segmenter = MelodySegmenter(
        pitch_backend=args.pitch_backend,
        crepe_step_size=args.crepe_step_ms,
    )
    segments, times, melody = segmenter.analyze(
        str(args.audio), target_segments=args.segments, min_duration=args.min_duration
    )
    summary = summarize_segments(segments)

    times = np.asarray(times, dtype=float)
    melody = np.asarray(melody, dtype=float)
    mask = np.isfinite(melody)

    fig, ax = plt.subplots(figsize=(12, 4))
    if mask.any():
        ax.plot(times[mask], melody[mask], color="#1f77b4", linewidth=1.5, label="Melodía (MIDI)")
    ax.set_xlabel("Tiempo (s)")
    ax.set_ylabel("Altura (MIDI)")
    ax.set_title("Contorno melódico y segmentos detectados")
    ymax = ax.get_ylim()[1]
    _plot_segments(ax, segments, ymax)
    ax.legend(loc="upper right")
    ax.set_xlim(times[0] if times.size else 0.0, times[-1] if times.size else 0.0)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()

    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(args.output, dpi=args.dpi, bbox_inches="tight")

    if not args.no_show:
        plt.show()
    else:
        plt.close(fig)

    if args.json is not None:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps(summary, indent=2, ensure_ascii=False))

    if not args.no_print:
        print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
