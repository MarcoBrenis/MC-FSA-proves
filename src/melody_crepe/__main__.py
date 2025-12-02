"""CLI para extraer contornos melódicos con CREPE."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

try:  # pragma: no cover - dependencia opcional
    import librosa
except Exception:  # pragma: no cover
    librosa = None  # type: ignore

from melody_analysis.visualization import plot_melody_contour, plot_spectrogram_with_segments

from .pipeline import CrepeAnalyzer


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Extrae el contorno melódico con CREPE y genera archivos de apoyo "
            "(JSON y figuras opcionales)."
        )
    )
    parser.add_argument("audio", type=Path, help="Ruta al archivo de audio a analizar")
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Ruta opcional donde guardar el resultado en formato JSON.",
    )
    parser.add_argument(
        "--sr",
        type=int,
        default=16000,
        help="Frecuencia de muestreo objetivo para la carga del audio.",
    )
    parser.add_argument(
        "--model-capacity",
        type=str,
        default="medium",
        choices=["tiny", "small", "medium", "large", "full"],
        help="Capacidad del modelo CREPE a utilizar.",
    )
    parser.add_argument(
        "--step-size-ms",
        type=int,
        default=10,
        help="Tamaño de salto entre estimaciones de pitch (ms).",
    )
    parser.add_argument(
        "--melody-plot",
        type=Path,
        default=None,
        help="Ruta donde guardar la gráfica del contorno melódico detectado.",
    )
    parser.add_argument(
        "--sections-plot",
        type=Path,
        default=None,
        help=(
            "Ruta donde guardar el espectrograma con las secciones del JSON. "
            "Si no se especifica, no se genera la figura."
        ),
    )

    args = parser.parse_args()

    if librosa is None:
        raise ImportError("librosa es requerida para cargar audio desde la línea de comandos")

    audio, sr = librosa.load(str(args.audio), sr=args.sr, mono=True)

    from .features import CrepeConfig

    analyzer = CrepeAnalyzer(
        sample_rate=args.sr,
        crepe_config=CrepeConfig(
            model_capacity=args.model_capacity,
            step_size_ms=args.step_size_ms,
        ),
    )
    result = analyzer.analyze_audio(audio, sr)
    payload = result.to_dict()

    if args.output is None:
        json.dump(payload, sys.stdout, ensure_ascii=False, indent=2)
        sys.stdout.write("\n")
    else:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open("w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)

    if args.melody_plot is not None:
        plot_melody_contour(result, output_path=args.melody_plot)

    if args.sections_plot is not None:
        plot_spectrogram_with_segments(
            audio,
            sr,
            result,
            output_path=args.sections_plot,
        )


if __name__ == "__main__":
    main()
