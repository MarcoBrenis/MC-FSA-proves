"""Ejemplo de ejecución del pipeline visual melódico."""

from __future__ import annotations

import argparse
from pathlib import Path

from melody_visual_pipeline import run_visual_melody_pipeline


def main() -> None:
    """Punto de entrada para la línea de comandos."""

    parser = argparse.ArgumentParser(description="Pipeline visual para análisis melódico")
    parser.add_argument("audio", type=str, help="Ruta al archivo de audio a analizar")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Carpeta donde se guardarán las figuras (por defecto junto al audio)",
    )
    parser.add_argument(
        "--show",
        action="store_true",
        help="Mostrar las gráficas además de guardarlas",
    )
    args = parser.parse_args()

    output_dir = args.output_dir
    if output_dir is None:
        output_dir = Path(args.audio).with_suffix("")
        output_dir = output_dir.parent / f"{output_dir.name}_visual_pipeline"

    outputs = run_visual_melody_pipeline(args.audio, output_dir, show_plots=args.show)

    print("Figuras generadas:")
    for name, path in outputs.items():
        print(f"- {name}: {path}")


if __name__ == "__main__":
    main()
