"""CLI clonada para la versión experimental ``melody_analysis_v2``."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .pipeline import MelodyAnalyzer


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Segment and classify melodic structure from an audio file.",
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
        default=22050,
        help="Frecuencia de muestreo objetivo para la carga del audio.",
    )
    parser.add_argument(
        "--hop-length",
        type=int,
        default=512,
        help="Longitud del hop empleada para la extracción de características.",
    )

    args = parser.parse_args()

    analyzer = MelodyAnalyzer(sample_rate=args.sr, hop_length=args.hop_length)
    result = analyzer.analyze_file(str(args.audio))
    payload = result.to_dict()

    if args.output is None:
        json.dump(payload, sys.stdout, ensure_ascii=False, indent=2)
        sys.stdout.write("\n")
    else:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open("w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    main()
