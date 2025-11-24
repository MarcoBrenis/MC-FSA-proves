"""CLI interactivo para analizar una canción y generar salidas útiles.

Este script simplifica la ejecución del pipeline sin recordar todos los
argumentos del módulo principal. Permite elegir la variante (v1, v2 o v3),
crea la carpeta de resultados automáticamente y guarda tanto el JSON como las
imágenes generadas a partir del espectrograma y el contorno melódico.
"""

from __future__ import annotations

import argparse
import json
from importlib import import_module
from pathlib import Path
from typing import Dict, Tuple

VARIANT_MODULES: Dict[str, str] = {
    "v1": "melody_analysis",
    "v2": "melody_analysis_v2",
    "v3": "melody_analysis_v3",
}


def load_variant(variant: str, sample_rate: int, hop_length: int) -> Tuple[object, object, object]:
    """Return (analyzer, contour_plot_fn, spectrogram_plot_fn)."""

    module = import_module(VARIANT_MODULES[variant])
    analyzer_cls = getattr(module, "MelodyAnalyzer")
    analyzer = analyzer_cls(sample_rate=sample_rate, hop_length=hop_length)
    plot_contour = getattr(module, "plot_melody_contour")
    plot_spectrogram = getattr(module, "plot_spectrogram_with_segments")
    return analyzer, plot_contour, plot_spectrogram


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Analiza un archivo de audio y genera JSON + espectrograma coloreado.",
    )
    parser.add_argument("audio", type=Path, help="Ruta al archivo de audio a analizar")
    parser.add_argument(
        "--variant",
        choices=sorted(VARIANT_MODULES.keys()),
        default="v1",
        help="Pipeline a utilizar: v1 (pyin), v2 (clon experimental) o v3 (CREPE).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("salidas_cli"),
        help="Carpeta donde guardar los resultados. Se crea si no existe.",
    )
    parser.add_argument(
        "--prefix",
        type=str,
        default=None,
        help="Prefijo para los archivos generados. Por defecto se usa el nombre del audio.",
    )
    parser.add_argument(
        "--sr",
        type=int,
        default=22050,
        help="Frecuencia de muestreo objetivo para cargar el audio.",
    )
    parser.add_argument(
        "--hop-length",
        type=int,
        default=512,
        help="Hop length para la extracción del contorno melódico.",
    )
    parser.add_argument(
        "--skip-plots",
        action="store_true",
        help="Si se establece, solo se guarda el JSON y no las figuras PNG.",
    )

    args = parser.parse_args()

    if not args.audio.exists():
        parser.error(f"No se encontró el archivo de audio: {args.audio}")
    if not args.audio.is_file():
        parser.error(f"La ruta proporcionada no es un archivo válido: {args.audio}")

    prefix = args.prefix or args.audio.stem
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    analyzer, plot_contour, plot_spectrogram = load_variant(
        args.variant, sample_rate=args.sr, hop_length=args.hop_length
    )
    result = analyzer.analyze_file(str(args.audio))
    payload = result.to_dict()

    json_path = output_dir / f"{prefix}_{args.variant}.json"
    with json_path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)

    if not args.skip_plots:
        contour_path = output_dir / f"{prefix}_{args.variant}_contorno.png"
        sections_path = output_dir / f"{prefix}_{args.variant}_secciones.png"
        plot_contour(result, output_path=contour_path)
        # La función de espectrograma necesita el audio, así que lo cargamos via librosa
        try:
            import librosa
        except Exception as exc:  # pragma: no cover - dependencia opcional
            raise ImportError(
                "librosa es requerida para generar los espectrogramas desde este script"
            ) from exc

        audio, sr = librosa.load(str(args.audio), sr=args.sr)
        plot_spectrogram(audio, sr, result, output_path=sections_path)

    print(f"Resultado guardado en {json_path}")
    if not args.skip_plots:
        print("Se generaron también las figuras PNG en la misma carpeta.")


if __name__ == "__main__":
    main()
