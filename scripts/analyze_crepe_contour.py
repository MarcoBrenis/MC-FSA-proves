"""Script sencillo para extraer y visualizar el contorno melódico con CREPE."""

from __future__ import annotations

import argparse
import os
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np

from mc_fsa import MelodyAnalyzer


def plot_melody(result, *, save_path: Optional[str] = None, show: bool = True) -> None:
    """Genera una figura con el contorno f0 y la confianza de CREPE."""

    if result.f0_times is None or result.f0_hz is None:
        raise ValueError("El resultado no contiene contorno f0 para mostrar")

    title = "Contorno melódico"
    if result.audio_path:
        title += f" - {os.path.basename(result.audio_path)}"

    fig, ax1 = plt.subplots(figsize=(12, 4))

    f0_masked = np.ma.masked_invalid(result.f0_hz)
    ax1.plot(result.f0_times, f0_masked, label="f0 (Hz)", color="C0")
    ax1.set_xlabel("Tiempo (s)")
    ax1.set_ylabel("Frecuencia (Hz)", color="C0")
    ax1.tick_params(axis="y", labelcolor="C0")
    ax1.set_title(title)

    if result.f0_confidence is not None:
        ax2 = ax1.twinx()
        ax2.plot(
            result.f0_times,
            result.f0_confidence,
            label="Confianza",
            color="C1",
            alpha=0.6,
        )
        ax2.set_ylabel("Confianza", color="C1")
        ax2.tick_params(axis="y", labelcolor="C1")

    fig.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150)
        print(f"Figura guardada en {save_path}")

    if show:
        plt.show()
    else:
        plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Extrae f0 con CREPE y muestra el contorno melódico."
    )
    parser.add_argument("audio_path", help="Ruta del archivo de audio a analizar")
    parser.add_argument(
        "--model-capacity",
        default="full",
        choices=["tiny", "small", "medium", "large", "full"],
        help="Capacidad del modelo CREPE",
    )
    parser.add_argument(
        "--step-size-ms",
        type=float,
        default=10.0,
        help="Paso temporal en milisegundos para CREPE",
    )
    parser.add_argument(
        "--conf-threshold",
        type=float,
        default=0.4,
        help="Umbral de confianza por debajo del cual se marca NaN",
    )
    parser.add_argument(
        "--no-show",
        action="store_true",
        help="No abrir ventana, útil en entornos sin interfaz gráfica",
    )
    parser.add_argument(
        "--save",
        type=str,
        help="Ruta opcional para guardar la figura del contorno",
    )

    args = parser.parse_args()

    analyzer = MelodyAnalyzer(
        step_size_ms=args.step_size_ms,
        model_capacity=args.model_capacity,
        conf_threshold=args.conf_threshold,
    )

    result = analyzer.analyze_file(args.audio_path)

    print("=== Resultados CREPE ===")
    print(f"Archivo: {result.audio_path}")
    print(f"Duración (s): {len(result.audio) / result.sr:.2f}")
    if result.f0_times is not None:
        print(f"Frames estimados: {len(result.f0_times)}")

    plot_melody(result, save_path=args.save, show=not args.no_show)


if __name__ == "__main__":
    main()
