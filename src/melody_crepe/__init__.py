"""Extractores de contorno melódico basados en CREPE.

Este paquete ofrece una variante del analizador original que utiliza
`crepe` para estimar el pitch cuadro a cuadro. De esta forma se facilita el
uso con archivos comprimidos (mp3, flac, ogg, etc.) y se obtiene un
contorno melódico listo para visualizar o segmentar con las mismas
herramientas del módulo principal.
"""

from .features import extract_crepe_melody_features
from .pipeline import CrepeAnalyzer, analyze_melody_with_crepe

__all__ = [
    "CrepeAnalyzer",
    "analyze_melody_with_crepe",
    "extract_crepe_melody_features",
]
