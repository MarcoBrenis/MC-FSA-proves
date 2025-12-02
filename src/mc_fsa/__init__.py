"""Core package for Melody-centered Functional Structure Analysis (MC-FSA)."""

from .audio_io import load_audio
from .melody_analyzer import MelodyAnalyzer
from .models import MelodyAnalysisResult
from .pitch_crepe import extract_f0_crepe_from_audio

__all__ = [
    "MelodyAnalyzer",
    "MelodyAnalysisResult",
    "extract_f0_crepe_from_audio",
    "load_audio",
]
