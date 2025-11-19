"""Feature extraction utilities for the CREPE-powered melody pipeline."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

try:  # pragma: no cover - optional dependency loaded lazily in tests
    import crepe
except Exception:  # pragma: no cover - handled gracefully
    crepe = None  # type: ignore


@dataclass
class MelodyFeatures:
    """Container for features derived from a melody contour."""

    times: np.ndarray
    """Time stamps (seconds) for each frame."""

    pitch_midi: np.ndarray
    """Estimated fundamental frequency expressed in MIDI note numbers."""

    confidence: np.ndarray
    """Confidence of the pitch estimate for each frame in the range [0, 1]."""

    energy: np.ndarray
    """Normalized energy for each frame."""

    @property
    def duration(self) -> float:
        """Return the duration of the feature sequence."""

        if self.times.size == 0:
            return 0.0
        return float(self.times[-1] - self.times[0])


def _interpolate_nans(values: np.ndarray) -> np.ndarray:
    """Interpolate NaN values using linear interpolation."""

    values = np.asarray(values, dtype=float)
    if np.isnan(values).all():
        return np.zeros_like(values)

    nans = np.isnan(values)
    if not np.any(nans):
        return values

    indices = np.arange(values.size)
    values[nans] = np.interp(indices[nans], indices[~nans], values[~nans])
    return values


def _compute_frame_rms(
    audio: np.ndarray,
    sample_rate: int,
    centers: np.ndarray,
    window_size: int,
) -> np.ndarray:
    """Compute RMS energy around each frame center."""

    if audio.size == 0:
        return np.zeros_like(centers, dtype=float)

    half = max(1, window_size // 2)
    energies = np.zeros_like(centers, dtype=float)
    for idx, center in enumerate(centers):
        start = max(0, center - half)
        end = min(audio.size, start + window_size)
        frame = audio[start:end]
        if frame.size == 0:
            energies[idx] = 0.0
        else:
            energies[idx] = float(np.sqrt(np.mean(frame**2)))
    return energies


def extract_melody_features(
    audio: np.ndarray,
    sample_rate: int,
    *,
    step_size_ms: float = 10.0,
    model_capacity: str = "medium",
    energy_window: float = 0.046,
) -> MelodyFeatures:
    """Estimate melody features from a raw audio signal using CREPE.

    Parameters
    ----------
    audio:
        Audio samples.
    sample_rate:
        Sampling rate of ``audio``.
    step_size_ms:
        Hop size employed by CREPE in milliseconds.
    model_capacity:
        CREPE model capacity ("tiny", "small", "medium", "large", "full").
    energy_window:
        Size of the RMS window (seconds) aligned with the CREPE hop.

    Returns
    -------
    MelodyFeatures
        Extracted time, pitch, confidence, and energy trajectories.
    """

    if crepe is None:
        raise ImportError(
            "crepe is required for CREPE-based feature extraction but is not available."
        )

    audio = np.asarray(audio, dtype=np.float32)
    if audio.ndim > 1:
        audio = np.mean(audio, axis=0)

    # Normalize audio to avoid numerical issues with pitch extraction.
    if np.max(np.abs(audio)) > 0:
        audio = audio / np.max(np.abs(audio))

    step_size_ms = float(step_size_ms)
    time, frequency, confidence, _ = crepe.predict(
        audio,
        sample_rate,
        viterbi=True,
        step_size=step_size_ms,
        model_capacity=model_capacity,
    )

    pitch_hz = np.asarray(frequency, dtype=float)
    pitch_hz[pitch_hz <= 0] = np.nan
    pitch_midi = 69.0 + 12.0 * np.log2(pitch_hz / 440.0)
    pitch_midi = _interpolate_nans(pitch_midi)

    confidence = np.asarray(confidence, dtype=float)
    confidence = np.clip(confidence, 0.0, 1.0)

    times = np.asarray(time, dtype=float)

    centers = np.clip((times * sample_rate).astype(int), 0, max(0, audio.size - 1))
    window_size = max(1, int(round(sample_rate * energy_window)))
    energy = _compute_frame_rms(audio, sample_rate, centers, window_size)
    energy = _interpolate_nans(energy)
    if energy.size > 0 and energy.max() > 0:
        energy = energy / energy.max()

    return MelodyFeatures(times=times, pitch_midi=pitch_midi, confidence=confidence, energy=energy)


__all__ = ["MelodyFeatures", "extract_melody_features"]
