"""Tools for melody segmentation and rhetorical classification.

This module implements a lightweight approximation of the ideas behind the
Music Structure Analysis Framework (MSAF), but tailored towards melodic
segment identification.  The algorithm focuses on extracting a pitch contour
from an audio file, computing novelty-based boundaries, and heuristically
classifying each segment with rhetorical labels such as "Question" or
"Answer".

The implementation is intentionally transparent and decomposed into
intermediate functions so that musicians and researchers can adapt the rules
for their own analytical frameworks.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Sequence, Tuple

import librosa
import numpy as np
from scipy.signal import find_peaks, medfilt


def _interpolate_nans(values: np.ndarray) -> np.ndarray:
    """Interpolate NaN values with linear interpolation.

    Parameters
    ----------
    values:
        Input array containing NaN values.

    Returns
    -------
    numpy.ndarray
        Array with NaNs replaced by linearly interpolated values.
    """

    values = np.asarray(values, dtype=float)
    if not np.isnan(values).any():
        return values

    indices = np.arange(values.size)
    mask = np.isnan(values)
    values[mask] = np.interp(indices[mask], indices[~mask], values[~mask])
    return values


def _normalize_features(features: np.ndarray) -> np.ndarray:
    """Standardize features column-wise to zero mean and unit variance."""

    features = np.asarray(features, dtype=float)
    mean = np.mean(features, axis=0, keepdims=True)
    std = np.std(features, axis=0, keepdims=True) + 1e-8
    return (features - mean) / std


def _checkerboard_kernel(size: int, sigma: float = 8.0) -> np.ndarray:
    """Construct a gaussian-weighted checkerboard kernel for novelty detection."""

    if size % 2 == 0:
        raise ValueError("Kernel size must be odd to be centered on a frame.")

    half = size // 2
    ramp = np.arange(-half, half + 1)
    gauss = np.exp(-(ramp ** 2) / (2 * sigma ** 2))
    envelope = np.outer(gauss, gauss)
    kernel = np.sign(np.add.outer(ramp, ramp))
    kernel[half, half] = 0.0
    kernel = kernel * envelope
    kernel = kernel - kernel.mean()
    return kernel


def _novelty_curve(similarity: np.ndarray, kernel_size: int = 33) -> np.ndarray:
    """Compute a novelty curve from a self-similarity matrix."""

    if kernel_size % 2 == 0:
        raise ValueError("Kernel size must be odd.")

    kernel = _checkerboard_kernel(kernel_size)
    pad = kernel_size // 2
    padded = np.pad(similarity, pad_width=pad, mode="constant")
    novelty = np.zeros(similarity.shape[0])
    for i in range(similarity.shape[0]):
        excerpt = padded[i : i + kernel_size, i : i + kernel_size]
        novelty[i] = np.sum(excerpt * kernel)
    novelty -= novelty.min()
    if novelty.max() > 0:
        novelty /= novelty.max()
    return novelty


def _detect_boundaries(novelty: np.ndarray, lag: int = 16, threshold: float = 0.2) -> List[int]:
    """Detect boundary positions from the novelty curve."""

    smoothed = medfilt(novelty, kernel_size=7)
    peaks, _ = find_peaks(smoothed, distance=lag, prominence=threshold)
    return peaks.tolist()


@dataclass
class MelodySegment:
    """Container for melodic segment information."""

    start_time: float
    end_time: float
    label: str
    confidence: float
    descriptors: Dict[str, float] = field(default_factory=dict)

    def duration(self) -> float:
        return self.end_time - self.start_time


class MelodySegmenter:
    """Segment and classify melodies using contour and energy heuristics."""

    def __init__(
        self,
        sample_rate: int = 22050,
        hop_length: int = 512,
        fmin: float = 80.0,
        fmax: float = 1000.0,
        pitch_backend: str = "pyin",
        crepe_step_size: float = 20.0,
    ) -> None:
        self.sample_rate = sample_rate
        self.hop_length = hop_length
        self.fmin = fmin
        self.fmax = fmax
        self.pitch_backend = pitch_backend.lower()
        self.crepe_step_size = crepe_step_size

    def _extract_melody(self, audio: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Estimate the melodic contour using the configured backend."""

        if self.pitch_backend == "pyin":
            return self._extract_melody_pyin(audio)
        if self.pitch_backend == "crepe":
            return self._extract_melody_crepe(audio)
        raise ValueError(f"Unsupported pitch backend: {self.pitch_backend}")

    def _extract_melody_pyin(self, audio: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        f0, _, _ = librosa.pyin(
            audio,
            sr=self.sample_rate,
            fmin=self.fmin,
            fmax=self.fmax,
            hop_length=self.hop_length,
        )
        f0 = _interpolate_nans(f0)
        f0_midi = librosa.hz_to_midi(f0)
        times = librosa.times_like(f0, sr=self.sample_rate, hop_length=self.hop_length)
        return f0_midi, times

    def _extract_melody_crepe(self, audio: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        try:
            import crepe
        except ImportError as exc:  # pragma: no cover - optional dependency
            raise RuntimeError(
                "The 'crepe' backend requires the optional crepe package to be installed."
            ) from exc

        target_sr = 16000
        if self.sample_rate != target_sr:
            audio_resampled = librosa.resample(audio, orig_sr=self.sample_rate, target_sr=target_sr)
        else:
            audio_resampled = audio

        step_size = max(5.0, float(self.crepe_step_size))
        times, frequency, confidence, _ = crepe.predict(
            audio_resampled,
            target_sr,
            viterbi=True,
            step_size=step_size,
        )
        f0_hz = np.asarray(frequency, dtype=float)
        f0_hz = np.where(np.asarray(confidence) < 0.3, np.nan, f0_hz)
        f0_midi = librosa.hz_to_midi(_interpolate_nans(f0_hz))
        return f0_midi, np.asarray(times, dtype=float)

    def _feature_matrix(self, melody: np.ndarray, audio: np.ndarray, times: np.ndarray) -> np.ndarray:
        """Build a feature matrix summarizing melodic and energy cues."""

        hop_length = max(1, int(self.hop_length))
        rms = librosa.feature.rms(y=audio, hop_length=hop_length)[0]
        rms_times = librosa.times_like(rms, sr=self.sample_rate, hop_length=hop_length)
        target_times = np.asarray(times, dtype=float)
        if target_times.size != melody.size:
            raise ValueError("Melody and time arrays must have the same length.")
        if rms.size != melody.size:
            if rms_times.size < 2:
                rms_times = np.linspace(0.0, target_times[-1] if target_times.size else 0.0, num=rms.size)
            rms = np.interp(target_times, rms_times, rms, left=rms[0], right=rms[-1])
        melody = _interpolate_nans(melody)
        velocity = np.gradient(melody)
        acceleration = np.gradient(velocity)
        features = np.vstack([melody, velocity, acceleration, rms]).T
        return _normalize_features(features)

    def _self_similarity(self, features: np.ndarray) -> np.ndarray:
        """Compute a cosine self-similarity matrix."""

        features = np.asarray(features)
        norm = np.linalg.norm(features, axis=1, keepdims=True) + 1e-8
        normalized = features / norm
        similarity = np.dot(normalized, normalized.T)
        similarity = (similarity + 1.0) / 2.0
        return similarity

    def segment(
        self,
        path: str,
        target_segments: int | None = None,
        min_duration: float = 1.5,
    ) -> List[MelodySegment]:
        """Segment and classify the melody of an audio file.

        Parameters
        ----------
        path:
            Path to an audio file supported by librosa.
        target_segments:
            Optional hint for the approximate number of segments.
        min_duration:
            Minimum allowed duration in seconds for a segment. Shorter segments
            will be merged with their neighbors.
        """

        audio, sr = librosa.load(path, sr=self.sample_rate)
        self.sample_rate = sr
        melody, times = self._extract_melody(audio)
        if times.size > 1:
            hop_duration = float(np.median(np.diff(times)))
            self.hop_length = max(1, int(round(hop_duration * self.sample_rate)))
        features = self._feature_matrix(melody, audio, times)
        similarity = self._self_similarity(features)

        lag = 16 if target_segments is None else max(4, similarity.shape[0] // (target_segments + 1))
        novelty = _novelty_curve(similarity)
        boundary_frames = _detect_boundaries(novelty, lag=lag)
        boundary_frames = [0] + boundary_frames + [similarity.shape[0] - 1]
        boundary_frames = sorted(set(boundary_frames))

        segments = self._classify_segments(boundary_frames, times, melody, features)
        segments = self._merge_short_segments(segments, min_duration)
        return segments

    def _merge_short_segments(self, segments: Sequence[MelodySegment], min_duration: float) -> List[MelodySegment]:
        if not segments:
            return []

        merged: List[MelodySegment] = []
        for seg in segments:
            if seg.duration() >= min_duration or not merged:
                merged.append(seg)
                continue

            prev = merged[-1]
            combined_duration = prev.duration() + seg.duration()
            weight_prev = prev.duration() / combined_duration
            weight_cur = seg.duration() / combined_duration
            descriptors = {
                key: weight_prev * prev.descriptors.get(key, 0.0) + weight_cur * seg.descriptors.get(key, 0.0)
                for key in set(prev.descriptors) | set(seg.descriptors)
            }
            merged[-1] = MelodySegment(
                start_time=prev.start_time,
                end_time=seg.end_time,
                label=prev.label,
                confidence=(weight_prev * prev.confidence + weight_cur * seg.confidence),
                descriptors=descriptors,
            )
        return merged

    def _classify_segments(
        self,
        boundary_frames: Sequence[int],
        times: np.ndarray,
        melody: np.ndarray,
        features: np.ndarray,
    ) -> List[MelodySegment]:
        global_mean = float(np.nanmean(melody))
        global_std = float(np.nanstd(melody) + 1e-6)
        durations: List[float] = []
        slopes: List[float] = []
        segments: List[MelodySegment] = []

        for start, end in zip(boundary_frames[:-1], boundary_frames[1:]):
            start_time = float(times[start])
            end_time = float(times[end])
            slice_melody = melody[start:end]
            slice_features = features[start:end]
            if slice_melody.size == 0:
                continue

            slope = float(np.mean(np.gradient(slice_melody)))
            pitch_range = float(np.nanmax(slice_melody) - np.nanmin(slice_melody))
            closing_pitch = float(slice_melody[-1])
            opening_pitch = float(slice_melody[0])
            energy = float(np.mean(slice_features[:, 3]))
            descriptors = {
                "slope": slope,
                "pitch_range": pitch_range,
                "closing_pitch": closing_pitch,
                "opening_pitch": opening_pitch,
                "energy": energy,
            }
            segment = MelodySegment(
                start_time=start_time,
                end_time=end_time,
                label="Unknown",
                confidence=0.0,
                descriptors=descriptors,
            )
            segments.append(segment)
            durations.append(segment.duration())
            slopes.append(slope)

        if not segments:
            return []

        median_duration = float(np.median(durations))
        slope_std = float(np.std(slopes) + 1e-6)

        for i, segment in enumerate(segments):
            desc = segment.descriptors
            slope = desc["slope"]
            closing_pitch = desc["closing_pitch"]
            opening_pitch = desc["opening_pitch"]
            energy = desc["energy"]

            label = "Development"
            confidence = 0.4

            if i == 0 and segment.duration() >= 0.8 * median_duration:
                label = "Theme Exposition"
                confidence = 0.9
            elif slope > 0.3 * slope_std and closing_pitch > opening_pitch + 0.5:
                label = "Question"
                confidence = 0.7
            elif slope < -0.3 * slope_std and closing_pitch < global_mean + 0.25 * global_std:
                label = "Answer"
                confidence = 0.7
            elif energy > 0.5:
                label = "Climax"
                confidence = 0.6

            segment.label = label
            segment.confidence = confidence

        return segments


def summarize_segments(segments: Iterable[MelodySegment]) -> List[Dict[str, float | str]]:
    """Summarize segments into serializable dictionaries."""

    summary: List[Dict[str, float | str]] = []
    for segment in segments:
        entry: Dict[str, float | str] = {
            "start": segment.start_time,
            "end": segment.end_time,
            "label": segment.label,
            "confidence": segment.confidence,
        }
        entry.update(segment.descriptors)
        summary.append(entry)
    return summary


__all__ = ["MelodySegmenter", "MelodySegment", "summarize_segments"]
