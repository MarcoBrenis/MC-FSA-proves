"""Segmentation logic for melody analysis."""

from __future__ import annotations

from dataclasses import dataclass
from typing import List

import numpy as np
from scipy.ndimage import gaussian_filter1d
from scipy.signal import find_peaks

from .features import MelodyFeatures


@dataclass
class MelodySegment:
    """Representation of a temporal segment within the melody."""

    start_time: float
    end_time: float
    start_index: int
    end_index: int

    def duration(self) -> float:
        """Duration of the segment in seconds."""

        return float(self.end_time - self.start_time)


class MelodySegmenter:
    """Detects structural boundaries within a melody contour.

    The segmentation strategy is inspired by novelty detection techniques employed
    in MSAF but adjusted to work with melodic descriptors.  The algorithm computes
    a hybrid novelty curve from the absolute derivative of the pitch and energy
    trajectories, smooths the curve, and selects salient peaks as boundaries.
    """

    def __init__(
        self,
        *,
        kernel_size: int = 2,
        peak_threshold: float = 0.2,
        min_separation: int = 6,
    ) -> None:
        """Create a segmenter.

        Parameters
        ----------
        kernel_size:
            Standard deviation of the Gaussian kernel applied to the novelty
            curve.  Larger values produce smoother novelty profiles.
        peak_threshold:
            Minimum relative height (0-1) for peaks to be considered as
            boundaries.
        min_separation:
            Minimum number of frames between boundaries.
        """

        self.kernel_size = kernel_size
        self.peak_threshold = peak_threshold
        self.min_separation = min_separation

    def compute_novelty(self, features: MelodyFeatures) -> np.ndarray:
        """Compute the novelty curve used for segmentation."""

        pitch = features.pitch_midi
        energy = features.energy

        pitch_diff = np.abs(np.diff(pitch, prepend=pitch[0]))
        energy_diff = np.abs(np.diff(energy, prepend=energy[0]))

        if np.max(pitch_diff) > 0:
            pitch_diff = pitch_diff / np.max(pitch_diff)
        if np.max(energy_diff) > 0:
            energy_diff = energy_diff / np.max(energy_diff)

        novelty = 0.7 * pitch_diff + 0.3 * energy_diff
        novelty = gaussian_filter1d(novelty, sigma=self.kernel_size)
        return novelty

    def find_boundaries(self, novelty: np.ndarray) -> np.ndarray:
        """Locate peaks in the novelty curve."""

        if novelty.size == 0:
            return np.array([], dtype=int)

        if np.max(novelty) > 0:
            height = self.peak_threshold * np.max(novelty)
        else:
            height = self.peak_threshold

        peaks, _ = find_peaks(novelty, height=height, distance=self.min_separation)
        return peaks.astype(int)

    def segment(self, features: MelodyFeatures) -> List[MelodySegment]:
        """Segment the melody based on extracted features."""

        novelty = self.compute_novelty(features)
        boundaries = self.find_boundaries(novelty)

        frame_indices = [0] + boundaries.tolist() + [len(features.times) - 1]
        segments: List[MelodySegment] = []
        for start, end in zip(frame_indices[:-1], frame_indices[1:]):
            start_idx = int(start)
            end_idx = int(end)
            if end_idx <= start_idx:
                continue
            segment = MelodySegment(
                start_time=float(features.times[start_idx]),
                end_time=float(features.times[end_idx]),
                start_index=start_idx,
                end_index=end_idx,
            )
            if segment.duration() <= 0:
                continue
            segments.append(segment)

        return segments


__all__ = ["MelodySegment", "MelodySegmenter"]
