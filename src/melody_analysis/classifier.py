"""Segment classification utilities."""

from __future__ import annotations

from dataclasses import dataclass
from typing import List

import numpy as np

from .features import MelodyFeatures
from .segmenter import MelodySegment


@dataclass
class MelodySegmentAnnotation:
    """Annotated segment with descriptive statistics and label."""

    segment: MelodySegment
    label: str
    confidence: float
    descriptor: dict


def _safe_polyfit(times: np.ndarray, values: np.ndarray) -> float:
    """Compute slope with a fall-back for short segments."""

    if times.size < 2:
        return 0.0
    times = times - times[0]
    slope = np.polyfit(times, values, 1)[0]
    return float(slope)


class MelodyClassifier:
    """Classifies melody segments using simple contour heuristics."""

    def __init__(
        self,
        *,
        slope_threshold: float = 0.5,
        energy_threshold: float = 1.05,
        range_threshold: float = 1.5,
    ) -> None:
        self.slope_threshold = slope_threshold
        self.energy_threshold = energy_threshold
        self.range_threshold = range_threshold

    def _segment_descriptor(
        self, features: MelodyFeatures, segment: MelodySegment
    ) -> dict:
        idx = slice(segment.start_index, segment.end_index + 1)
        times = features.times[idx]
        pitch = features.pitch_midi[idx]
        energy = features.energy[idx]

        slope = _safe_polyfit(times, pitch)
        delta_pitch = float(pitch[-1] - pitch[0])
        pitch_range = float(np.max(pitch) - np.min(pitch))
        energy_mean = float(np.mean(energy))
        energy_delta = float(energy[-1] - energy[0])

        descriptor = {
            "slope": slope,
            "delta_pitch": delta_pitch,
            "pitch_range": pitch_range,
            "energy_mean": energy_mean,
            "energy_delta": energy_delta,
        }
        return descriptor

    def _classify_descriptor(self, descriptor: dict, index: int, total: int) -> str:
        slope = descriptor["slope"]
        delta_pitch = descriptor["delta_pitch"]
        pitch_range = descriptor["pitch_range"]
        energy_mean = descriptor["energy_mean"]
        energy_delta = descriptor["energy_delta"]

        slope_abs = abs(slope)

        if index == 0 and slope_abs < self.slope_threshold and energy_mean >= 1.0:
            return "Initiation"

        if index == total - 1 and slope_abs < self.slope_threshold and energy_mean < 1.0:
            return "Cadence"

        if slope > self.slope_threshold or delta_pitch > self.range_threshold:
            return "Antecedent"

        if slope < -self.slope_threshold or delta_pitch < -self.range_threshold:
            return "Consequent"

        if pitch_range > self.range_threshold and energy_mean > self.energy_threshold:
            return "Continuation"

        if energy_delta > 0.1:
            return "Continuation"

        return "Continuation"

    def classify(
        self, features: MelodyFeatures, segments: List[MelodySegment]
    ) -> List[MelodySegmentAnnotation]:
        annotations: List[MelodySegmentAnnotation] = []
        if not segments:
            return annotations

        global_energy_mean = float(np.mean(features.energy)) or 1.0
        for i, segment in enumerate(segments):
            descriptor = self._segment_descriptor(features, segment)
            descriptor["energy_mean"] /= global_energy_mean
            label = self._classify_descriptor(descriptor, i, len(segments))
            slope = descriptor["slope"]
            confidence = float(1.0 - min(abs(slope) / (self.slope_threshold + 1e-6), 1.0))
            annotations.append(
                MelodySegmentAnnotation(
                    segment=segment,
                    label=label,
                    confidence=confidence,
                    descriptor=descriptor,
                )
            )
        return annotations


__all__ = ["MelodySegmentAnnotation", "MelodyClassifier"]
