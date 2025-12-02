"""Melody analysis pipeline with multi-method f0 extraction and functional segmentation.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import Callable, Iterable, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
try:
    import crepe
except ImportError:  # pragma: no cover - optional dependency during tests
    crepe = None

import librosa
import librosa.display


# ============================================================================
# Data models
# ============================================================================


@dataclass
class TimeSegment:
    """Simple representation of a time interval in seconds."""

    start_time: float
    end_time: float


@dataclass
class FunctionalSegment:
    """Functional melodic segment with label and heuristic confidence."""

    segment: TimeSegment
    label: str
    confidence: float


@dataclass
class MelodyAnalysisResult:
    """Container for all intermediate and final analysis outputs."""

    audio_path: str
    sr: int
    f0_hz: np.ndarray
    f0_times: np.ndarray
    f0_confidence: np.ndarray
    mel_spectrogram: np.ndarray
    mel_frequencies: np.ndarray
    melody_self_similarity: np.ndarray
    segments: List[FunctionalSegment]


# ============================================================================
# Melody analyzer
# ============================================================================


class MelodyAnalyzer:
    """Analyze melodic structure using multiple pitch extraction backends."""

    def __init__(
        self,
        sr: int = 44100,
        hop_length: int = 512,
        n_mels: int = 128,
        fmin: float = 55.0,
        fmax: float = 1760.0,
        crepe_model: str = "full",
        crepe_step_size_ms: float = 10.0,
        pitch_method: str = "crepe",
    ):
        self.sr = sr
        self.hop_length = hop_length
        self.n_mels = n_mels
        self.fmin = fmin
        self.fmax = fmax
        self.crepe_model = crepe_model
        self.crepe_step_size_ms = crepe_step_size_ms
        self.pitch_method = pitch_method

    # ------------------------------------------------------------------
    # Audio loading
    # ------------------------------------------------------------------
    def load_audio(self, audio_path: str) -> Tuple[np.ndarray, int]:
        """Load audio file in mono and resample to analyzer sample rate.

        Parameters
        ----------
        audio_path : str
            Path to the audio file.

        Returns
        -------
        y : np.ndarray
            Audio signal in mono.
        sr : int
            Sampling rate of the returned signal.
        """

        if not os.path.exists(audio_path):
            raise FileNotFoundError(f"Audio file not found: {audio_path}")

        y, sr = librosa.load(audio_path, sr=self.sr, mono=True)
        return y, sr

    # ------------------------------------------------------------------
    # Pitch extraction backends
    # ------------------------------------------------------------------
    def extract_f0_crepe(self, y: np.ndarray, sr: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Extract f0 using CREPE.

        Applies a confidence threshold to mark unvoiced frames as NaN.
        """

        if crepe is None:
            raise ImportError("crepe is not installed. Please install it to use this backend.")

        # CREPE expects 16-bit PCM; we pass float audio and let it handle scaling.
        time, frequency, confidence, _activation = crepe.predict(
            y,
            sr,
            model=self.crepe_model,
            viterbi=True,
            step_size=int(self.crepe_step_size_ms),
        )

        conf_threshold = 0.4
        f0_hz = frequency.astype(float)
        f0_hz[confidence < conf_threshold] = np.nan
        return f0_hz, time, confidence

    def extract_f0_pyin(self, y: np.ndarray, sr: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Extract f0 using librosa.pyin.

        Uses the voiced/unvoiced mask as a proxy for confidence.
        """

        f0, voiced_flag, voiced_prob = librosa.pyin(
            y,
            fmin=self.fmin,
            fmax=self.fmax,
            sr=sr,
            hop_length=self.hop_length,
        )
        times = librosa.times_like(f0, sr=sr, hop_length=self.hop_length)

        f0_hz = f0.astype(float)
        f0_hz[~voiced_flag] = np.nan
        confidence = np.where(voiced_flag, voiced_prob, 0.0)
        return f0_hz, times, confidence

    def extract_f0_yin(self, y: np.ndarray, sr: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Extract f0 using librosa.yin.

        Generates a heuristic confidence based on energy and pitch range validity.
        """

        f0 = librosa.yin(
            y,
            fmin=self.fmin,
            fmax=self.fmax,
            sr=sr,
            hop_length=self.hop_length,
        )
        times = librosa.times_like(f0, sr=sr, hop_length=self.hop_length)

        # Heuristic confidence: normalized energy within a sliding window.
        frame_energy = librosa.feature.rms(y=y, hop_length=self.hop_length, frame_length=self.hop_length * 2)[0]
        frame_energy = librosa.util.fix_length(frame_energy, size=f0.shape[0])
        norm_energy = frame_energy / (np.max(frame_energy) + 1e-8)

        # Penalize out-of-range values.
        valid = np.logical_and(f0 >= self.fmin, f0 <= self.fmax)
        confidence = norm_energy * valid.astype(float)
        f0_hz = f0.astype(float)
        f0_hz[~valid] = np.nan
        return f0_hz, times, confidence

    def extract_f0(self, y: np.ndarray, sr: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Dispatch pitch extraction according to the configured method."""

        method = self.pitch_method.lower()
        if method == "crepe":
            return self.extract_f0_crepe(y, sr)
        if method == "pyin":
            return self.extract_f0_pyin(y, sr)
        if method == "yin":
            return self.extract_f0_yin(y, sr)
        raise ValueError(f"Unsupported pitch extraction method: {self.pitch_method}")

    # ------------------------------------------------------------------
    # Spectrograms and self-similarity
    # ------------------------------------------------------------------
    def compute_mel_spectrogram(self, y: np.ndarray, sr: int) -> Tuple[np.ndarray, np.ndarray]:
        """Compute mel-spectrogram in dB scale and its frequency axis."""

        S = librosa.feature.melspectrogram(
            y=y,
            sr=sr,
            hop_length=self.hop_length,
            n_mels=self.n_mels,
            fmin=self.fmin,
            fmax=self.fmax,
            power=2.0,
        )
        S_db = librosa.power_to_db(S, ref=np.max)
        mel_frequencies = librosa.mel_frequencies(n_mels=self.n_mels, fmin=self.fmin, fmax=self.fmax)
        return S_db, mel_frequencies

    def compute_melody_self_similarity(self, f0_hz: np.ndarray) -> np.ndarray:
        """Compute melody self-similarity matrix from f0 contour.

        Converts f0 to MIDI, normalizes, and builds a similarity matrix using
        Euclidean distances mapped to a bounded similarity measure.
        """

        # Convert to MIDI and normalize contour shape.
        f0_midi = librosa.hz_to_midi(f0_hz)
        valid = ~np.isnan(f0_midi)
        if np.any(valid):
            mean = np.nanmean(f0_midi)
            std = np.nanstd(f0_midi) + 1e-8
            contour = (f0_midi - mean) / std
            contour[~valid] = 0.0
        else:
            contour = np.zeros_like(f0_midi)

        # Build distance matrix.
        contour = contour[:, np.newaxis]
        diff = contour - contour.T
        distance = np.sqrt(diff ** 2)
        similarity = 1.0 / (1.0 + distance)
        return similarity

    # ------------------------------------------------------------------
    # Segmentation
    # ------------------------------------------------------------------
    def _compute_novelty_from_self_similarity(self, sim: np.ndarray, kernel_size: int = 16) -> np.ndarray:
        """Compute a simple novelty curve from the self-similarity matrix.

        Uses a checkerboard kernel to emphasize changes along the diagonal.
        """

        if kernel_size % 2 == 1:
            kernel_size += 1
        half = kernel_size // 2
        kernel = np.block(
            [[np.ones((half, half)), -np.ones((half, half))],
             [-np.ones((half, half)), np.ones((half, half))]]
        )
        novelty = np.zeros(sim.shape[0])
        for i in range(half, sim.shape[0] - half):
            sub = sim[i - half : i + half, i - half : i + half]
            novelty[i] = np.sum(sub * kernel)
        novelty = np.maximum(novelty, 0.0)
        if np.max(novelty) > 0:
            novelty /= np.max(novelty)
        return novelty

    def _frame_indices_to_time(self, indices: Iterable[int], times: np.ndarray) -> List[float]:
        return [times[int(i)] if 0 <= int(i) < len(times) else float(times[-1]) for i in indices]

    def segment_melody(
        self,
        f0_hz: np.ndarray,
        times: np.ndarray,
        confidences: np.ndarray,
        self_similarity: np.ndarray,
    ) -> List[FunctionalSegment]:
        """Segment melody using pitch changes, pauses, and novelty peaks."""

        # Identify unvoiced regions as potential boundaries.
        voiced = ~np.isnan(f0_hz)
        pauses = np.where(~voiced)[0]

        # Intervàlic jumps as boundaries: large differences in MIDI.
        f0_midi = librosa.hz_to_midi(f0_hz)
        f0_midi[np.isnan(f0_midi)] = np.interp(
            np.flatnonzero(np.isnan(f0_midi)),
            np.flatnonzero(~np.isnan(f0_midi)),
            f0_midi[~np.isnan(f0_midi)],
            left=np.nanmean(f0_midi),
            right=np.nanmean(f0_midi),
        )
        df = np.abs(np.diff(f0_midi))
        jump_indices = np.where(df > 3.0)[0]  # >3 semitones between frames

        # Novelty peaks from self-similarity.
        novelty = self._compute_novelty_from_self_similarity(self_similarity)
        novelty_peaks = np.where(novelty > 0.5)[0]

        candidate_frames = np.unique(
            np.concatenate([
                pauses,
                jump_indices,
                novelty_peaks,
                np.array([0, len(times) - 1]),
            ])
        )
        candidate_frames.sort()

        # Enforce minimal duration by merging close boundaries.
        min_duration = 1.5
        boundaries = [candidate_frames[0]]
        for idx in candidate_frames[1:]:
            if times[idx] - times[boundaries[-1]] >= min_duration:
                boundaries.append(idx)
            else:
                continue
        if boundaries[-1] != len(times) - 1:
            boundaries.append(len(times) - 1)

        segments_times: List[TimeSegment] = []
        for start_idx, end_idx in zip(boundaries[:-1], boundaries[1:]):
            start_time = float(times[start_idx])
            end_time = float(times[end_idx])
            if end_time - start_time < min_duration:
                continue
            segments_times.append(TimeSegment(start_time=start_time, end_time=end_time))

        segment_features = self._compute_segment_features(segments_times, f0_hz, times, confidences, novelty)
        labeled_segments = self.assign_functional_labels(segments_times, segment_features, self_similarity)
        return labeled_segments

    def _compute_segment_features(
        self,
        segments: List[TimeSegment],
        f0_hz: np.ndarray,
        times: np.ndarray,
        confidences: np.ndarray,
        novelty: np.ndarray,
    ) -> dict:
        """Compute descriptive features for each segment."""

        f0_midi = librosa.hz_to_midi(f0_hz)
        features = {}
        for idx, seg in enumerate(segments):
            mask = (times >= seg.start_time) & (times <= seg.end_time)
            seg_f0 = f0_midi[mask]
            seg_conf = confidences[mask]
            seg_times = times[mask]
            seg_indices = np.where(mask)[0]
            duration = seg.end_time - seg.start_time

            def safe_stat(arr: np.ndarray, func: Callable[[np.ndarray], float], default: float = np.nan) -> float:
                arr = arr[~np.isnan(arr)]
                if arr.size == 0:
                    return default
                return float(func(arr))

            # Pitch descriptors.
            first_len = max(1, int(len(seg_f0) * 0.1))
            last_len = max(1, int(len(seg_f0) * 0.1))
            pitch_start = safe_stat(seg_f0[:first_len], np.median)
            pitch_end = safe_stat(seg_f0[-last_len:], np.median)
            pitch_mean = safe_stat(seg_f0, np.mean)
            pitch_std = safe_stat(seg_f0, np.std)
            pitch_range = safe_stat(seg_f0, np.nanmax) - safe_stat(seg_f0, np.nanmin)
            pitch_slope = (pitch_end - pitch_start) / max(duration, 1e-6)

            conf_mean = safe_stat(seg_conf, np.mean, default=0.0)

            # Novelty around boundaries to detect section changes.
            novelty_mask = (times >= seg.start_time) & (times <= seg.end_time)
            local_novelty = float(np.mean(novelty[novelty_mask])) if np.any(novelty_mask) else 0.0

            features[idx] = {
                "duration": duration,
                "pitch_start": pitch_start,
                "pitch_end": pitch_end,
                "pitch_slope": pitch_slope,
                "pitch_mean": pitch_mean,
                "pitch_std": pitch_std,
                "pitch_range": pitch_range,
                "conf_mean": conf_mean,
                "local_novelty": local_novelty,
                "seg_times": seg_times,
                "seg_indices": seg_indices,
                "seg_f0": seg_f0,
            }
        return features

    def assign_functional_labels(
        self,
        segments_times: List[TimeSegment],
        segment_features: dict,
        self_similarity: np.ndarray,
    ) -> List[FunctionalSegment]:
        """Assign functional labels using heuristic rules inspired by FSA literature."""

        labels = [None] * len(segments_times)
        confidences = [0.5] * len(segments_times)
        total_duration = segments_times[-1].end_time if segments_times else 0.0

        # 1) Climax: segment with highest mean pitch and large range.
        climax_idx = None
        if segments_times:
            pitch_means = np.array([segment_features[i]["pitch_mean"] for i in range(len(segments_times))])
            pitch_ranges = np.array([segment_features[i]["pitch_range"] for i in range(len(segments_times))])
            candidate = np.nanargmax(pitch_means)
            if segment_features[candidate]["duration"] > 1.0 and pitch_ranges[candidate] > 5:
                climax_idx = candidate
                labels[candidate] = "climax"
                confidences[candidate] = 0.95

        # 2) Cadence near the end with descending slope and stability.
        for i, seg in enumerate(segments_times):
            if labels[i] is not None:
                continue
            feat = segment_features[i]
            position_ratio = seg.end_time / max(total_duration, 1e-6)
            if position_ratio >= 0.75 and feat["pitch_slope"] < -0.5 and feat["pitch_std"] < 1.5:
                labels[i] = "cadence"
                confidences[i] = 0.85

        # 3) Exposition: early segments with moderate variation.
        for i, seg in enumerate(segments_times):
            if labels[i] is not None:
                continue
            feat = segment_features[i]
            position_ratio = seg.end_time / max(total_duration, 1e-6)
            if position_ratio <= 0.3 and feat["pitch_std"] < 3.0 and feat["pitch_range"] < 8.0:
                labels[i] = "exposition"
                confidences[i] = 0.7

        # 4) Development: later segments that vary previously presented material.
        for i, seg in enumerate(segments_times):
            if labels[i] is not None:
                continue
            feat = segment_features[i]
            position_ratio = seg.end_time / max(total_duration, 1e-6)
            # Similarity to earlier segments: mean similarity between frame blocks.
            earlier_indices = [j for j in range(i) if labels[j] in {"exposition", "development"}]
            similarity_score = 0.0
            for j in earlier_indices:
                idx_i = np.ix_(segment_features[i]["seg_indices"], segment_features[j]["seg_indices"])
                block_sim = np.nanmean(self_similarity[idx_i]) if idx_i[0].size and idx_i[1].size else 0.0
                similarity_score = max(similarity_score, block_sim)
            if position_ratio > 0.3 and feat["pitch_std"] >= 2.0 and similarity_score > 0.3:
                labels[i] = "development"
                confidences[i] = 0.65

        # 5) Question: rising contour ending higher and unstable.
        for i, seg in enumerate(segments_times):
            if labels[i] is not None:
                continue
            feat = segment_features[i]
            if feat["pitch_slope"] > 0.3 and (feat["pitch_end"] - feat["pitch_start"] > 2.0) and feat["pitch_std"] >= 1.5:
                labels[i] = "question"
                confidences[i] = 0.75

        # 6) Answer: stable or descending segment similar to a prior question.
        for i, seg in enumerate(segments_times):
            if labels[i] is not None:
                continue
            feat = segment_features[i]
            question_indices = [j for j, l in enumerate(labels) if l == "question"]
            for q_idx in question_indices:
                # Similarity check between frame blocks from question and candidate answer.
                idx_pair = np.ix_(segment_features[i]["seg_indices"], segment_features[q_idx]["seg_indices"])
                sim_block = np.nanmean(self_similarity[idx_pair]) if idx_pair[0].size and idx_pair[1].size else 0.0
                if sim_block > 0.4 and feat["pitch_slope"] <= 0.1:
                    labels[i] = "answer"
                    confidences[i] = 0.8
                    break

        # 7) Fallback: exposition early, development late.
        for i, seg in enumerate(segments_times):
            if labels[i] is None:
                position_ratio = seg.end_time / max(total_duration, 1e-6)
                feat = segment_features[i]
                if position_ratio < 0.5 and feat["pitch_std"] < 2.5:
                    labels[i] = "exposition"
                else:
                    labels[i] = "development"
                confidences[i] = 0.55

        functional_segments: List[FunctionalSegment] = []
        for idx, seg in enumerate(segments_times):
            functional_segments.append(
                FunctionalSegment(segment=seg, label=labels[idx], confidence=float(confidences[idx]))
            )
        return functional_segments

    # ------------------------------------------------------------------
    # End-to-end analysis
    # ------------------------------------------------------------------
    def analyze_file(self, audio_path: str) -> MelodyAnalysisResult:
        """Run the full analysis pipeline for a given audio file."""

        y, sr = self.load_audio(audio_path)
        f0_hz, f0_times, f0_conf = self.extract_f0(y, sr)
        mel_spec, mel_freqs = self.compute_mel_spectrogram(y, sr)
        self_sim = self.compute_melody_self_similarity(f0_hz)
        segments = self.segment_melody(f0_hz, f0_times, f0_conf, self_sim)

        return MelodyAnalysisResult(
            audio_path=audio_path,
            sr=sr,
            f0_hz=f0_hz,
            f0_times=f0_times,
            f0_confidence=f0_conf,
            mel_spectrogram=mel_spec,
            mel_frequencies=mel_freqs,
            melody_self_similarity=self_sim,
            segments=segments,
        )

    def to_json(self, result: MelodyAnalysisResult) -> str:
        """Serialize analysis result to JSON, omitting heavy matrices."""

        payload = {
            "audio_path": result.audio_path,
            "sr": result.sr,
            "pitch_method": self.pitch_method,
            "segments": [
                {
                    "label": seg.label,
                    "start_time": seg.segment.start_time,
                    "end_time": seg.segment.end_time,
                    "confidence": seg.confidence,
                }
                for seg in result.segments
            ],
        }
        return json.dumps(payload, indent=2, ensure_ascii=False)


# ============================================================================
# Visualization utilities
# ============================================================================


def _plot_segments(ax: plt.Axes, segments: List[FunctionalSegment], ymin: float, ymax: float):
    colors = {
        "climax": "red",
        "cadence": "navy",
        "exposition": "green",
        "development": "orange",
        "question": "purple",
        "answer": "cyan",
    }
    for seg in segments:
        color = colors.get(seg.label, "gray")
        ax.axvspan(seg.segment.start_time, seg.segment.end_time, color=color, alpha=0.2)
        ax.text(
            (seg.segment.start_time + seg.segment.end_time) / 2,
            ymax,
            seg.label,
            color=color,
            ha="center",
            va="bottom",
            fontsize=9,
            rotation=0,
        )


def plot_f0_contour(result: MelodyAnalysisResult, show_segments: bool = False):
    """Plot raw f0 contour with optional functional segment overlays."""

    fig, ax = plt.subplots(figsize=(12, 4))
    ax.plot(result.f0_times, result.f0_hz, label="f0 (Hz)")
    ax.set_xlabel("Tiempo (s)")
    ax.set_ylabel("Frecuencia (Hz)")
    ax.set_title("Contorno f0")
    if show_segments:
        ymin, ymax = ax.get_ylim()
        _plot_segments(ax, result.segments, ymin, ymax)
    ax.legend()
    plt.tight_layout()
    plt.show()


def plot_melodic_contour(result: MelodyAnalysisResult, normalize: bool = True, show_segments: bool = True):
    """Plot melodic contour in MIDI space with optional normalization and segment bands."""

    f0_midi = librosa.hz_to_midi(result.f0_hz)
    if normalize:
        mean = np.nanmean(f0_midi)
        std = np.nanstd(f0_midi) + 1e-8
        f0_midi = (f0_midi - mean) / std
    fig, ax = plt.subplots(figsize=(12, 4))
    ax.plot(result.f0_times, f0_midi, label="Contorno melódico (MIDI)")
    ax.set_xlabel("Tiempo (s)")
    ax.set_ylabel("MIDI normalizado" if normalize else "MIDI")
    ax.set_title("Contorno melódico normalizado" if normalize else "Contorno melódico")
    if show_segments:
        ymin, ymax = ax.get_ylim()
        _plot_segments(ax, result.segments, ymin, ymax)
    ax.legend()
    plt.tight_layout()
    plt.show()


def plot_mel_and_f0_with_segments(result: MelodyAnalysisResult):
    """Plot mel-spectrogram with overlaid f0 contour and functional segments."""

    fig, ax = plt.subplots(figsize=(12, 6))
    img = librosa.display.specshow(
        result.mel_spectrogram,
        sr=result.sr,
        hop_length=512,
        x_axis="time",
        y_axis="mel",
        fmax=np.max(result.mel_frequencies),
        ax=ax,
    )
    ax.plot(result.f0_times, result.f0_hz, color="white", linewidth=2, label="f0")
    ymin, ymax = ax.get_ylim()
    _plot_segments(ax, result.segments, ymin, ymax)
    fig.colorbar(img, ax=ax, format="%+2.0f dB")
    ax.set_title("Mel-espectrograma + f0 + segmentos funcionales")
    ax.legend()
    plt.tight_layout()
    plt.show()


def plot_melody_self_similarity(result: MelodyAnalysisResult):
    """Display the melody self-similarity matrix."""

    fig, ax = plt.subplots(figsize=(6, 6))
    cax = ax.imshow(result.melody_self_similarity, origin="lower", aspect="auto", cmap="magma")
    ax.set_xlabel("Frames")
    ax.set_ylabel("Frames")
    ax.set_title("Matriz de autosimilitud melódica")
    fig.colorbar(cax, ax=ax)
    plt.tight_layout()
    plt.show()


def plot_comparison_f0_and_segments(
    result_ref: MelodyAnalysisResult,
    result_cover: MelodyAnalysisResult,
    normalize_time: bool = True,
):
    """Compare two analyses (e.g., original vs cover) aligning time axes if desired."""

    fig, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=normalize_time)

    def _plot(ax: plt.Axes, result: MelodyAnalysisResult, title: str):
        times = result.f0_times
        if normalize_time and len(times) > 0:
            times = times / times[-1]
        ax.plot(times, result.f0_hz, label="f0")
        ymin, ymax = ax.get_ylim()
        _plot_segments(ax, result.segments, ymin, ymax)
        ax.set_ylabel("Hz")
        ax.set_title(title)
        ax.legend()

    _plot(axes[0], result_ref, "Referencia")
    _plot(axes[1], result_cover, "Cover")
    axes[-1].set_xlabel("Tiempo normalizado" if normalize_time else "Tiempo (s)")
    plt.tight_layout()
    plt.show()


# ============================================================================
# Example usage
# ============================================================================


if __name__ == "__main__":
    # Ejemplo 1: análisis de una sola pieza
    audio_path = "AClassicEducation_NightOwl_MIX.wav"  # ejemplo
    analyzer = MelodyAnalyzer(pitch_method="crepe")  # o "pyin", "yin"
    result = analyzer.analyze_file(audio_path)

    print("Segmentos detectados (pieza única):")
    for s in result.segments:
        print(
            f"{s.label:>12} | "
            f"{s.segment.start_time:7.3f} → {s.segment.end_time:7.3f} s "
            f"(conf={s.confidence:.2f})"
        )

    # Exportar a JSON
    json_str = analyzer.to_json(result)
    with open("analysis_segments.json", "w", encoding="utf-8") as f:
        f.write(json_str)

    # Visualización básica del f0 (solo contorno)
    plot_f0_contour(result, show_segments=False)

    # Visualización del f0 con segmentos
    plot_f0_contour(result, show_segments=True)

    # Visualización del contorno melódico normalizado
    plot_melodic_contour(result, normalize=True, show_segments=True)

    # Visualización completa mel + f0 + segmentos
    plot_mel_and_f0_with_segments(result)

    # Matriz de autosimilitud
    plot_melody_self_similarity(result)

    # Ejemplo 2: comparación original vs cover
    original_path = "original.wav"
    cover_path = "cover.wav"

    result_original = analyzer.analyze_file(original_path)
    result_cover = analyzer.analyze_file(cover_path)

    print("Comparación original vs cover (segmentos originales):")
    for s in result_original.segments:
        print(
            f"[ORIGINAL] {s.label:>12} | "
            f"{s.segment.start_time:7.3f} → {s.segment.end_time:7.3f} s "
            f"(conf={s.confidence:.2f})"
        )

    print("Comparación original vs cover (segmentos cover):")
    for s in result_cover.segments:
        print(
            f"[COVER]    {s.label:>12} | "
            f"{s.segment.start_time:7.3f} → {s.segment.end_time:7.3f} s "
            f"(conf={s.confidence:.2f})"
        )

    # Visualización comparativa de f0 + segmentos
    plot_comparison_f0_and_segments(result_original, result_cover, normalize_time=True)
