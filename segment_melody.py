"""Command line entry-point for melody segmentation and classification."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from melody_analysis import MelodySegmenter, summarize_segments


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Segment and classify melodic structure")
    parser.add_argument("audio", type=Path, help="Path to the audio file to analyse")
    parser.add_argument("--segments", type=int, default=None, help="Approximate number of target segments")
    parser.add_argument(
        "--min-duration",
        type=float,
        default=1.5,
        help="Minimum allowed segment duration in seconds before merging",
    )
    parser.add_argument("--json", type=Path, help="Path to save the resulting JSON summary", default=None)
    parser.add_argument(
        "--pitch-backend",
        choices=["pyin", "crepe"],
        default="pyin",
        help="Select the pitch extraction backend (requires the crepe package when chosen)",
    )
    parser.add_argument(
        "--crepe-step-ms",
        type=float,
        default=20.0,
        help="Temporal resolution in milliseconds for CREPE predictions when using that backend",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    segmenter = MelodySegmenter(pitch_backend=args.pitch_backend, crepe_step_size=args.crepe_step_ms)
    segments = segmenter.segment(
        str(args.audio), target_segments=args.segments, min_duration=args.min_duration
    )
    summary = summarize_segments(segments)

    if args.json is not None:
        args.json.write_text(json.dumps(summary, indent=2, ensure_ascii=False))
    else:
        print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
