"""Smoke tests for package-level imports."""

from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def test_package_exports_summarize_segments() -> None:
    """The top-level package should expose summarize_segments."""

    from melody_analysis import summarize_segments

    # The helper should be callable without additional setup when given no
    # segments.
    assert summarize_segments([]) == []
