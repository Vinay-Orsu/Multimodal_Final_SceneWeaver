from __future__ import annotations

from typing import Dict

from driftguard.utils.types import StoryWindow


def overlap_meta(window: StoryWindow, overlap_seconds: int) -> Dict[str, int]:
    if overlap_seconds <= 0:
        return {"overlap_start": 0, "overlap_end": 0}
    length = max(0, window.end_sec - window.start_sec)
    ov = min(overlap_seconds, length)
    return {
        "overlap_start": ov,
        "overlap_end": ov,
    }
