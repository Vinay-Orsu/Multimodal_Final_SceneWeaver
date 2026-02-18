from __future__ import annotations

import math
import re
from typing import List

from driftguard.utils.types import StoryWindow


def _split_beats(storyline: str) -> List[str]:
    parts = [p.strip() for p in re.split(r"[.;\n]+", storyline) if p.strip()]
    return parts if parts else [storyline.strip()]


def make_storyboard(storyline: str, total_minutes: float, window_seconds: int) -> List[StoryWindow]:
    beats = _split_beats(storyline)
    window_count = max(1, math.ceil((total_minutes * 60) / window_seconds))

    windows: List[StoryWindow] = []
    for i in range(window_count):
        beat = beats[i] if i < len(beats) else beats[-1]
        start = i * window_seconds
        windows.append(
            StoryWindow(
                index=i,
                start_sec=start,
                end_sec=start + window_seconds,
                beat=beat,
                prompt="",
            )
        )
    return windows
