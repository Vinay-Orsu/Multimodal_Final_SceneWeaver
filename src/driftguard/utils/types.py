from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class Chunk:
    chunk_id: str
    text: str
    source: str
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class StoryWindow:
    index: int
    start_sec: int
    end_sec: int
    beat: str
    prompt: str = ""


@dataclass
class WindowResult:
    window: StoryWindow
    prompt: str
    output_path: str
    generated: bool
    critic_scores: Dict[str, float]
    accepted: bool
    notes: Optional[str] = None
