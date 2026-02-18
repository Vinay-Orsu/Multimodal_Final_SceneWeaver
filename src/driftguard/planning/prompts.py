from __future__ import annotations

from typing import Dict, Optional

from driftguard.utils.types import StoryWindow


def build_window_prompt(
    window: StoryWindow,
    canon: Dict[str, object],
    previous_prompt: str,
    repair_constraints: Optional[str] = None,
) -> str:
    entities = canon.get("entities", [])
    entity_hint = ", ".join(entities[:4]) if isinstance(entities, list) else ""

    if window.index == 0:
        base = (
            f"Establishing cinematic shot. Story beat: {window.beat}. "
            f"Preserve character identity and setting continuity."
        )
    else:
        base = (
            f"Continue naturally from previous clip with temporal continuity. "
            f"Story beat: {window.beat}."
        )
    if entity_hint:
        base += f" Key entities: {entity_hint}."
    if previous_prompt:
        base += f" Previous context: {previous_prompt}"
    if repair_constraints:
        base += f" Repair constraints: {repair_constraints}"
    return base.strip()


def build_repair_prompt(window: StoryWindow, critic_notes: str) -> str:
    return (
        f"Repair drift for window {window.index}. Keep story beat '{window.beat}' and apply constraints: "
        f"{critic_notes}. Maintain continuity with previous and global story state."
    )
