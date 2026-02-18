from __future__ import annotations

from typing import Dict


def canon_alignment_score(prompt: str, canon: Dict[str, object]) -> float:
    entities = canon.get("entities", [])
    if not isinstance(entities, list) or not entities:
        return 0.5
    entities = entities[:8]
    lower_prompt = prompt.lower()
    hit = 0
    for e in entities:
        if isinstance(e, str) and e.lower() in lower_prompt:
            hit += 1
    return hit / max(len(entities), 1)
