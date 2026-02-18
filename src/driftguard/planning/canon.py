from __future__ import annotations

import re
from typing import Dict, List

from driftguard.utils.types import Chunk


def _extract_entities(text: str) -> List[str]:
    tokens = re.findall(r"[A-Za-z][A-Za-z0-9_-]{2,}", text)
    seen = set()
    out: List[str] = []
    for t in tokens:
        key = t.lower()
        if key in seen:
            continue
        seen.add(key)
        out.append(t)
    return out[:20]


def build_canon(storyline: str, retrieved_chunks: List[Chunk]) -> Dict[str, object]:
    ref_text = " ".join(c.text for c in retrieved_chunks[:8]).strip()
    merged = f"{storyline} {ref_text}".strip()
    entities = _extract_entities(merged)
    return {
        "summary": storyline.strip(),
        "entities": entities,
        "context_snippets": [c.text[:180] for c in retrieved_chunks[:5]],
    }
