from __future__ import annotations

import re


TOKEN_RE = re.compile(r"[a-zA-Z0-9_]+")


def _tokenize(text: str) -> set[str]:
    return set(TOKEN_RE.findall(text.lower()))


def prompt_similarity(prompt_a: str, prompt_b: str) -> float:
    ta = _tokenize(prompt_a)
    tb = _tokenize(prompt_b)
    if not ta or not tb:
        return 0.0
    return len(ta & tb) / max(len(ta | tb), 1)
