from __future__ import annotations

import re
from dataclasses import dataclass
from typing import List, Tuple

from driftguard.utils.types import Chunk


TOKEN_RE = re.compile(r"[a-zA-Z0-9_]+")


def _tokenize(text: str) -> set[str]:
    return set(TOKEN_RE.findall(text.lower()))


@dataclass
class SimpleIndex:
    chunks: List[Chunk]
    token_sets: List[set[str]]

    @classmethod
    def build(cls, chunks: List[Chunk]) -> "SimpleIndex":
        return cls(chunks=chunks, token_sets=[_tokenize(c.text) for c in chunks])

    def search(self, query: str, top_k: int = 5) -> List[Tuple[Chunk, float]]:
        q = _tokenize(query)
        if not q:
            return []
        scored: List[Tuple[Chunk, float]] = []
        for chunk, ts in zip(self.chunks, self.token_sets):
            inter = len(q & ts)
            if inter == 0:
                continue
            union = len(q | ts)
            score = inter / max(union, 1)
            scored.append((chunk, score))
        scored.sort(key=lambda x: x[1], reverse=True)
        return scored[:top_k]
