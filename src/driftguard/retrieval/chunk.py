from __future__ import annotations

from typing import List

from driftguard.utils.types import Chunk


def chunk_text(
    text: str,
    source: str,
    chunk_size_chars: int = 600,
    chunk_overlap_chars: int = 120,
) -> List[Chunk]:
    if chunk_size_chars <= 0:
        raise ValueError("chunk_size_chars must be > 0")
    if chunk_overlap_chars < 0:
        raise ValueError("chunk_overlap_chars must be >= 0")
    if chunk_overlap_chars >= chunk_size_chars:
        raise ValueError("chunk_overlap_chars must be smaller than chunk_size_chars")

    chunks: List[Chunk] = []
    start = 0
    i = 0
    n = len(text)
    while start < n:
        end = min(n, start + chunk_size_chars)
        part = text[start:end].strip()
        if part:
            chunks.append(
                Chunk(
                    chunk_id=f"{source}_{i:04d}",
                    text=part,
                    source=source,
                    metadata={"start": start, "end": end},
                )
            )
            i += 1
        if end >= n:
            break
        start = end - chunk_overlap_chars
    return chunks
