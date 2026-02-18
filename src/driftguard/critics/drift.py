from __future__ import annotations

from typing import Any, Optional

import numpy as np


def cosine_similarity(v1: Optional[np.ndarray], v2: Optional[np.ndarray]) -> float:
    if v1 is None or v2 is None:
        return 0.5
    denom = (np.linalg.norm(v1) * np.linalg.norm(v2)) + 1e-12
    return float(np.dot(v1, v2) / denom)


def drift_score_from_embeddings(local_embedding: Any, previous_embedding: Any, global_embedding: Any) -> float:
    local_sim = cosine_similarity(local_embedding, previous_embedding)
    global_sim = cosine_similarity(local_embedding, global_embedding)
    return float((local_sim + global_sim) / 2.0)
