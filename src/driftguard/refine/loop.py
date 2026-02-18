from __future__ import annotations

from typing import Dict, Optional, Tuple

from driftguard.critics.canon import canon_alignment_score
from driftguard.critics.prompt import prompt_similarity


def combined_score(
    *,
    prompt: str,
    previous_prompt: str,
    canon: Dict[str, object],
    drift_score: float,
    weights: Dict[str, float],
) -> Tuple[float, Dict[str, float]]:
    pscore = prompt_similarity(prompt, previous_prompt) if previous_prompt else 0.5
    cscore = canon_alignment_score(prompt, canon)
    dscore = drift_score

    final_score = (
        weights.get("drift_weight", 0.4) * dscore
        + weights.get("canon_weight", 0.3) * cscore
        + weights.get("prompt_weight", 0.3) * pscore
    )
    details = {
        "drift_score": round(dscore, 4),
        "canon_score": round(cscore, 4),
        "prompt_score": round(pscore, 4),
        "final_score": round(final_score, 4),
    }
    return final_score, details


def should_accept(score: float, acceptance_score: float) -> bool:
    return score >= acceptance_score


def repair_constraints_from_scores(score_details: Dict[str, float]) -> Optional[str]:
    notes = []
    if score_details.get("drift_score", 1.0) < 0.35:
        notes.append("preserve recent motion and camera trajectory")
    if score_details.get("canon_score", 1.0) < 0.35:
        notes.append("reinforce canon entities and setting")
    if score_details.get("prompt_score", 1.0) < 0.35:
        notes.append("keep lexical continuity with previous clip description")
    if not notes:
        return None
    return "; ".join(notes)
