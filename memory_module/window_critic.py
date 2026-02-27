from __future__ import annotations

import re
from dataclasses import dataclass
from typing import List, Optional, Set


@dataclass
class CandidateCriticResult:
    final_score: float
    story_progress_score: float
    continuity_score: Optional[float]
    feedback: str

    def to_dict(self) -> dict:
        return {
            "final_score": self.final_score,
            "story_progress_score": self.story_progress_score,
            "continuity_score": self.continuity_score,
            "feedback": self.feedback,
        }


def _tokenize(text: str) -> Set[str]:
    return {w for w in re.findall(r"[a-z0-9]+", (text or "").lower()) if len(w) > 2}


def _strip_phase(beat: str) -> str:
    return re.sub(r"\s*\((start|middle|end)\s+phase\)\s*$", "", (beat or "").strip(), flags=re.IGNORECASE)


def story_progress_score(previous_beat: str, current_beat: str) -> float:
    if not previous_beat.strip():
        return 1.0

    prev_core = _strip_phase(previous_beat)
    curr_core = _strip_phase(current_beat)
    if prev_core.lower() == curr_core.lower() and previous_beat.strip() != current_beat.strip():
        # Same core beat but phase changed; allow moderate score.
        return 0.6

    prev_tokens = _tokenize(previous_beat)
    curr_tokens = _tokenize(current_beat)
    union = prev_tokens | curr_tokens
    if not union:
        return 0.7
    overlap = len(prev_tokens & curr_tokens) / len(union)
    return max(0.0, min(1.0, 1.0 - overlap))


def evaluate_candidate(
    *,
    current_beat: str,
    previous_beat: str,
    transition_similarity: Optional[float],
    environment_similarity: Optional[float],
    continuity_score: Optional[float],
    story_weight: float = 0.15,
    continuity_weight: float = 0.85,
    attempt_index: int = 0,
) -> CandidateCriticResult:
    story_score = story_progress_score(previous_beat=previous_beat, current_beat=current_beat)

    base_continuity = continuity_score
    if base_continuity is None:
        vals: List[float] = []
        if transition_similarity is not None:
            vals.append(float(transition_similarity))
        if environment_similarity is not None:
            vals.append(float(environment_similarity))
        base_continuity = (sum(vals) / len(vals)) if vals else 0.5

    story_weight = max(0.0, float(story_weight))
    continuity_weight = max(0.0, float(continuity_weight))
    if story_weight == 0.0 and continuity_weight == 0.0:
        final = float(base_continuity)
    else:
        total_weight = story_weight + continuity_weight
        final = float((story_score * story_weight + float(base_continuity) * continuity_weight) / total_weight)

    notes: List[str] = []
    if transition_similarity is not None and transition_similarity < 0.70:
        notes.append("Match opening frame composition and subject pose to previous window ending.")
    if environment_similarity is not None and environment_similarity < 0.65:
        notes.append("Keep the same location layout, anchor objects, and lighting.")
    if story_score < 0.40:
        notes.append("Advance to the current beat and avoid repeating the previous action.")
    if attempt_index > 0:
        notes.append("Do not restart the story progression; continue naturally from prior window.")
    if not notes:
        notes.append("Preserve continuity while advancing the current beat clearly.")

    return CandidateCriticResult(
        final_score=final,
        story_progress_score=story_score,
        continuity_score=continuity_score,
        feedback=" ".join(notes),
    )
