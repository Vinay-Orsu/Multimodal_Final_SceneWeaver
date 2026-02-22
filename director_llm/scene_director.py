from __future__ import annotations

import json
import math
import re
from dataclasses import dataclass
from typing import Any, Dict, List, Optional


@dataclass
class SceneWindow:
    index: int
    start_sec: int
    end_sec: int
    beat: str
    prompt_seed: str


@dataclass
class SceneDirectorConfig:
    model_id: Optional[str] = None
    temperature: float = 0.7
    max_new_tokens: int = 512


class SceneDirector:
    """
    Converts a storyline into scene windows and refines prompts per window.
    """

    def __init__(self, config: SceneDirectorConfig, window_seconds: int = 10):
        self.config = config
        self.window_seconds = window_seconds
        self._generator = None

    def load(self) -> None:
        if not self.config.model_id:
            return
        try:
            from transformers import pipeline
        except ImportError as exc:
            raise ImportError(
                "transformers is required to run SceneDirector with --director_model_id."
            ) from exc
        self._generator = pipeline("text-generation", model=self.config.model_id)

    def plan_windows(self, storyline: str, total_minutes: float) -> List[SceneWindow]:
        window_count = max(1, math.ceil((total_minutes * 60) / self.window_seconds))

        beats = self._extract_story_beats(storyline)
        if not beats:
            beats = [storyline.strip() or "Continue the story naturally."]

        beat_plan = self._expand_beats_for_windows(beats, window_count)
        windows: List[SceneWindow] = []
        for i in range(window_count):
            beat = beat_plan[i]
            start_sec = i * self.window_seconds
            end_sec = start_sec + self.window_seconds
            prompt_seed = self._default_visual_seed(beat=beat, is_opening=(i == 0))
            windows.append(
                SceneWindow(
                    index=i,
                    start_sec=start_sec,
                    end_sec=end_sec,
                    beat=beat,
                    prompt_seed=prompt_seed,
                )
            )
        return windows

    def refine_prompt(
        self,
        storyline: str,
        window: SceneWindow,
        previous_prompt: str,
        memory_feedback: Optional[Dict[str, Any]],
    ) -> str:
        if self._generator is None:
            return self._heuristic_refine_prompt(window, previous_prompt, memory_feedback)

        compact_prev = self._compact_previous_prompt(previous_prompt)
        compact_storyline = self._compact_storyline(storyline)
        memory_text = self._memory_feedback_text(memory_feedback)
        context = {
            "storyline": compact_storyline,
            "window_index": window.index,
            "window_time": f"{window.start_sec}-{window.end_sec}s",
            "beat": window.beat,
            "previous_prompt": compact_prev,
            "memory_feedback": memory_text,
        }
        prompt = (
            "You are a strict scene director for text-to-video generation.\n"
            "Task: produce ONE concise shot prompt for the CURRENT window only.\n"
            "Rules:\n"
            "1) Keep subject identity and scene continuity from previous prompt.\n"
            "2) Advance only the current beat.\n"
            "3) Avoid adding new random characters or objects.\n"
            "4) Keep camera and motion explicit and realistic.\n"
            "Return JSON only: {\"prompt\": \"...\"}\n"
            f"Context:\n{json.dumps(context, ensure_ascii=False)}\n"
        )

        out = self._generator(
            prompt,
            max_new_tokens=self.config.max_new_tokens,
            temperature=self.config.temperature,
            do_sample=True,
        )[0]["generated_text"]

        parsed = self._extract_json_object(out)
        if parsed and isinstance(parsed.get("prompt"), str):
            cleaned = self._normalize_prompt(parsed["prompt"])
            if cleaned:
                return cleaned
        return self._heuristic_refine_prompt(window, previous_prompt, memory_feedback)

    @staticmethod
    def _extract_story_beats(storyline: str) -> List[str]:
        text = storyline.strip()
        if not text:
            return []

        # Prefer sentence/semicolon boundaries first.
        beats = [b.strip() for b in re.split(r"[.\n;:]+", text) if b.strip()]

        # If user provided one long comma-delimited line, split that into beats.
        if len(beats) <= 1 and "," in text:
            beats = [b.strip() for b in re.split(r",\s*", text) if b.strip()]

        # If still a single block, break gentle temporal connectors into beats.
        if len(beats) <= 1:
            beats = [
                b.strip()
                for b in re.split(r"\b(?:then|after that|next|finally|eventually)\b", text, flags=re.IGNORECASE)
                if b.strip()
            ]

        return beats

    @staticmethod
    def _expand_beats_for_windows(beats: List[str], window_count: int) -> List[str]:
        if len(beats) >= window_count:
            return beats[:window_count]

        # Spread beats across windows so each beat gets contiguous windows.
        plan: List[str] = []
        for i, beat in enumerate(beats):
            start = round(i * window_count / len(beats))
            end = round((i + 1) * window_count / len(beats))
            count = max(1, end - start)
            for j in range(count):
                phase = "start" if j == 0 else ("end" if j == count - 1 else "middle")
                plan.append(f"{beat} ({phase} phase)")
        return plan[:window_count]

    @staticmethod
    def _default_visual_seed(beat: str, is_opening: bool) -> str:
        if is_opening:
            return (
                "Establishing cinematic shot, clear subjects, stable identity and environment, "
                f"story action: {beat}"
            )
        return f"Continue naturally from previous clip with motion continuity, story action: {beat}"

    @staticmethod
    def _heuristic_refine_prompt(
        window: SceneWindow,
        previous_prompt: str,
        memory_feedback: Optional[Dict[str, Any]],
    ) -> str:
        continuity = "maintain continuity with previous clip"
        if memory_feedback:
            note = memory_feedback.get("suggested_constraints")
            if isinstance(note, str) and note.strip():
                continuity = note.strip()
        previous_context = ""
        if previous_prompt:
            compact_prev = previous_prompt.split(" Previous visual context:")[0].strip()
            compact_prev = compact_prev[:240]
            previous_context = f" Previous visual context: {compact_prev}"
        base = (
            f"{window.prompt_seed}. Time window {window.start_sec}-{window.end_sec}s, "
            f"{continuity}.{previous_context}"
        )
        return SceneDirector._normalize_prompt(base)

    @staticmethod
    def _normalize_prompt(text: str) -> str:
        prompt = " ".join((text or "").strip().split())
        return prompt[:700]

    @staticmethod
    def _compact_previous_prompt(text: str) -> str:
        if not text:
            return ""
        cleaned = text.split(" Previous visual context:")[0].strip()
        return cleaned[:260]

    @staticmethod
    def _compact_storyline(storyline: str) -> str:
        text = " ".join((storyline or "").split())
        return text[:500]

    @staticmethod
    def _memory_feedback_text(memory_feedback: Optional[Dict[str, Any]]) -> str:
        if not memory_feedback:
            return ""
        note = memory_feedback.get("suggested_constraints")
        if isinstance(note, str):
            return note[:220]
        return ""

    @staticmethod
    def _extract_json_object(text: str) -> Optional[Dict[str, Any]]:
        start = text.find("{")
        end = text.rfind("}")
        if start == -1 or end == -1 or end <= start:
            return None
        candidate = text[start : end + 1]
        try:
            parsed = json.loads(candidate)
        except json.JSONDecodeError:
            return None
        if isinstance(parsed, dict):
            return parsed
        return None
