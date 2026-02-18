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

        windows: List[SceneWindow] = []
        for i in range(window_count):
            beat = beats[min(i, len(beats) - 1)] if i < len(beats) else beats[i % len(beats)]
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

        context = {
            "storyline": storyline,
            "window_index": window.index,
            "window_time": f"{window.start_sec}-{window.end_sec}s",
            "beat": window.beat,
            "previous_prompt": previous_prompt,
            "memory_feedback": memory_feedback or {},
        }
        prompt = (
            "You are a scene director for text-to-video generation.\n"
            "Return JSON only: {\"prompt\": \"...\"}.\n"
            "Make it cinematic, temporally continuous, and consistent with previous clip.\n"
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
            return parsed["prompt"].strip()
        return self._heuristic_refine_prompt(window, previous_prompt, memory_feedback)

    @staticmethod
    def _extract_story_beats(storyline: str) -> List[str]:
        raw = re.split(r"[.\n;]+", storyline)
        beats = [b.strip() for b in raw if b.strip()]
        return beats

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
        previous_context = f" Previous visual context: {previous_prompt}" if previous_prompt else ""
        return (
            f"{window.prompt_seed}. Time window {window.start_sec}-{window.end_sec}s, "
            f"{continuity}.{previous_context}"
        )

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
