from __future__ import annotations

import dataclasses
import re
from typing import Any, List, Optional, Tuple


@dataclasses.dataclass
class CaptionerConfig:
    model_id: str = "Salesforce/blip2-flan-t5-xl"
    device: str = "cpu"  # auto|cuda|mps|cpu
    max_new_tokens: int = 60
    num_beams: int = 1
    stub_fallback: bool = True


class Captioner:
    """
    Lightweight wrapper around BLIP-2 style vision-language captioners.
    Falls back to a stub captioner when model loading is unavailable
    (useful for tests or CPU-only quick runs).
    """

    def __init__(self, config: CaptionerConfig):
        self.config = config
        self.processor = None
        self.model = None
        self.torch = None
        self._stub = False

    def load(self) -> None:
        model_id = (self.config.model_id or "").strip()
        if model_id in {"", "stub", "__stub__"}:
            self._stub = True
            return

        try:
            import torch
            from transformers import AutoProcessor, Blip2ForConditionalGeneration
        except Exception:
            if self.config.stub_fallback:
                self._stub = True
                return
            raise

        self.torch = torch
        device = self._resolve_device(torch)
        self.processor = AutoProcessor.from_pretrained(model_id)
        self.model = Blip2ForConditionalGeneration.from_pretrained(
            model_id, torch_dtype=torch.float16 if device == "cuda" else torch.float32
        ).to(device)
        self.model.eval()

    def caption_frames(self, frames: List[Any]) -> Tuple[List[str], str, bool]:
        """
        Returns (captions_per_frame, merged_summary, has_duplicate_flag)
        """
        sampled = self._sample_frames(frames)
        if self._stub:
            captions = [self._stub_caption(i) for i in range(len(sampled))]
        else:
            if self.processor is None or self.model is None:
                raise RuntimeError("Captioner not loaded. Call load() first.")
            inputs = self.processor(images=sampled, return_tensors="pt").to(self.model.device)
            with self.torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=self.config.max_new_tokens,
                    num_beams=self.config.num_beams,
                )
            captions = self.processor.batch_decode(outputs, skip_special_tokens=True)
            captions = [c.strip() for c in captions]

        merged = self._merge_captions(captions)
        has_dupe = self._detect_duplicates(merged)
        return captions, merged, has_dupe

    # Helpers -----------------------------------------------------------------
    @staticmethod
    def _sample_frames(frames: List[Any], count: int = 3) -> List[Any]:
        if not frames:
            return []
        if len(frames) <= count:
            return frames
        stride = max(1, len(frames) // (count - 1))
        return [frames[0], frames[len(frames) // 2], frames[-1]][:count]

    @staticmethod
    def _merge_captions(captions: List[str]) -> str:
        if not captions:
            return ""
        # Keep unique nouns-ish words for a compact anchor.
        tokens = []
        seen = set()
        for c in captions:
            for tok in re.findall(r"[a-zA-Z][a-zA-Z0-9_-]{2,}", c.lower()):
                if tok in seen:
                    continue
                seen.add(tok)
                tokens.append(tok)
        summary = " ".join(tokens)
        return summary[:260] if summary else "scene anchor summary unavailable"

    @staticmethod
    def _detect_duplicates(summary: str) -> bool:
        text = summary.lower()
        dupe_patterns = [
            r"\btwo\b",
            r"\bthree\b",
            r"\bmultiple\b",
            r"\banother\b",
            r"\bseveral\b",
            r"\bextra\b",
            r"\bduplicate\b",
            r"\bduplicates\b",
            r"\bclones?\b",
        ]
        return any(re.search(p, text) for p in dupe_patterns)

    def _resolve_device(self, torch_module: Any) -> str:
        device = self.config.device
        if device == "auto":
            if torch_module.cuda.is_available():
                return "cuda"
            if hasattr(torch_module.backends, "mps") and torch_module.backends.mps.is_available():
                return "mps"
            return "cpu"
        return device

    @staticmethod
    def _stub_caption(idx: int) -> str:
        presets = [
            "a single crow beside a clay pot on a wooden table",
            "courtyard scene with one crow and scattered pebbles",
            "one crow near a pot; tree branch and daylight visible",
        ]
        return presets[idx % len(presets)]
