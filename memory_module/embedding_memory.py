from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional


@dataclass
class VisionEmbedderConfig:
    backend: str = "clip"  # clip | dinov2
    model_id: Optional[str] = None
    device: str = "auto"


@dataclass
class MemoryFeedback:
    window_index: int
    local_similarity: Optional[float]
    global_similarity: Optional[float]
    drift_detected: bool
    suggested_constraints: str

    def to_dict(self) -> Dict[str, Any]:
        return {
            "window_index": self.window_index,
            "local_similarity": self.local_similarity,
            "global_similarity": self.global_similarity,
            "drift_detected": self.drift_detected,
            "suggested_constraints": self.suggested_constraints,
        }


class VisionEmbedder:
    """
    Visual embedding extractor for generated frames using CLIP or DINOv2.
    """

    def __init__(self, config: VisionEmbedderConfig):
        self.config = config
        self.processor = None
        self.model = None
        self.torch = None
        self._resolved_device = "cpu"

    def load(self) -> None:
        try:
            import torch
            from transformers import (
                AutoImageProcessor,
                AutoModel,
                CLIPModel,
                CLIPProcessor,
            )
        except ImportError as exc:
            raise ImportError(
                "transformers and torch are required for embedding extraction."
            ) from exc

        self.torch = torch
        self._resolved_device = self._resolve_device(torch)

        backend = self.config.backend.lower()
        if backend == "clip":
            model_id = self.config.model_id or "openai/clip-vit-base-patch32"
            self.processor = CLIPProcessor.from_pretrained(model_id)
            self.model = CLIPModel.from_pretrained(model_id).to(self._resolved_device)
            self.model.eval()
            return

        if backend == "dinov2":
            model_id = self.config.model_id or "facebook/dinov2-base"
            self.processor = AutoImageProcessor.from_pretrained(model_id)
            self.model = AutoModel.from_pretrained(model_id).to(self._resolved_device)
            self.model.eval()
            return

        raise ValueError("Unsupported backend. Use 'clip' or 'dinov2'.")

    def embed_frames(self, frames: List[Any], sample_count: int = 4):
        if self.model is None or self.processor is None or self.torch is None:
            raise RuntimeError("VisionEmbedder not loaded. Call load() first.")
        if not frames:
            raise ValueError("frames is empty.")

        sampled = self._sample_frames(frames, sample_count=sample_count)
        pil_frames = [self._to_pil_image(f) for f in sampled]

        backend = self.config.backend.lower()
        if backend == "clip":
            inputs = self.processor(images=pil_frames, return_tensors="pt")
            inputs = {k: v.to(self._resolved_device) for k, v in inputs.items()}
            with self.torch.no_grad():
                embeds = self.model.get_image_features(**inputs)
        else:
            inputs = self.processor(images=pil_frames, return_tensors="pt")
            inputs = {k: v.to(self._resolved_device) for k, v in inputs.items()}
            with self.torch.no_grad():
                out = self.model(**inputs)
            embeds = out.last_hidden_state.mean(dim=1)

        clip_embedding = embeds.mean(dim=0)
        clip_embedding = clip_embedding / (clip_embedding.norm() + 1e-12)
        return clip_embedding.detach().cpu().numpy()

    @staticmethod
    def _sample_frames(frames: List[Any], sample_count: int) -> List[Any]:
        if len(frames) <= sample_count:
            return frames
        stride = max(1, len(frames) // sample_count)
        sampled = frames[::stride][:sample_count]
        return sampled

    @staticmethod
    def _to_pil_image(frame: Any):
        from PIL import Image
        import numpy as np

        if isinstance(frame, Image.Image):
            return frame.convert("RGB")
        if isinstance(frame, np.ndarray):
            if frame.dtype != np.uint8:
                frame = frame.clip(0, 255).astype(np.uint8)
            return Image.fromarray(frame).convert("RGB")
        raise TypeError("Frame type not supported for embedding extraction.")

    def _resolve_device(self, torch_module: Any) -> str:
        if self.config.device != "auto":
            return self.config.device
        if torch_module.cuda.is_available():
            return "cuda"
        if hasattr(torch_module.backends, "mps") and torch_module.backends.mps.is_available():
            return "mps"
        return "cpu"


class NarrativeMemory:
    """
    Tracks local (previous clip) and global (running) embeddings.
    """

    def __init__(self, local_threshold: float = 0.25, global_threshold: float = 0.20):
        self.local_threshold = local_threshold
        self.global_threshold = global_threshold
        self.local_embedding = None
        self.global_embedding = None
        self.window_count = 0

    def register_window(self, window_index: int, embedding) -> MemoryFeedback:
        import numpy as np

        local_sim = self._cosine_similarity(embedding, self.local_embedding)
        global_sim = self._cosine_similarity(embedding, self.global_embedding)

        drift_detected = False
        notes: List[str] = []
        if local_sim is not None and local_sim < self.local_threshold:
            drift_detected = True
            notes.append("Preserve recent motion trajectory, camera direction, and subject identity.")
        if global_sim is not None and global_sim < self.global_threshold:
            drift_detected = True
            notes.append("Reconnect with core storyline entities, setting, and mood from earlier windows.")
        if not notes:
            notes.append("Maintain style and continuity while advancing the next story beat.")

        if self.global_embedding is None:
            self.global_embedding = embedding
        else:
            n = self.window_count
            merged = (self.global_embedding * n + embedding) / (n + 1)
            self.global_embedding = merged / (np.linalg.norm(merged) + 1e-12)

        self.local_embedding = embedding
        self.window_count += 1

        return MemoryFeedback(
            window_index=window_index,
            local_similarity=local_sim,
            global_similarity=global_sim,
            drift_detected=drift_detected,
            suggested_constraints=" ".join(notes),
        )

    @staticmethod
    def _cosine_similarity(v1, v2) -> Optional[float]:
        if v1 is None or v2 is None:
            return None
        import numpy as np

        denom = (np.linalg.norm(v1) * np.linalg.norm(v2)) + 1e-12
        return float(np.dot(v1, v2) / denom)
