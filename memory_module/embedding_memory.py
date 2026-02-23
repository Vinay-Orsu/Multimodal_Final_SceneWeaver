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
    transition_similarity: Optional[float]
    drift_detected: bool
    suggested_constraints: str

    def to_dict(self) -> Dict[str, Any]:
        return {
            "window_index": self.window_index,
            "local_similarity": self.local_similarity,
            "global_similarity": self.global_similarity,
            "transition_similarity": self.transition_similarity,
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
        flattened_frames = self._prepare_flattened_frames(frames)
        sampled = self._sample_frames(flattened_frames, sample_count=sample_count)
        pil_frames = [self._to_pil_image(f) for f in sampled]
        embeds = self._encode_pil_frames(pil_frames)
        clip_embedding = embeds.mean(dim=0)
        clip_embedding = clip_embedding / (clip_embedding.norm() + 1e-12)
        return clip_embedding.detach().cpu().numpy()

    def embed_frame(self, frame: Any):
        if self.model is None or self.processor is None or self.torch is None:
            raise RuntimeError("VisionEmbedder not loaded. Call load() first.")
        pil_frame = self._to_pil_image(frame)
        embeds = self._encode_pil_frames([pil_frame])
        frame_embedding = embeds[0]
        frame_embedding = frame_embedding / (frame_embedding.norm() + 1e-12)
        return frame_embedding.detach().cpu().numpy()

    def embed_first_frame(self, frames: List[Any]):
        flattened_frames = self._prepare_flattened_frames(frames)
        return self.embed_frame(flattened_frames[0])

    def embed_last_frame(self, frames: List[Any]):
        flattened_frames = self._prepare_flattened_frames(frames)
        return self.embed_frame(flattened_frames[-1])

    def _prepare_flattened_frames(self, frames: List[Any]) -> List[Any]:
        if self.model is None or self.processor is None or self.torch is None:
            raise RuntimeError("VisionEmbedder not loaded. Call load() first.")
        if frames is None:
            raise ValueError("frames is empty.")
        try:
            frame_count = len(frames)
        except TypeError:
            raise TypeError("frames must be a sequence or array-like object.")
        if frame_count == 0:
            raise ValueError("frames is empty.")

        flattened_frames = self._flatten_frame_sequence(frames)
        if len(flattened_frames) == 0:
            raise ValueError("frames is empty after flattening.")
        return flattened_frames

    def _encode_pil_frames(self, pil_frames: List[Any]):
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
        return embeds

    @staticmethod
    def _sample_frames(frames: List[Any], sample_count: int) -> List[Any]:
        if len(frames) <= sample_count:
            return frames
        stride = max(1, len(frames) // sample_count)
        sampled = frames[::stride][:sample_count]
        return sampled

    @staticmethod
    def _flatten_frame_sequence(frames: Any) -> List[Any]:
        import numpy as np

        flattened: List[Any] = []
        for item in list(frames):
            arr = np.asarray(item)

            # Common batched clip layouts from video pipelines.
            if arr.ndim == 5 and arr.shape[0] == 1:
                arr = arr[0]
            if arr.ndim == 4:
                if arr.shape[-1] in (1, 2, 3, 4):
                    flattened.extend([arr[i] for i in range(arr.shape[0])])
                    continue
                if arr.shape[1] in (1, 2, 3, 4):
                    flattened.extend([np.transpose(arr[i], (1, 2, 0)) for i in range(arr.shape[0])])
                    continue

            flattened.append(item)

        return flattened

    @staticmethod
    def _to_pil_image(frame: Any):
        from PIL import Image
        import numpy as np

        if isinstance(frame, Image.Image):
            return frame.convert("RGB")
        if isinstance(frame, np.ndarray):
            arr = frame
            while arr.ndim > 3 and arr.shape[0] == 1:
                arr = arr[0]

            if arr.ndim == 3 and arr.shape[0] in (1, 2, 3, 4) and arr.shape[-1] not in (1, 2, 3, 4):
                arr = np.transpose(arr, (1, 2, 0))

            if arr.ndim == 2:
                pass
            elif arr.ndim == 3 and arr.shape[-1] in (1, 2, 3, 4):
                pass
            else:
                raise TypeError(f"Unsupported frame shape for embedding extraction: {arr.shape}")

            if arr.dtype.kind in ("f", "c"):
                arr_min = float(np.min(arr))
                arr_max = float(np.max(arr))
                if 0.0 <= arr_min and arr_max <= 1.0:
                    arr = (arr * 255.0).round().astype(np.uint8)
                elif -1.1 <= arr_min and arr_max <= 1.1:
                    arr = (((arr + 1.0) / 2.0) * 255.0).round().astype(np.uint8)
                elif arr_max > arr_min:
                    arr = ((arr - arr_min) / (arr_max - arr_min) * 255.0).round().astype(np.uint8)
                else:
                    arr = np.zeros_like(arr, dtype=np.uint8)
            elif arr.dtype != np.uint8:
                arr = np.clip(arr, 0, 255).astype(np.uint8)

            return Image.fromarray(arr).convert("RGB")
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

    def register_window(
        self,
        window_index: int,
        embedding,
        transition_similarity: Optional[float] = None,
    ) -> MemoryFeedback:
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
        if transition_similarity is not None and transition_similarity < 0.70:
            drift_detected = True
            notes.append("Match next clip opening frame to the previous final frame composition and subject pose.")
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
            transition_similarity=transition_similarity,
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
