from __future__ import annotations

import json
import platform
from dataclasses import dataclass
from pathlib import Path
from typing import Any, List, Optional


@dataclass
class WanBackboneConfig:
    model_id: str = "Wan-AI/Wan2.0-T2V-14B"
    torch_dtype: str = "bfloat16"
    device: str = "auto"
    enable_cpu_offload: bool = True


class WanBackbone:
    """
    Thin wrapper for loading and running Wan 2.0 text-to-video generation.
    """

    def __init__(self, config: WanBackboneConfig):
        self.config = config
        self.pipeline: Optional[Any] = None

    @staticmethod
    def _fix_text_encoder_embedding_tie(pipe: Any) -> None:
        """
        Some runtime stacks can leave UMT5 encoder token embeddings randomly initialized
        (shared.weight loaded, encoder.embed_tokens.weight missing). Force-tie them.
        """
        text_encoder = getattr(pipe, "text_encoder", None)
        if text_encoder is None:
            return

        shared = getattr(text_encoder, "shared", None)
        encoder = getattr(text_encoder, "encoder", None)
        embed_tokens = getattr(encoder, "embed_tokens", None) if encoder is not None else None
        if shared is None or embed_tokens is None:
            return
        if not hasattr(shared, "weight") or not hasattr(embed_tokens, "weight"):
            return

        if shared.weight.data_ptr() != embed_tokens.weight.data_ptr():
            embed_tokens.weight = shared.weight

    def _get_torch_dtype(self, torch_module: Any):
        dtype_name = self.config.torch_dtype.lower()
        if dtype_name == "float16":
            return torch_module.float16
        if dtype_name == "float32":
            return torch_module.float32
        if dtype_name == "bfloat16":
            return torch_module.bfloat16
        raise ValueError(
            f"Unsupported torch_dtype='{self.config.torch_dtype}'. "
            "Use one of: float16, float32, bfloat16."
        )

    def load(self) -> None:
        model_path = Path(self.config.model_id)
        if model_path.is_dir():
            model_index = model_path / "model_index.json"
            if not model_index.exists():
                hint = ""
                config_path = model_path / "config.json"
                if config_path.exists():
                    try:
                        with config_path.open("r", encoding="utf-8") as f:
                            config = json.load(f)
                        if config.get("_class_name") == "WanModel":
                            hint = (
                                " Detected native Wan checkpoint layout "
                                "(config _class_name='WanModel')."
                            )
                    except Exception:
                        pass
                raise RuntimeError(
                    f"Local model directory is missing model_index.json: {model_index}.{hint} "
                    "This runtime expects a diffusers pipeline directory for --video_model_id."
                )

        try:
            import torch
        except ImportError as exc:
            raise ImportError(
                "Missing/unsupported runtime dependencies for text-to-video. "
                "Install or upgrade: torch, diffusers, transformers, accelerate. "
                "Example: pip install -U 'diffusers>=0.30' transformers accelerate"
            ) from exc
        try:
            from diffusers import AutoPipelineForText2Video as PipelineClass
        except ImportError:
            try:
                # Fallback for diffusers builds that do not expose AutoPipelineForText2Video.
                from diffusers import DiffusionPipeline as PipelineClass
            except ImportError as exc:
                raise ImportError(
                    "Could not import a usable diffusers pipeline class. "
                    "Expected AutoPipelineForText2Video or DiffusionPipeline."
                ) from exc

        cuda_available = torch.cuda.is_available()
        mps_available = hasattr(torch.backends, "mps") and torch.backends.mps.is_available()

        # Resolve target device safely.
        if self.config.device == "auto":
            if cuda_available:
                resolved_device = "cuda"
            elif mps_available:
                resolved_device = "mps"
            else:
                resolved_device = "cpu"
        else:
            resolved_device = self.config.device

        if resolved_device == "cuda" and not cuda_available:
            if mps_available:
                resolved_device = "mps"
            else:
                resolved_device = "cpu"

        # Large Wan checkpoints are not practical on Apple Silicon without CUDA.
        model_id_lower = self.config.model_id.lower()
        is_very_large_model = "14b" in model_id_lower
        is_apple_silicon = platform.system() == "Darwin" and platform.machine().startswith("arm")
        if is_apple_silicon and resolved_device != "cuda" and is_very_large_model:
            raise RuntimeError(
                "Wan 2.0 14B is not a practical local target on Apple Silicon/MPS and can crash the process. "
                "Use a smaller Wan checkpoint for local tests, or run 14B on a CUDA GPU machine."
            )

        torch_dtype = self._get_torch_dtype(torch)
        # Some pipeline classes (e.g., WanPipeline) don't accept trust_remote_code.
        try:
            pipe = PipelineClass.from_pretrained(
                self.config.model_id,
                torch_dtype=torch_dtype,
            )
        except TypeError:
            pipe = PipelineClass.from_pretrained(
                self.config.model_id,
                torch_dtype=torch_dtype,
                trust_remote_code=True,
            )

        self._fix_text_encoder_embedding_tie(pipe)

        use_cpu_offload = self.config.enable_cpu_offload and resolved_device == "cuda"
        if use_cpu_offload:
            pipe.enable_model_cpu_offload()
        else:
            pipe.to(resolved_device)

        self.pipeline = pipe

    def generate_clip(
        self,
        prompt: str,
        negative_prompt: str = "",
        num_frames: int = 49,
        num_inference_steps: int = 30,
        guidance_scale: float = 6.0,
        height: int = 480,
        width: int = 832,
        seed: Optional[int] = None,
    ) -> List[Any]:
        if self.pipeline is None:
            raise RuntimeError("Pipeline is not loaded. Call load() first.")

        generator = None
        if seed is not None:
            try:
                import torch
            except ImportError as exc:
                raise ImportError("torch is required when using a fixed seed.") from exc
            generator = torch.Generator(device="cpu").manual_seed(seed)

        result = self.pipeline(
            prompt=prompt,
            negative_prompt=negative_prompt,
            num_frames=num_frames,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            height=height,
            width=width,
            generator=generator,
        )

        frames = getattr(result, "frames", None)
        if frames is None:
            raise RuntimeError("Pipeline output does not contain `frames`.")
        if len(frames) > 0 and isinstance(frames[0], list):
            return frames[0]
        return frames

    @staticmethod
    def save_video(frames: List[Any], output_path: str, fps: int = 8) -> None:
        out = Path(output_path)
        out.parent.mkdir(parents=True, exist_ok=True)

        try:
            import imageio.v3 as iio
            import numpy as np
        except ImportError as exc:
            raise ImportError("imageio is required to save MP4/GIF outputs.") from exc

        normalized = []
        for frame in frames:
            arr = np.asarray(frame)

            # Some pipelines return a whole clip per item: (T, H, W, C) or (T, C, H, W).
            if arr.ndim == 4:
                if arr.shape[-1] in (1, 2, 3, 4):
                    frame_batch = [arr[i] for i in range(arr.shape[0])]
                elif arr.shape[1] in (1, 2, 3, 4):
                    frame_batch = [np.transpose(arr[i], (1, 2, 0)) for i in range(arr.shape[0])]
                else:
                    raise ValueError(
                        f"Unsupported clip shape {arr.shape}. Expected TxHxWxC or TxCxHxW with C in 1..4."
                    )
            else:
                frame_batch = [arr]

            for f in frame_batch:
                # Convert CHW -> HWC when needed.
                if f.ndim == 3 and f.shape[0] in (1, 2, 3, 4) and f.shape[-1] not in (1, 2, 3, 4):
                    f = np.transpose(f, (1, 2, 0))

                if f.ndim == 2:
                    pass
                elif f.ndim == 3 and f.shape[-1] in (1, 2, 3, 4):
                    pass
                else:
                    raise ValueError(
                        f"Unsupported frame shape {f.shape}. Expected HxW, HxWxC, or CxHxW with C in 1..4."
                    )

                if f.dtype.kind in ("f", "c"):
                    f_min = float(np.min(f))
                    f_max = float(np.max(f))

                    # Common model output ranges:
                    # 1) [0, 1] -> scale directly
                    # 2) [-1, 1] -> shift+scale
                    # 3) already [0, 255] float -> clip/cast
                    if 0.0 <= f_min and f_max <= 1.0:
                        f = (f * 255.0).round().astype(np.uint8)
                    elif -1.1 <= f_min and f_max <= 1.1:
                        f = (((f + 1.0) / 2.0) * 255.0).round().astype(np.uint8)
                    else:
                        # Unknown float range: normalize dynamically to avoid near-black outputs.
                        if f_max > f_min:
                            f = ((f - f_min) / (f_max - f_min) * 255.0).round().astype(np.uint8)
                        else:
                            f = np.zeros_like(f, dtype=np.uint8)
                elif f.dtype != np.uint8:
                    f = np.clip(f, 0, 255).astype(np.uint8)

                normalized.append(f)

        try:
            video = np.stack(normalized, axis=0)
        except ValueError as exc:
            raise ValueError("Frames have inconsistent shapes and cannot be encoded into a video.") from exc

        iio.imwrite(out.as_posix(), video, fps=fps)
