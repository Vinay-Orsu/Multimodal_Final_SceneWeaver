from __future__ import annotations

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
        try:
            import torch
            from diffusers import AutoPipelineForText2Video
        except ImportError as exc:
            raise ImportError(
                "Missing dependencies. Install at least: torch, diffusers, transformers, accelerate."
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
        pipe = AutoPipelineForText2Video.from_pretrained(
            self.config.model_id,
            torch_dtype=torch_dtype,
            trust_remote_code=True,
        )

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
        except ImportError as exc:
            raise ImportError("imageio is required to save MP4/GIF outputs.") from exc

        iio.imwrite(out.as_posix(), frames, fps=fps)
