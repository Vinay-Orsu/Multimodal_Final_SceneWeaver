from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import sys
import platform
from typing import Any, Dict, List, Optional, Tuple

from driftguard.utils.io import ensure_dir, write_text


@dataclass
class VideoGenConfig:
    model_id: str
    dtype: str = "bfloat16"
    device: str = "auto"
    cpu_offload: bool = True
    fps: int = 8
    num_frames: int = 49
    num_steps: int = 30
    guidance_scale: float = 6.0
    height: int = 480
    width: int = 832


class VideoGenerator:
    """
    Thin adapter around legacy WanBackbone for actual generation.
    In dry-run mode, generates placeholder artifacts for pipeline validation.
    """

    def __init__(self, config: VideoGenConfig, dry_run: bool = True):
        self.config = config
        self.dry_run = dry_run
        self._backend = None

    def load(self) -> None:
        if self.dry_run:
            return
        # Hard guard: avoid local hard-crash path on Apple Silicon for large Wan runs.
        is_apple_silicon = platform.system() == "Darwin" and platform.machine().startswith("arm")
        is_large_wan = "wan" in self.config.model_id.lower() and "14b" in self.config.model_id.lower()
        if is_apple_silicon and is_large_wan:
            raise RuntimeError(
                "Real generation with Wan 14B is blocked on Apple Silicon in this pipeline because it can "
                "segfault during backend initialization. Run this command on a CUDA cluster node."
            )
        try:
            # Ensure legacy top-level modules are importable when running from scripts/.
            repo_root = Path(__file__).resolve().parents[3]
            if str(repo_root) not in sys.path:
                sys.path.insert(0, str(repo_root))
            from video_backbone import WanBackbone, WanBackboneConfig
        except ImportError as exc:
            raise ImportError(
                "video_backbone.WanBackbone not available. Keep dry_run or install/graft backend."
            ) from exc
        self._backend = WanBackbone(
            WanBackboneConfig(
                model_id=self.config.model_id,
                torch_dtype=self.config.dtype,
                device=self.config.device,
                enable_cpu_offload=self.config.cpu_offload,
            )
        )
        self._backend.load()

    def generate(
        self,
        prompt: str,
        output_path: Path,
        seed: Optional[int] = None,
    ) -> Tuple[bool, Optional[List[Any]], Dict[str, Any]]:
        ensure_dir(output_path.parent)
        if self.dry_run:
            write_text(output_path.with_suffix(".txt"), prompt)
            return True, None, {"mode": "dry_run", "artifact": output_path.with_suffix(".txt").as_posix()}

        if self._backend is None:
            raise RuntimeError(
                "Video backend is not loaded. Call load() before generate(), or run in dry_run mode."
            )

        frames = self._backend.generate_clip(
            prompt=prompt,
            num_frames=self.config.num_frames,
            num_inference_steps=self.config.num_steps,
            guidance_scale=self.config.guidance_scale,
            height=self.config.height,
            width=self.config.width,
            seed=seed,
        )
        self._backend.save_video(frames=frames, output_path=output_path.as_posix(), fps=self.config.fps)
        return True, frames, {"mode": "generated", "artifact": output_path.as_posix()}
