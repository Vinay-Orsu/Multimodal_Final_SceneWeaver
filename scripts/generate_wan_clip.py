import argparse
import sys
from pathlib import Path

# Ensure project root is importable when running:
# `python scripts/generate_wan_clip.py ...`
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from video_backbone import WanBackbone, WanBackboneConfig


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate one clip using Wan 2.0.")
    parser.add_argument("--prompt", type=str, required=True, help="Text prompt for video generation.")
    parser.add_argument("--negative_prompt", type=str, default="", help="Optional negative prompt.")
    parser.add_argument("--model_id", type=str, default="Wan-AI/Wan2.0-T2V-14B", help="HF model id.")
    parser.add_argument("--output", type=str, default="outputs/wan_clip.mp4", help="Output video path.")
    parser.add_argument("--num_frames", type=int, default=49, help="Number of frames.")
    parser.add_argument("--steps", type=int, default=30, help="Diffusion steps.")
    parser.add_argument("--guidance_scale", type=float, default=6.0, help="CFG guidance scale.")
    parser.add_argument("--height", type=int, default=480, help="Video height.")
    parser.add_argument("--width", type=int, default=832, help="Video width.")
    parser.add_argument("--fps", type=int, default=8, help="Output fps.")
    parser.add_argument("--seed", type=int, default=None, help="Random seed.")
    parser.add_argument(
        "--dtype",
        type=str,
        default="bfloat16",
        choices=["float16", "float32", "bfloat16"],
        help="Torch dtype.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        choices=["auto", "cuda", "mps", "cpu"],
        help="Target device. 'auto' picks cuda -> mps -> cpu.",
    )
    parser.add_argument(
        "--no_cpu_offload",
        action="store_true",
        help="Disable model CPU offloading and move pipeline directly to --device.",
    )
    args = parser.parse_args()

    config = WanBackboneConfig(
        model_id=args.model_id,
        torch_dtype=args.dtype,
        device=args.device,
        enable_cpu_offload=not args.no_cpu_offload,
    )
    backbone = WanBackbone(config)

    print(f"[wan] loading model: {args.model_id}")
    backbone.load()

    print("[wan] generating clip...")
    frames = backbone.generate_clip(
        prompt=args.prompt,
        negative_prompt=args.negative_prompt,
        num_frames=args.num_frames,
        num_inference_steps=args.steps,
        guidance_scale=args.guidance_scale,
        height=args.height,
        width=args.width,
        seed=args.seed,
    )

    backbone.save_video(frames=frames, output_path=args.output, fps=args.fps)
    print(f"[wan] saved: {args.output}")


if __name__ == "__main__":
    main()
