from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import List


def _latest_run_dir(runs_root: Path) -> Path:
    runs = sorted([p for p in runs_root.glob("*") if p.is_dir()])
    if not runs:
        raise RuntimeError(f"No runs found in {runs_root}")
    return runs[-1]


def _load_prompts(run_dir: Path) -> List[tuple[int, str]]:
    log_path = run_dir / "run_log.jsonl"
    if log_path.exists():
        rows = [json.loads(line) for line in log_path.read_text(encoding="utf-8").splitlines() if line.strip()]
        return [(int(r["window_index"]), str(r.get("prompt", ""))) for r in rows]

    clips_dir = run_dir / "clips"
    prompts = []
    for p in sorted(clips_dir.glob("window_*.txt")):
        idx = int(p.stem.split("_")[-1])
        prompts.append((idx, p.read_text(encoding="utf-8")))
    return prompts


def _draw_text_frame(text: str, t: int, width: int, height: int):
    from PIL import Image, ImageDraw

    # Animated solid background (avoids NumPy/OpenMP dependency issues)
    r = int(90 + 60 * math.sin(t / 10.0))
    g = int(110 + 70 * math.cos(t / 13.0))
    b = int(150 + 50 * math.sin(t / 17.0))
    img = Image.new("RGB", (width, height), (max(0, min(r, 255)), max(0, min(g, 255)), max(0, min(b, 255))))

    draw = ImageDraw.Draw(img)
    margin = 28
    box_h = int(height * 0.42)
    draw.rounded_rectangle(
        (margin, height - box_h - margin, width - margin, height - margin),
        radius=18,
        fill=(12, 14, 20),
    )
    content = text[:360]
    draw.multiline_text(
        (margin + 18, height - box_h - margin + 16),
        content,
        fill=(240, 240, 245),
        spacing=6,
    )
    return img


def _write_preview_clip(
    prompt: str,
    out_path: Path,
    seconds: float,
    fps: int,
    width: int,
    height: int,
) -> None:
    import imageio.v3 as iio

    frames = []
    n = max(1, int(seconds * fps))
    for t in range(n):
        frames.append(_draw_text_frame(prompt, t=t, width=width, height=height))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    iio.imwrite(out_path.as_posix(), frames, fps=fps)


def main() -> None:
    parser = argparse.ArgumentParser(description="Create visual preview clips from dry-run prompts.")
    parser.add_argument("--run_dir", type=str, default="", help="Specific run directory under outputs/runs.")
    parser.add_argument("--runs_root", type=str, default="outputs/runs", help="Runs root directory.")
    parser.add_argument("--seconds", type=float, default=4.0, help="Preview clip duration.")
    parser.add_argument("--fps", type=int, default=8, help="Preview fps.")
    parser.add_argument("--width", type=int, default=832, help="Preview width.")
    parser.add_argument("--height", type=int, default=480, help="Preview height.")
    args = parser.parse_args()

    runs_root = Path(args.runs_root)
    run_dir = Path(args.run_dir) if args.run_dir else _latest_run_dir(runs_root)
    preview_dir = run_dir / "preview_clips"
    prompts = _load_prompts(run_dir)
    if not prompts:
        raise RuntimeError(f"No prompt files found in {run_dir}")

    for idx, prompt in prompts:
        out_path = preview_dir / f"preview_{idx:03d}.mp4"
        _write_preview_clip(
            prompt=prompt,
            out_path=out_path,
            seconds=args.seconds,
            fps=args.fps,
            width=args.width,
            height=args.height,
        )
        print(out_path.as_posix())


if __name__ == "__main__":
    main()
