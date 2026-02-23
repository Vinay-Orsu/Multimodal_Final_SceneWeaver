import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

# Ensure project root is importable when running:
# `python scripts/run_story_pipeline.py ...`
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from director_llm import SceneDirector, SceneDirectorConfig
from memory_module import NarrativeMemory, VisionEmbedder, VisionEmbedderConfig
from video_backbone import WanBackbone, WanBackboneConfig


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def export_jsonl(rows: List[Dict[str, Any]], out_path: Path) -> None:
    with out_path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def maybe_init_embedder(
    backend: str,
    model_id: Optional[str],
    device: str,
) -> Optional[VisionEmbedder]:
    if backend == "none":
        return None
    embedder = VisionEmbedder(
        VisionEmbedderConfig(
            backend=backend,
            model_id=model_id,
            device=device,
        )
    )
    embedder.load()
    return embedder


def build_generation_prompt(
    refined_prompt: str,
    beat: str,
    style_prefix: str,
    character_lock: str,
) -> str:
    parts = []
    if style_prefix.strip():
        parts.append(style_prefix.strip())
    if character_lock.strip():
        parts.append(f"Character continuity: {character_lock.strip()}")
    parts.append(f"Current beat: {beat.strip()}")
    parts.append(f"Shot prompt: {refined_prompt.strip()}")
    parts.append(
        "Strictly follow the current beat and keep the same characters, identities, and scene context. "
        "No unrelated objects, no random scene changes."
    )
    return " ".join(parts)


def _compact_previous_prompt(prompt: str) -> str:
    if not prompt:
        return ""
    head = prompt.split(" Previous visual context:")[0].strip()
    return head[:240]


def _cosine_similarity(v1: Optional[Any], v2: Optional[Any]) -> Optional[float]:
    if v1 is None or v2 is None:
        return None
    import numpy as np

    arr1 = np.asarray(v1)
    arr2 = np.asarray(v2)
    denom = (np.linalg.norm(arr1) * np.linalg.norm(arr2)) + 1e-12
    return float(np.dot(arr1, arr2) / denom)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Storyline -> Scene Director -> Wan clip windows with local/global memory feedback."
    )
    parser.add_argument("--storyline", type=str, required=True, help="Full storyline or plot text.")
    parser.add_argument("--output_dir", type=str, default="outputs/story_run", help="Run output directory.")
    parser.add_argument("--total_minutes", type=float, default=0.5, help="Target video length in minutes.")
    parser.add_argument("--window_seconds", type=int, default=10, help="Seconds per clip window.")

    parser.add_argument("--director_model_id", type=str, default="", help="Optional HF LLM id for director.")
    parser.add_argument("--director_temperature", type=float, default=0.7, help="Director LLM temperature.")

    parser.add_argument("--video_model_id", type=str, default="Wan-AI/Wan2.0-T2V-14B", help="Wan model id.")
    parser.add_argument("--dtype", type=str, default="bfloat16", choices=["float16", "float32", "bfloat16"])
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cuda", "mps", "cpu"])
    parser.add_argument("--no_cpu_offload", action="store_true")
    parser.add_argument("--num_frames", type=int, default=49)
    parser.add_argument("--steps", type=int, default=30)
    parser.add_argument("--guidance_scale", type=float, default=6.0)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--width", type=int, default=832)
    parser.add_argument("--fps", type=int, default=8)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument(
        "--style_prefix",
        type=str,
        default="cinematic realistic, coherent motion, stable camera, high detail",
        help="Global style and quality prefix prepended to each generation prompt.",
    )
    parser.add_argument(
        "--character_lock",
        type=str,
        default=(
            "one rabbit and one tortoise only; keep same appearance, size, and colors across all windows; "
            "no extra animals or humans"
        ),
        help="Continuity constraints to keep subject identity stable across windows.",
    )
    parser.add_argument(
        "--negative_prompt",
        type=str,
        default=(
            "blurry, low quality, flicker, frame jitter, deformed anatomy, duplicate subjects, extra limbs, "
            "extra animals, wrong species, text, subtitles, watermark, logo, collage, split-screen, glitch"
        ),
        help="Negative prompt passed into the video generator.",
    )

    parser.add_argument("--embedding_backend", type=str, default="clip", choices=["none", "clip", "dinov2"])
    parser.add_argument("--embedding_model_id", type=str, default="", help="Optional embedder model id.")
    parser.add_argument(
        "--last_frame_memory",
        action="store_true",
        help="Use previous clip last-frame embedding to rank candidate next clips by first-frame continuity.",
    )
    parser.add_argument(
        "--continuity_candidates",
        type=int,
        default=1,
        help="Candidate clips per window when last-frame memory is available (higher is slower).",
    )
    parser.add_argument("--dry_run", action="store_true", help="Only plan/refine prompts. No video generation.")
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    clips_dir = out_dir / "clips"
    ensure_dir(out_dir)
    ensure_dir(clips_dir)

    director = SceneDirector(
        SceneDirectorConfig(
            model_id=args.director_model_id or None,
            temperature=args.director_temperature,
        ),
        window_seconds=args.window_seconds,
    )
    director.load()
    windows = director.plan_windows(storyline=args.storyline, total_minutes=args.total_minutes)

    backbone = None
    if not args.dry_run:
        backbone = WanBackbone(
            WanBackboneConfig(
                model_id=args.video_model_id,
                torch_dtype=args.dtype,
                device=args.device,
                enable_cpu_offload=not args.no_cpu_offload,
            )
        )
        backbone.load()

    embedder = None
    memory = None
    if args.embedding_backend != "none":
        embedder = maybe_init_embedder(
            backend=args.embedding_backend,
            model_id=args.embedding_model_id or None,
            device=args.device,
        )
        memory = NarrativeMemory()

    previous_prompt = ""
    memory_feedback = None
    previous_last_frame_embedding = None
    log_rows: List[Dict[str, Any]] = []

    for window in windows:
        refined_prompt = director.refine_prompt(
            storyline=args.storyline,
            window=window,
            previous_prompt=previous_prompt,
            memory_feedback=memory_feedback.to_dict() if memory_feedback else None,
        )
        generation_prompt = build_generation_prompt(
            refined_prompt=refined_prompt,
            beat=window.beat,
            style_prefix=args.style_prefix,
            character_lock=args.character_lock,
        )

        clip_path = clips_dir / f"window_{window.index:03d}.mp4"
        row: Dict[str, Any] = {
            "window_index": window.index,
            "time_range": [window.start_sec, window.end_sec],
            "beat": window.beat,
            "prompt_seed": window.prompt_seed,
            "refined_prompt": refined_prompt,
            "generation_prompt": generation_prompt,
            "negative_prompt": args.negative_prompt,
            "clip_path": clip_path.as_posix(),
            "generated": False,
            "memory_feedback": None,
        }

        if not args.dry_run:
            base_seed = None if args.seed is None else args.seed + window.index
            continuity_active = (
                embedder is not None
                and args.last_frame_memory
                and previous_last_frame_embedding is not None
                and args.continuity_candidates > 1
            )
            num_candidates = args.continuity_candidates if continuity_active else 1
            candidate_rows: List[Dict[str, Any]] = []
            best_frames = None
            best_seed = base_seed
            best_transition_similarity = None

            for candidate_idx in range(num_candidates):
                candidate_seed = None
                if base_seed is not None:
                    candidate_seed = base_seed + (candidate_idx * 100000)
                candidate_frames = backbone.generate_clip(
                    prompt=generation_prompt,
                    negative_prompt=args.negative_prompt,
                    num_frames=args.num_frames,
                    num_inference_steps=args.steps,
                    guidance_scale=args.guidance_scale,
                    height=args.height,
                    width=args.width,
                    seed=candidate_seed,
                )

                transition_similarity = None
                if embedder is not None and args.last_frame_memory and previous_last_frame_embedding is not None:
                    candidate_first_embedding = embedder.embed_first_frame(candidate_frames)
                    transition_similarity = _cosine_similarity(candidate_first_embedding, previous_last_frame_embedding)

                candidate_rows.append(
                    {
                        "candidate_index": candidate_idx,
                        "seed": candidate_seed,
                        "transition_similarity": transition_similarity,
                    }
                )

                if best_frames is None:
                    best_frames = candidate_frames
                    best_seed = candidate_seed
                    best_transition_similarity = transition_similarity
                elif transition_similarity is not None and (
                    best_transition_similarity is None or transition_similarity > best_transition_similarity
                ):
                    best_frames = candidate_frames
                    best_seed = candidate_seed
                    best_transition_similarity = transition_similarity

            frames = best_frames
            window_seed = best_seed
            backbone.save_video(frames=frames, output_path=clip_path.as_posix(), fps=args.fps)
            row["generated"] = True
            row["seed"] = window_seed
            row["continuity_candidates"] = num_candidates
            row["selected_transition_similarity"] = best_transition_similarity
            if len(candidate_rows) > 1:
                row["candidate_scores"] = candidate_rows

            if embedder is not None and memory is not None:
                embedding = embedder.embed_frames(frames)
                memory_feedback = memory.register_window(
                    window.index,
                    embedding,
                    transition_similarity=best_transition_similarity,
                )
                row["memory_feedback"] = memory_feedback.to_dict()
                if args.last_frame_memory:
                    previous_last_frame_embedding = embedder.embed_last_frame(frames)

        previous_prompt = _compact_previous_prompt(refined_prompt)
        log_rows.append(row)
        print(f"[scene {window.index:03d}] {window.start_sec}-{window.end_sec}s ready")

    export_jsonl(log_rows, out_dir / "run_log.jsonl")
    summary = {
        "storyline": args.storyline,
        "total_minutes": args.total_minutes,
        "window_seconds": args.window_seconds,
        "num_windows": len(windows),
        "dry_run": args.dry_run,
        "director_model_id": args.director_model_id or None,
        "video_model_id": None if args.dry_run else args.video_model_id,
        "embedding_backend": args.embedding_backend,
        "last_frame_memory": args.last_frame_memory,
        "continuity_candidates": args.continuity_candidates,
        "output_dir": out_dir.as_posix(),
        "run_log": (out_dir / "run_log.jsonl").as_posix(),
    }
    with (out_dir / "run_summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print(f"[done] windows: {len(windows)}")
    print(f"[done] logs: {(out_dir / 'run_log.jsonl').as_posix()}")


if __name__ == "__main__":
    main()
