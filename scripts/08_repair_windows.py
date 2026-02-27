import argparse
import json
import shutil
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from memory_module import VisionEmbedder, VisionEmbedderConfig
from video_backbone import WanBackbone, WanBackboneConfig


def _ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def _read_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _read_jsonl(path: Path) -> List[Dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _write_json(path: Path, obj: Dict[str, Any]) -> None:
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")


def _write_jsonl(path: Path, rows: List[Dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def _cosine_similarity(v1: Optional[Any], v2: Optional[Any]) -> Optional[float]:
    if v1 is None or v2 is None:
        return None
    import numpy as np

    arr1 = np.asarray(v1)
    arr2 = np.asarray(v2)
    denom = (np.linalg.norm(arr1) * np.linalg.norm(arr2)) + 1e-12
    return float(np.dot(arr1, arr2) / denom)


def _mean(values: List[Optional[float]]) -> Optional[float]:
    vals = [float(v) for v in values if v is not None]
    if not vals:
        return None
    return float(sum(vals) / len(vals))


def _resolve_clip_path(run_dir: Path, row: Dict[str, Any]) -> Path:
    raw = str(row.get("clip_path", "")).strip()
    if raw:
        p = Path(raw)
        if not p.is_absolute():
            p = PROJECT_ROOT / p
        if p.exists():
            return p
    idx = int(row["window_index"])
    fallback = run_dir / "clips" / f"window_{idx:03d}.mp4"
    if fallback.exists():
        return fallback
    raise FileNotFoundError(f"Clip missing for window {idx}")


def _load_video_frames(clip_path: Path) -> List[Any]:
    import imageio.v3 as iio

    frames = [frame for frame in iio.imiter(clip_path.as_posix())]
    if not frames:
        raise RuntimeError(f"No frames decoded from {clip_path}")
    return frames


def _compose_repair_prompt(
    base_prompt: str,
    beat: str,
    previous_beat: str,
    next_beat: str,
    repair_notes: str,
) -> str:
    parts = [
        base_prompt.strip(),
        f"Repair current beat only: {beat.strip()}",
    ]
    if previous_beat.strip():
        parts.append(f"Previous beat already completed: {previous_beat.strip()}")
    if next_beat.strip():
        parts.append(f"Do not jump to next beat yet: {next_beat.strip()}")
    if repair_notes.strip():
        parts.append(f"Repair constraints: {repair_notes.strip()}")
    parts.append(
        "Keep one continuous scene; no teleporting objects; no stones falling from the sky; "
        "preserve identity and layout from neighboring windows."
    )
    return " ".join(parts)


def _window_badness(row: Dict[str, Any], critic_threshold: float, transition_threshold: float) -> float:
    critic_score = float(row.get("selected_critic_score") or 0.0)
    transition = float(row.get("selected_transition_similarity") or 0.0)
    missing_critic = max(0.0, critic_threshold - critic_score)
    missing_transition = max(0.0, transition_threshold - transition)
    return missing_critic + missing_transition


def main() -> None:
    parser = argparse.ArgumentParser(description="Second-pass continuity repair for generated windows.")
    parser.add_argument("--run_dir", type=str, required=True, help="Run directory containing run_log.jsonl and clips/")
    parser.add_argument("--embedding_backend", type=str, default="dinov2", choices=["clip", "dinov2"])
    parser.add_argument("--embedding_model_id", type=str, default="", help="Embedder model path or id.")
    parser.add_argument("--video_model_id", type=str, default="", help="Override video model path or id.")
    parser.add_argument("--dtype", type=str, default="bfloat16", choices=["float16", "float32", "bfloat16"])
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cuda", "mps", "cpu"])
    parser.add_argument("--no_cpu_offload", action="store_true")
    parser.add_argument("--num_frames", type=int, default=96)
    parser.add_argument("--steps", type=int, default=30)
    parser.add_argument("--guidance_scale", type=float, default=6.0)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--width", type=int, default=832)
    parser.add_argument("--fps", type=int, default=12)
    parser.add_argument("--candidates", type=int, default=8)
    parser.add_argument("--attempts", type=int, default=3)
    parser.add_argument("--accept_score", type=float, default=0.84)
    parser.add_argument("--critic_threshold", type=float, default=0.80)
    parser.add_argument("--transition_threshold", type=float, default=0.82)
    parser.add_argument("--max_repairs", type=int, default=0, help="0 means repair all weak windows.")
    parser.add_argument("--replace_original", action="store_true", help="Overwrite original clips with repaired clips.")
    args = parser.parse_args()

    run_dir = Path(args.run_dir)
    log_path = run_dir / "run_log.jsonl"
    summary_path = run_dir / "run_summary.json"
    if not log_path.exists():
        raise FileNotFoundError(f"Missing run log: {log_path}")
    if not summary_path.exists():
        raise FileNotFoundError(f"Missing run summary: {summary_path}")

    summary = _read_json(summary_path)
    rows = _read_jsonl(log_path)
    rows = sorted(rows, key=lambda r: int(r["window_index"]))

    video_model_id = args.video_model_id or str(summary.get("video_model_id") or "")
    if not video_model_id:
        raise ValueError("video_model_id is required (missing in args and run_summary.json).")

    embedding_model_id = args.embedding_model_id
    if not embedding_model_id:
        local_default = PROJECT_ROOT / "models" / "dinov2-base"
        embedding_model_id = local_default.as_posix() if local_default.exists() else ""
    if not embedding_model_id:
        raise ValueError("embedding_model_id is required for repair scoring.")

    embedder = VisionEmbedder(
        VisionEmbedderConfig(
            backend=args.embedding_backend,
            model_id=embedding_model_id,
            device=args.device,
        )
    )
    embedder.load()

    backbone = WanBackbone(
        WanBackboneConfig(
            model_id=video_model_id,
            torch_dtype=args.dtype,
            device=args.device,
            enable_cpu_offload=not args.no_cpu_offload,
        )
    )
    backbone.load()

    clips_dir = run_dir / "clips"
    repaired_dir = run_dir / "clips_repaired"
    _ensure_dir(repaired_dir)

    current_clip_paths: Dict[int, Path] = {}
    for row in rows:
        idx = int(row["window_index"])
        current_clip_paths[idx] = _resolve_clip_path(run_dir, row)

    embed_cache: Dict[int, Dict[str, Any]] = {}

    def get_embeddings(idx: int) -> Dict[str, Any]:
        if idx in embed_cache:
            return embed_cache[idx]
        frames = _load_video_frames(current_clip_paths[idx])
        e = {
            "first": embedder.embed_first_frame(frames),
            "last": embedder.embed_last_frame(frames),
            "clip": embedder.embed_frames(frames),
        }
        embed_cache[idx] = e
        return e

    weak_rows: List[Tuple[float, Dict[str, Any]]] = []
    for row in rows:
        badness = _window_badness(row, args.critic_threshold, args.transition_threshold)
        if badness > 0:
            weak_rows.append((badness, row))
    weak_rows.sort(key=lambda t: t[0], reverse=True)
    if args.max_repairs > 0:
        weak_rows = weak_rows[: args.max_repairs]

    repaired_indices: List[int] = []
    for _, row in weak_rows:
        idx = int(row["window_index"])
        prev_idx = idx - 1
        next_idx = idx + 1

        prev_beat = rows[prev_idx]["beat"] if prev_idx >= 0 and prev_idx < len(rows) else ""
        next_beat = rows[next_idx]["beat"] if next_idx >= 0 and next_idx < len(rows) else ""
        base_prompt = str(row.get("generation_prompt") or row.get("refined_prompt") or row.get("beat"))
        initial_note = str(row.get("selected_critic_feedback") or "")

        best: Optional[Dict[str, Any]] = None
        repair_note = initial_note
        base_seed = int(row.get("seed") or (42 + idx))

        for attempt_idx in range(max(1, args.attempts)):
            prompt = _compose_repair_prompt(
                base_prompt=base_prompt,
                beat=str(row["beat"]),
                previous_beat=prev_beat,
                next_beat=next_beat,
                repair_notes=repair_note,
            )
            for candidate_idx in range(max(1, args.candidates)):
                seed = base_seed + candidate_idx + (attempt_idx * max(16, args.candidates))
                frames = backbone.generate_clip(
                    prompt=prompt,
                    negative_prompt=str(row.get("negative_prompt") or ""),
                    num_frames=args.num_frames,
                    num_inference_steps=args.steps,
                    guidance_scale=args.guidance_scale,
                    height=args.height,
                    width=args.width,
                    seed=seed,
                )
                cand_first = embedder.embed_first_frame(frames)
                cand_last = embedder.embed_last_frame(frames)
                cand_clip = embedder.embed_frames(frames)

                left_transition = None
                right_transition = None
                env_prev = None
                env_next = None
                if prev_idx >= 0 and prev_idx in current_clip_paths:
                    prev_e = get_embeddings(prev_idx)
                    left_transition = _cosine_similarity(cand_first, prev_e["last"])
                    env_prev = _cosine_similarity(cand_clip, prev_e["clip"])
                if next_idx < len(rows) and next_idx in current_clip_paths:
                    next_e = get_embeddings(next_idx)
                    right_transition = _cosine_similarity(cand_last, next_e["first"])
                    env_next = _cosine_similarity(cand_clip, next_e["clip"])

                boundary = _mean([left_transition, right_transition])
                env = _mean([env_prev, env_next])
                final = _mean([boundary, boundary, env])  # boundary gets higher weight
                final_score = float(final) if final is not None else 0.0

                feedback_parts: List[str] = []
                if left_transition is not None and left_transition < args.transition_threshold:
                    feedback_parts.append("Opening frame must match previous window ending composition.")
                if right_transition is not None and right_transition < args.transition_threshold:
                    feedback_parts.append("Ending frame must flow into next window opening composition.")
                if env is not None and env < args.critic_threshold:
                    feedback_parts.append("Preserve fixed courtyard layout, pot position, and object continuity.")
                if not feedback_parts:
                    feedback_parts.append("Continuity objective satisfied.")
                feedback = " ".join(feedback_parts)

                cand = {
                    "frames": frames,
                    "prompt": prompt,
                    "seed": seed,
                    "attempt_index": attempt_idx,
                    "candidate_index": candidate_idx,
                    "left_transition": left_transition,
                    "right_transition": right_transition,
                    "environment_similarity": env,
                    "repair_score": final_score,
                    "feedback": feedback,
                    "clip_embedding": cand_clip,
                    "first_embedding": cand_first,
                    "last_embedding": cand_last,
                }
                if best is None or cand["repair_score"] > best["repair_score"]:
                    best = cand

            if best is not None and best["repair_score"] >= float(args.accept_score):
                break
            if best is not None:
                repair_note = best["feedback"]

        if best is None:
            continue

        out_path = repaired_dir / f"window_{idx:03d}.mp4"
        backbone.save_video(best["frames"], out_path.as_posix(), fps=args.fps)
        repaired_indices.append(idx)
        current_clip_paths[idx] = out_path
        embed_cache[idx] = {
            "first": best["first_embedding"],
            "last": best["last_embedding"],
            "clip": best["clip_embedding"],
        }
        row["repair_pass"] = {
            "repaired": True,
            "repair_clip_path": out_path.as_posix(),
            "repair_score": best["repair_score"],
            "left_transition": best["left_transition"],
            "right_transition": best["right_transition"],
            "environment_similarity": best["environment_similarity"],
            "repair_feedback": best["feedback"],
            "repair_seed": best["seed"],
            "repair_attempt_index": best["attempt_index"],
            "repair_candidate_index": best["candidate_index"],
        }
        if args.replace_original:
            original = clips_dir / f"window_{idx:03d}.mp4"
            _ensure_dir(original.parent)
            shutil.copy2(out_path, original)

    repaired_log = run_dir / "run_log_repaired.jsonl"
    _write_jsonl(repaired_log, rows)
    repair_summary = {
        "run_dir": run_dir.as_posix(),
        "repaired_windows": repaired_indices,
        "num_repaired": len(repaired_indices),
        "clips_repaired_dir": repaired_dir.as_posix(),
        "repaired_log": repaired_log.as_posix(),
        "replace_original": args.replace_original,
        "accept_score": args.accept_score,
        "candidates": args.candidates,
        "attempts": args.attempts,
        "transition_threshold": args.transition_threshold,
        "critic_threshold": args.critic_threshold,
    }
    _write_json(run_dir / "repair_summary.json", repair_summary)
    print(f"[repair] repaired_windows={len(repaired_indices)}")
    print(f"[repair] log={repaired_log.as_posix()}")


if __name__ == "__main__":
    main()
