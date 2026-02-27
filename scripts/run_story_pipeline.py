import argparse
import json
import re
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
from memory_module.window_critic import evaluate_candidate
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
    adapter_ckpt: Optional[str],
    device: str,
) -> Optional[VisionEmbedder]:
    if backend == "none":
        return None
    # Backward-compatible with older VisionEmbedderConfig versions that
    # do not expose adapter_ckpt yet.
    try:
        cfg = VisionEmbedderConfig(
            backend=backend,
            model_id=model_id,
            adapter_ckpt=adapter_ckpt,
            device=device,
        )
    except TypeError:
        cfg = VisionEmbedderConfig(
            backend=backend,
            model_id=model_id,
            device=device,
        )
    embedder = VisionEmbedder(cfg)
    embedder.load()
    return embedder


def build_generation_prompt(
    refined_prompt: str,
    beat: str,
    style_prefix: str,
    character_lock: str,
    environment_anchor: str,
    scene_change_requested: bool,
    story_state_hint: str,
    repair_hint: str = "",
) -> str:
    parts = []
    if style_prefix.strip():
        parts.append(style_prefix.strip())
    if character_lock.strip():
        parts.append(f"Character continuity: {character_lock.strip()}")
    if environment_anchor.strip():
        if scene_change_requested:
            parts.append(f"Previous environment anchor (transition smoothly): {environment_anchor.strip()}")
        else:
            parts.append(f"Environment continuity anchor: {environment_anchor.strip()}")
    parts.append(f"Current beat: {beat.strip()}")
    if story_state_hint.strip():
        parts.append(f"Story state: {story_state_hint.strip()}")
    parts.append(f"Shot prompt: {refined_prompt.strip()}")
    if scene_change_requested:
        parts.append("Beat suggests a setting change; transition from the previous clip naturally, not abruptly.")
    elif environment_anchor.strip():
        parts.append(
            "Preserve location, background layout, lighting, weather, time-of-day, and camera viewpoint from the environment anchor."
        )
    parts.append(
        "Strictly follow the current beat and keep the same characters, identities, and scene context. "
        "No unrelated objects, no random scene changes."
    )
    if repair_hint.strip():
        parts.append(f"Critic repair constraints: {repair_hint.strip()}")
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


def _extract_environment_anchor(prompt: str) -> str:
    if not prompt:
        return ""
    parts = re.split(r"[.;]", prompt)
    env_keywords = (
        "location",
        "setting",
        "environment",
        "background",
        "scene",
        "room",
        "house",
        "street",
        "forest",
        "field",
        "park",
        "beach",
        "mountain",
        "indoor",
        "outdoor",
        "interior",
        "exterior",
        "sky",
        "rain",
        "snow",
        "fog",
        "sunset",
        "night",
        "daylight",
        "lighting",
        "camera",
    )
    selected: List[str] = []
    for part in parts:
        candidate = " ".join(part.strip().split())
        if not candidate:
            continue
        lower = candidate.lower()
        if any(token in lower for token in env_keywords):
            selected.append(candidate)
        if len(selected) >= 2:
            break
    if selected:
        return "; ".join(selected)[:260]
    fallback = " ".join(prompt.strip().split())
    return fallback[:220]


def _beat_requests_scene_change(beat: str) -> bool:
    text = (beat or "").lower()
    hints = (
        "new location",
        "cut to",
        "arrive",
        "arrives",
        "enter",
        "enters",
        "exit",
        "leave",
        "leaves",
        "move to",
        "moves to",
        "travel",
        "travels",
        "inside",
        "outside",
        "indoors",
        "outdoors",
        "back at",
    )
    return any(token in text for token in hints)


def _combine_continuity_score(
    transition_similarity: Optional[float],
    environment_similarity: Optional[float],
    transition_weight: float,
    environment_weight: float,
) -> Optional[float]:
    weighted_components = []
    if transition_similarity is not None and transition_weight > 0.0:
        weighted_components.append((transition_similarity, transition_weight))
    if environment_similarity is not None and environment_weight > 0.0:
        weighted_components.append((environment_similarity, environment_weight))
    if not weighted_components:
        return None
    total_weight = sum(weight for _, weight in weighted_components)
    if total_weight <= 0.0:
        return None
    weighted_sum = sum(score * weight for score, weight in weighted_components)
    return float(weighted_sum / total_weight)


def _build_story_state_hint(windows: List[Any], pos: int) -> str:
    previous_beat = windows[pos - 1].beat if pos > 0 else ""
    current_beat = windows[pos].beat
    next_beat = windows[pos + 1].beat if pos + 1 < len(windows) else ""
    future_beat = windows[pos + 2].beat if pos + 2 < len(windows) else ""

    hints: List[str] = []
    if previous_beat:
        hints.append(f"Completed previous beat: {previous_beat}.")
    hints.append(f"Required now: {current_beat}.")
    if next_beat:
        hints.append(f"Next beat after this window: {next_beat}.")
    if future_beat:
        hints.append(f"Do not jump ahead to later beat yet: {future_beat}.")
    return " ".join(hints)


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
        "--seed_strategy",
        type=str,
        default="fixed",
        choices=["fixed", "window_offset"],
        help="Seed scheduling across windows. 'fixed' keeps base seed constant for continuity.",
    )
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
            "one crow only; keep the same crow appearance, size, and colors across all windows; "
            "keep one clay pot and small stones consistent; no extra animals or humans"
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
        "--embedding_adapter_ckpt",
        type=str,
        default="",
        help="Optional continuity adapter checkpoint (.pt) for the visual embedder.",
    )
    parser.add_argument(
        "--last_frame_memory",
        action="store_true",
        help="Use previous clip last-frame embedding to rank candidate next clips by first-frame continuity.",
    )
    parser.add_argument(
        "--continuity_candidates",
        type=int,
        default=1,
        help="Candidate clips per window when continuity ranking is enabled (higher is slower).",
    )
    parser.add_argument(
        "--environment_memory",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Use previous clip embedding to preserve environment across windows.",
    )
    parser.add_argument(
        "--transition_weight",
        type=float,
        default=0.65,
        help="Weight for first-frame to previous-last-frame similarity in candidate ranking.",
    )
    parser.add_argument(
        "--environment_weight",
        type=float,
        default=0.35,
        help="Weight for whole-clip environment similarity in candidate ranking.",
    )
    parser.add_argument(
        "--scene_change_env_decay",
        type=float,
        default=0.25,
        help="Multiplier for environment_weight when beat indicates a location/setting change.",
    )
    parser.add_argument(
        "--continuity_min_score",
        type=float,
        default=0.72,
        help="Minimum critic score required to accept a window candidate.",
    )
    parser.add_argument(
        "--continuity_regen_attempts",
        type=int,
        default=2,
        help="Max candidate-generation attempts per window when critic score is below threshold.",
    )
    parser.add_argument(
        "--critic_story_weight",
        type=float,
        default=0.15,
        help="Weight for story progression score in critic final score.",
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
            adapter_ckpt=args.embedding_adapter_ckpt or None,
            device=args.device,
        )
        memory = NarrativeMemory()

    previous_prompt = ""
    memory_feedback = None
    previous_last_frame_embedding = None
    previous_clip_embedding = None
    previous_environment_anchor = ""
    log_rows: List[Dict[str, Any]] = []

    for window_pos, window in enumerate(windows):
        scene_change_requested = _beat_requests_scene_change(window.beat)
        story_state_hint = _build_story_state_hint(windows, window_pos)
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
            environment_anchor=previous_environment_anchor,
            scene_change_requested=scene_change_requested,
            story_state_hint=story_state_hint,
            repair_hint="",
        )

        clip_path = clips_dir / f"window_{window.index:03d}.mp4"
        row: Dict[str, Any] = {
            "window_index": window.index,
            "time_range": [window.start_sec, window.end_sec],
            "beat": window.beat,
            "prompt_seed": window.prompt_seed,
            "refined_prompt": refined_prompt,
            "generation_prompt": generation_prompt,
            "scene_change_requested": scene_change_requested,
            "environment_anchor": previous_environment_anchor,
            "negative_prompt": args.negative_prompt,
            "clip_path": clip_path.as_posix(),
            "generated": False,
            "memory_feedback": None,
        }

        if not args.dry_run:
            if args.seed is None:
                base_seed = None
            elif args.seed_strategy == "window_offset":
                base_seed = args.seed + window.index
            else:
                base_seed = args.seed
            transition_ref_available = (
                embedder is not None and args.last_frame_memory and previous_last_frame_embedding is not None
            )
            environment_ref_available = (
                embedder is not None and args.environment_memory and previous_clip_embedding is not None
            )
            continuity_active = (
                embedder is not None
                and args.continuity_candidates > 1
                and (transition_ref_available or environment_ref_available)
            )
            num_candidates = args.continuity_candidates if continuity_active else 1
            max_attempts = max(1, int(args.continuity_regen_attempts))
            candidate_rows: List[Dict[str, Any]] = []
            best_overall: Optional[Dict[str, Any]] = None
            selected: Optional[Dict[str, Any]] = None

            transition_weight = max(0.0, float(args.transition_weight))
            environment_weight = max(0.0, float(args.environment_weight))
            if scene_change_requested:
                scene_change_decay = max(0.0, float(args.scene_change_env_decay))
                environment_weight *= scene_change_decay
            previous_beat = windows[window_pos - 1].beat if window_pos > 0 else ""
            repair_hint = ""
            for attempt_idx in range(max_attempts):
                generation_prompt = build_generation_prompt(
                    refined_prompt=refined_prompt,
                    beat=window.beat,
                    style_prefix=args.style_prefix,
                    character_lock=args.character_lock,
                    environment_anchor=previous_environment_anchor,
                    scene_change_requested=scene_change_requested,
                    story_state_hint=story_state_hint,
                    repair_hint=repair_hint,
                )

                best_attempt: Optional[Dict[str, Any]] = None
                for candidate_idx in range(num_candidates):
                    candidate_seed = None
                    if base_seed is not None:
                        candidate_seed = base_seed + candidate_idx + (attempt_idx * max(32, num_candidates))
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
                        transition_similarity = _cosine_similarity(
                            candidate_first_embedding,
                            previous_last_frame_embedding,
                        )

                    environment_similarity = None
                    candidate_embedding = None
                    if embedder is not None and args.environment_memory and previous_clip_embedding is not None:
                        candidate_embedding = embedder.embed_frames(candidate_frames)
                        environment_similarity = _cosine_similarity(candidate_embedding, previous_clip_embedding)

                    continuity_score = _combine_continuity_score(
                        transition_similarity=transition_similarity,
                        environment_similarity=environment_similarity,
                        transition_weight=transition_weight,
                        environment_weight=environment_weight,
                    )
                    critic = evaluate_candidate(
                        current_beat=window.beat,
                        previous_beat=previous_beat,
                        transition_similarity=transition_similarity,
                        environment_similarity=environment_similarity,
                        continuity_score=continuity_score,
                        story_weight=float(args.critic_story_weight),
                        continuity_weight=max(0.0, 1.0 - float(args.critic_story_weight)),
                        attempt_index=attempt_idx,
                    )

                    candidate_entry = {
                        "attempt_index": attempt_idx,
                        "candidate_index": candidate_idx,
                        "seed": candidate_seed,
                        "transition_similarity": transition_similarity,
                        "environment_similarity": environment_similarity,
                        "continuity_score": continuity_score,
                        "critic_score": critic.final_score,
                        "critic_story_progress_score": critic.story_progress_score,
                        "critic_feedback": critic.feedback,
                    }
                    candidate_rows.append(candidate_entry)

                    candidate_state = {
                        "frames": candidate_frames,
                        "seed": candidate_seed,
                        "transition_similarity": transition_similarity,
                        "environment_similarity": environment_similarity,
                        "continuity_score": continuity_score,
                        "clip_embedding": candidate_embedding,
                        "critic_score": critic.final_score,
                        "critic_feedback": critic.feedback,
                        "generation_prompt": generation_prompt,
                        "attempt_index": attempt_idx,
                        "candidate_index": candidate_idx,
                    }
                    if best_attempt is None or candidate_state["critic_score"] > best_attempt["critic_score"]:
                        best_attempt = candidate_state

                if best_attempt is None:
                    continue
                if best_overall is None or best_attempt["critic_score"] > best_overall["critic_score"]:
                    best_overall = best_attempt
                if best_attempt["critic_score"] >= float(args.continuity_min_score):
                    selected = best_attempt
                    break
                repair_hint = best_attempt["critic_feedback"]
                refined_prompt = director.refine_prompt(
                    storyline=args.storyline,
                    window=window,
                    previous_prompt=previous_prompt,
                    memory_feedback={
                        "suggested_constraints": repair_hint,
                    },
                )

            selected = selected or best_overall
            if selected is None:
                raise RuntimeError(f"No candidate generated for window {window.index}")

            frames = selected["frames"]
            window_seed = selected["seed"]
            backbone.save_video(frames=frames, output_path=clip_path.as_posix(), fps=args.fps)
            row["generated"] = True
            row["seed"] = window_seed
            row["continuity_candidates"] = num_candidates
            row["selected_transition_similarity"] = selected["transition_similarity"]
            row["selected_environment_similarity"] = selected["environment_similarity"]
            row["selected_continuity_score"] = selected["continuity_score"]
            row["selected_critic_score"] = selected["critic_score"]
            row["selected_critic_feedback"] = selected["critic_feedback"]
            row["generation_prompt"] = selected["generation_prompt"]
            row["selected_attempt_index"] = selected["attempt_index"]
            row["continuity_min_score"] = args.continuity_min_score
            row["continuity_regen_attempts"] = max_attempts
            if len(candidate_rows) > 1:
                row["candidate_scores"] = candidate_rows

            if embedder is not None and memory is not None:
                best_clip_embedding = selected["clip_embedding"]
                embedding = best_clip_embedding if best_clip_embedding is not None else embedder.embed_frames(frames)
                memory_feedback = memory.register_window(
                    window.index,
                    embedding,
                    transition_similarity=selected["transition_similarity"],
                )
                memory_feedback.suggested_constraints = (
                    f"{memory_feedback.suggested_constraints} {selected['critic_feedback']}".strip()
                )
                row["memory_feedback"] = memory_feedback.to_dict()
                if args.last_frame_memory:
                    previous_last_frame_embedding = embedder.embed_last_frame(frames)
                if args.environment_memory:
                    previous_clip_embedding = embedding

        previous_prompt = _compact_previous_prompt(refined_prompt)
        next_environment_anchor = _extract_environment_anchor(refined_prompt)
        if next_environment_anchor:
            previous_environment_anchor = next_environment_anchor
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
        "embedding_adapter_ckpt": args.embedding_adapter_ckpt or None,
        "last_frame_memory": args.last_frame_memory,
        "continuity_candidates": args.continuity_candidates,
        "environment_memory": args.environment_memory,
        "transition_weight": args.transition_weight,
        "environment_weight": args.environment_weight,
        "scene_change_env_decay": args.scene_change_env_decay,
        "seed": args.seed,
        "seed_strategy": args.seed_strategy,
        "continuity_min_score": args.continuity_min_score,
        "continuity_regen_attempts": args.continuity_regen_attempts,
        "critic_story_weight": args.critic_story_weight,
        "output_dir": out_dir.as_posix(),
        "run_log": (out_dir / "run_log.jsonl").as_posix(),
    }
    with (out_dir / "run_summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print(f"[done] windows: {len(windows)}")
    print(f"[done] logs: {(out_dir / 'run_log.jsonl').as_posix()}")


if __name__ == "__main__":
    main()
