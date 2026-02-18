from __future__ import annotations

import datetime as dt
from pathlib import Path
from typing import Any, Dict, List, Optional

from driftguard.generation.seeds import seed_for_window
from driftguard.generation.video import VideoGenConfig, VideoGenerator
from driftguard.generation.windowing import overlap_meta
from driftguard.planning.canon import build_canon
from driftguard.planning.prompts import build_window_prompt
from driftguard.planning.storyboard import make_storyboard
from driftguard.refine.budget import RegenBudget
from driftguard.refine.loop import combined_score, repair_constraints_from_scores, should_accept
from driftguard.retrieval.chunk import chunk_text
from driftguard.retrieval.clean import normalize_text
from driftguard.retrieval.fetch import FetchConfig, fetch_texts
from driftguard.retrieval.index import SimpleIndex
from driftguard.utils.io import ensure_dir, merge_dict, read_text, write_json, write_jsonl
from driftguard.utils.logging import get_logger


def _maybe_init_scene_director(cfg: Dict[str, Any], window_seconds: int):
    plan_cfg = cfg.get("planning", {})
    if not plan_cfg.get("use_director_model", False):
        return None
    model_id = plan_cfg.get("director_model_id", "")
    if not model_id:
        return None
    try:
        from director_llm import SceneDirector, SceneDirectorConfig
    except Exception:
        return None
    director = SceneDirector(
        SceneDirectorConfig(
            model_id=model_id,
            temperature=float(plan_cfg.get("director_temperature", 0.7)),
        ),
        window_seconds=window_seconds,
    )
    director.load()
    return director


def _maybe_init_memory(cfg: Dict[str, Any], dry_run: bool):
    mem_cfg = cfg.get("memory", {})
    if (not mem_cfg.get("enabled", False)) or dry_run:
        return None, None
    try:
        from memory_module import NarrativeMemory, VisionEmbedder, VisionEmbedderConfig
    except Exception:
        return None, None
    embedder = VisionEmbedder(
        VisionEmbedderConfig(
            backend=mem_cfg.get("embedding_backend", "clip"),
            model_id=mem_cfg.get("embedding_model_id") or None,
            device=cfg.get("models", {}).get("video", {}).get("device", "auto"),
        )
    )
    embedder.load()
    memory = NarrativeMemory(
        local_threshold=float(mem_cfg.get("local_threshold", 0.25)),
        global_threshold=float(mem_cfg.get("global_threshold", 0.20)),
    )
    return embedder, memory


def load_config(base_cfg: Path, overlays: Optional[List[Path]] = None) -> Dict[str, Any]:
    from driftguard.utils.io import load_yaml

    cfg = load_yaml(base_cfg)
    for p in overlays or []:
        cfg = merge_dict(cfg, load_yaml(p))
    return cfg


def run_pipeline(
    cfg: Dict[str, Any],
    *,
    storyline: Optional[str] = None,
    storyline_file: Optional[Path] = None,
    dry_run_override: Optional[bool] = None,
) -> Path:
    logger = get_logger("driftguard")

    if storyline_file and not storyline:
        storyline = read_text(storyline_file)
    if not storyline:
        raise ValueError("storyline is required via --storyline or --storyline_file")
    storyline = storyline.strip()

    run_name = cfg.get("run", {}).get("name", "run")
    run_id = f"{run_name}_{dt.datetime.now(dt.timezone.utc).strftime('%Y%m%dT%H%M%SZ')}"
    out_root = Path(cfg.get("run", {}).get("output_root", "outputs/runs"))
    run_dir = out_root / run_id
    ensure_dir(run_dir)
    clips_dir = run_dir / "clips"
    ensure_dir(clips_dir)

    retrieval_cfg = cfg.get("retrieval", {})
    chunks = []
    if retrieval_cfg.get("enabled", False):
        urls = retrieval_cfg.get("urls", [])
        docs = fetch_texts(
            urls=urls,
            config=FetchConfig(
                whitelist_domains=retrieval_cfg.get("whitelist_domains", []),
                timeout_sec=20,
            ),
        )
        for i, doc in enumerate(docs):
            clean = normalize_text(doc)
            chunks.extend(
                chunk_text(
                    text=clean,
                    source=f"url{i}",
                    chunk_size_chars=int(retrieval_cfg.get("chunk_size_chars", 600)),
                    chunk_overlap_chars=int(retrieval_cfg.get("chunk_overlap_chars", 120)),
                )
            )
    index = SimpleIndex.build(chunks)
    retrieved = [c for c, _ in index.search(storyline, top_k=5)]
    canon = build_canon(storyline=storyline, retrieved_chunks=retrieved)

    story_cfg = cfg.get("story", {})
    windows = make_storyboard(
        storyline=storyline,
        total_minutes=float(story_cfg.get("total_minutes", 0.5)),
        window_seconds=int(story_cfg.get("window_seconds", 10)),
    )
    director = _maybe_init_scene_director(cfg, window_seconds=int(story_cfg.get("window_seconds", 10)))
    ov_seconds = int(story_cfg.get("overlap_seconds", 2))

    gen_cfg = cfg.get("generation", {})
    dry_run = bool(gen_cfg.get("dry_run", True)) if dry_run_override is None else dry_run_override
    generator = VideoGenerator(
        VideoGenConfig(
            model_id=cfg.get("models", {}).get("video", {}).get("model_id", "Wan-AI/Wan2.0-T2V-14B"),
            dtype=cfg.get("models", {}).get("video", {}).get("dtype", "bfloat16"),
            device=cfg.get("models", {}).get("video", {}).get("device", "auto"),
            cpu_offload=bool(cfg.get("models", {}).get("video", {}).get("cpu_offload", True)),
            fps=int(gen_cfg.get("fps", 8)),
            num_frames=int(gen_cfg.get("num_frames", 49)),
            num_steps=int(gen_cfg.get("num_steps", 30)),
            guidance_scale=float(gen_cfg.get("guidance_scale", 6.0)),
            height=int(gen_cfg.get("height", 480)),
            width=int(gen_cfg.get("width", 832)),
        ),
        dry_run=dry_run,
    )
    # If we are doing a real run (not dry-run), backend must be loaded.
    if not dry_run:
        generator.load()
    embedder, memory = _maybe_init_memory(cfg, dry_run=dry_run)

    weights = cfg.get("critics", {})
    acceptance = float(weights.get("acceptance_score", 0.55))
    budget = RegenBudget(max_regens_per_window=int(gen_cfg.get("max_regens_per_window", 1)))

    log_rows = []
    previous_prompt = ""
    for w in windows:
        accepted = False
        best = None
        repair_constraints = None
        for regen_try in budget.attempts():
            if director is not None:
                prompt = director.refine_prompt(
                    storyline=storyline,
                    window=w,
                    previous_prompt=previous_prompt,
                    memory_feedback=None,
                )
                if repair_constraints:
                    prompt = f"{prompt} Repair constraints: {repair_constraints}"
            else:
                prompt = build_window_prompt(
                    window=w,
                    canon=canon,
                    previous_prompt=previous_prompt,
                    repair_constraints=repair_constraints,
                )
            out_path = clips_dir / f"window_{w.index:03d}.mp4"
            seed = seed_for_window(int(gen_cfg.get("seed", 42)), w.index, regen_try=regen_try)
            generated, _, gen_meta = generator.generate(prompt=prompt, output_path=out_path, seed=seed)

            # Basic working critic: prompt/canon scores + fixed drift proxy.
            drift_proxy = 0.5 if w.index == 0 else 0.65
            score, details = combined_score(
                prompt=prompt,
                previous_prompt=previous_prompt,
                canon=canon,
                drift_score=drift_proxy,
                weights=weights,
            )
            accepted = should_accept(score, acceptance)
            row = {
                "window_index": w.index,
                "time_range": [w.start_sec, w.end_sec],
                "beat": w.beat,
                "prompt": prompt,
                "overlap": overlap_meta(w, ov_seconds),
                "seed": seed,
                "regen_try": regen_try,
                "generated": generated,
                "gen_meta": gen_meta,
                "critic_scores": details,
                "accepted": accepted,
            }
            if (embedder is not None) and (memory is not None) and generated and (not dry_run):
                try:
                    # generation backend may expose frames in future versions.
                    frames = None
                    if isinstance(gen_meta, dict):
                        frames = gen_meta.get("frames")
                    if frames is not None:
                        emb = embedder.embed_frames(frames)
                        mem_feedback = memory.register_window(w.index, emb).to_dict()
                        row["memory_feedback"] = mem_feedback
                except Exception as exc:
                    row["memory_feedback_error"] = str(exc)
            best = row
            if accepted:
                break
            repair_constraints = repair_constraints_from_scores(details)

        log_rows.append(best)
        previous_prompt = best["prompt"]
        logger.info(
            "window=%03d accepted=%s score=%.4f",
            w.index,
            best["accepted"],
            best["critic_scores"]["final_score"],
        )

    summary = {
        "run_id": run_id,
        "storyline": storyline,
        "num_windows": len(windows),
        "dry_run": dry_run,
        "output_dir": run_dir.as_posix(),
    }
    write_json(run_dir / "run_summary.json", summary)
    write_json(run_dir / "canon.json", canon)
    write_jsonl(run_dir / "run_log.jsonl", log_rows)
    return run_dir
