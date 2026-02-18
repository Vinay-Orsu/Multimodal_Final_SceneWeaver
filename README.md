# SceneWeaver (WIP)

## Wan 2.0 Video Backbone

Current status: a first `video_backbone` implementation is available with a CLI generator.

### Files
- `video_backbone/wan_backbone.py`: Wan wrapper (`load`, `generate_clip`, `save_video`)
- `scripts/generate_wan_clip.py`: one-shot prompt-to-video test script

### Install
```bash
pip install torch diffusers transformers accelerate imageio
```

### Run
```bash
python scripts/generate_wan_clip.py \
  --prompt "A cinematic drone shot over snowy mountains at sunrise" \
  --model_id "Wan-AI/Wan2.0-T2V-14B" \
  --output outputs/wan_clip.mp4
```

### Apple Silicon note
- `Wan-AI/Wan2.0-T2V-14B` is generally too heavy for local MPS runs and may crash.
- For Mac local testing, use a smaller Wan checkpoint; keep 14B for CUDA GPU machines.

## Scene Director + Memory Loop

The architecture now includes:
- `director_llm/scene_director.py`: storyline -> scene windows -> prompt refinement
- `memory_module/embedding_memory.py`: CLIP/DINOv2 visual embeddings + local/global drift feedback
- `scripts/run_story_pipeline.py`: end-to-end orchestration per 10s window

### Dry run (prompt planning/refinement only)
```bash
python scripts/run_story_pipeline.py \
  --storyline "A race starts between a rabbit and a tortoise, rabbit sprints early, then slows, tortoise steadily advances and wins." \
  --total_minutes 0.5 \
  --window_seconds 10 \
  --embedding_backend none \
  --dry_run
```

### Full loop (cluster / CUDA)
```bash
python scripts/run_story_pipeline.py \
  --storyline "A race starts between a rabbit and a tortoise, rabbit sprints early, then slows, tortoise steadily advances and wins." \
  --total_minutes 5 \
  --window_seconds 10 \
  --video_model_id "Wan-AI/Wan2.0-T2V-14B" \
  --embedding_backend clip \
  --device cuda
```
