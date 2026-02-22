# SceneWeaver Story Pipeline

Active and supported runtime is the story pipeline script stack:
- `run_story_pipeline.sh`
- `scripts/run_story_pipeline.py`
- `director_llm/scene_director.py`
- `video_backbone/wan_backbone.py`
- `memory_module/embedding_memory.py`

## Install
```bash
pip install -r requirements.txt
```

## Quick Start (Cluster)
```bash
sbatch --partition=a40 --gres=gpu:a40:1 \
  --export=ALL,ENV_PATH=sceneweaver311,HF_HOME=$PWD/.hf,DRY_RUN=0,AUTO_FALLBACK_DRY_RUN=0 \
  run_story_pipeline.sh
```

## Architecture (Active Path)
1. Storyline is split into time windows by `SceneDirector`.
2. Each window prompt is refined with continuity context.
3. `WanBackbone` (or another diffusers T2V model id) generates frames.
4. Frames are encoded to per-window MP4 clips.
5. Optional memory embeddings (`clip` or `dinov2`) provide continuity feedback.

## Note on `src/driftguard`
`src/driftguard` is retained only as archived experimental code and is not a supported runtime path for current jobs.
