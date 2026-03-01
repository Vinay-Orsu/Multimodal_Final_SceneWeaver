#!/bin/bash -l
#SBATCH --job-name=sceneweaver_simple
#SBATCH --output=slurm_logs/sceneweaver_simple_%j.out
#SBATCH --error=slurm_logs/sceneweaver_simple_%j.err
#SBATCH --partition=a40
#SBATCH --gres=gpu:a40:1
#SBATCH --time=03:00:00
#SBATCH --cpus-per-task=16

set -euo pipefail

PROJECT_ROOT="/home/hpc/v123be/v123be37/Multimodal_Final_SceneWeaver"
DINO_MODEL_ID="${PROJECT_ROOT}/models/dinov2-base"
ADAPTER_CKPT="${PROJECT_ROOT}/outputs/pororo_continuity_adapter.pt"

STORYLINE="${STORYLINE:-A race starts between a rabbit and a tortoise, rabbit sprints early, then slows, tortoise steadily advances and wins.}"
TOTAL_MINUTES="${TOTAL_MINUTES:-1}"
WINDOW_SECONDS="${WINDOW_SECONDS:-10}"
CONTINUITY_CANDIDATES="${CONTINUITY_CANDIDATES:-2}"
RUN_STAMP="$(date +%Y%m%d_%H%M%S)"
OUTPUT_DIR="${OUTPUT_DIR:-${PROJECT_ROOT}/outputs/story_run_${RUN_STAMP}}"

source /apps/python/3.12-conda/etc/profile.d/conda.sh
conda activate sceneweaver311
export PYTHONNOUSERSITE=1
unset PYTHONPATH
export PYTHONUNBUFFERED=1
export HF_HOME="${HOME}/.cache/huggingface_${SLURM_JOB_ID}"
export HF_HUB_DISABLE_XET=1
export TRANSFORMERS_OFFLINE=1

cd "${PROJECT_ROOT}"
mkdir -p slurm_logs outputs "${HF_HOME}"

if [ ! -d "${DINO_MODEL_ID}" ]; then
  echo "Missing local DINO model dir: ${DINO_MODEL_ID}"
  exit 1
fi
if [ ! -f "${ADAPTER_CKPT}" ]; then
  echo "Missing adapter checkpoint: ${ADAPTER_CKPT}"
  exit 1
fi

echo "RUNNING ON $(hostname)"
echo "OUTPUT_DIR=${OUTPUT_DIR}"

python -s -u scripts/run_story_pipeline.py \
  --storyline "${STORYLINE}" \
  --output_dir "${OUTPUT_DIR}" \
  --total_minutes "${TOTAL_MINUTES}" \
  --window_seconds "${WINDOW_SECONDS}" \
  --device auto \
  --embedding_backend none \
  --last_frame_memory \
  --continuity_candidates "${CONTINUITY_CANDIDATES}" \
  --continuity_adapter_ckpt "${ADAPTER_CKPT}" \
  --continuity_adapter_model_id "${DINO_MODEL_ID}"

# Combine all generated windows to one preview video.
if command -v ffmpeg >/dev/null 2>&1 && [ -d "${OUTPUT_DIR}/clips" ] && compgen -G "${OUTPUT_DIR}/clips/window_*.mp4" >/dev/null; then
  CONCAT_LIST="${OUTPUT_DIR}/concat_windows.txt"
  : > "${CONCAT_LIST}"
  for clip in "${OUTPUT_DIR}"/clips/window_*.mp4; do
    printf "file '%s'\n" "$(realpath "${clip}")" >> "${CONCAT_LIST}"
  done
  if ! ffmpeg -y -f concat -safe 0 -i "${CONCAT_LIST}" -c copy "${OUTPUT_DIR}/story_windows_concat.mp4"; then
    ffmpeg -y -f concat -safe 0 -i "${CONCAT_LIST}" -c:v libx264 -preset veryfast -crf 18 -c:a aac "${OUTPUT_DIR}/story_windows_concat.mp4"
  fi
  echo "Combined continuity preview: ${OUTPUT_DIR}/story_windows_concat.mp4"
else
  echo "Skipping combine step (ffmpeg missing or no clips found)."
fi

