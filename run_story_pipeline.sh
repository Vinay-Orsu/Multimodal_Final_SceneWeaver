#!/bin/bash -l
#SBATCH --job-name=sceneweaver_full
#SBATCH --output=slurm_logs/sceneweaver_%j.out
#SBATCH --error=slurm_logs/sceneweaver_%j.err
#SBATCH --time=3:00:00
#SBATCH --partition=a100
#SBATCH --gres=gpu:a100:1

set -euo pipefail

PROJECT_ROOT="/home/hpc/v123be/v123be36/Multimodal_Final_SceneWeaver"
ENV_PATH="/home/hpc/v123be/v123be36/.conda/envs/lit_eval"
WAN_LOCAL_MODEL="/home/vault/v123be/v123be36/Wan2.2-TI2V-5B"

module purge
module load python/3.12-conda
module load cuda/12.4.1

mkdir -p "${PROJECT_ROOT}/slurm_logs"
mkdir -p "${PROJECT_ROOT}/outputs"
mkdir -p "${PROJECT_ROOT}/.hf"

source /apps/python/3.12-conda/etc/profile.d/conda.sh
if [ ! -d "${ENV_PATH}" ]; then
  echo "Conda env not found: ${ENV_PATH}"
  exit 1
fi
conda activate "${ENV_PATH}"

export PYTHONUNBUFFERED=1
export HF_HOME="${PROJECT_ROOT}/.hf"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export DIFFUSERS_OFFLINE=1

cd "${PROJECT_ROOT}"

# Override any of these when submitting, e.g.:
# sbatch --export=ALL,STORYLINE="your story",TOTAL_MINUTES=2 run_story_pipeline.slurm
STORYLINE="${STORYLINE:-A race starts between a rabbit and a tortoise, rabbit sprints early, then slows, tortoise steadily advances and wins.}"
TOTAL_MINUTES="${TOTAL_MINUTES:-5}"
WINDOW_SECONDS="${WINDOW_SECONDS:-10}"
VIDEO_MODEL_ID="${VIDEO_MODEL_ID:-${WAN_LOCAL_MODEL}}"
EMBEDDING_BACKEND="${EMBEDDING_BACKEND:-clip}"
RUN_STAMP="$(date +%Y%m%d_%H%M%S)"
OUTPUT_DIR="outputs/story_run_${RUN_STAMP}"

python scripts/run_story_pipeline.py \
  --storyline "${STORYLINE}" \
  --total_minutes "${TOTAL_MINUTES}" \
  --window_seconds "${WINDOW_SECONDS}" \
  --video_model_id "${VIDEO_MODEL_ID}" \
  --embedding_backend "${EMBEDDING_BACKEND}" \
  --device cuda \
  --output_dir "${OUTPUT_DIR}"
