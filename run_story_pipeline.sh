#!/bin/bash -l
#SBATCH --job-name=sceneweaver_full
#SBATCH --output=slurm_logs/sceneweaver_%j.out
#SBATCH --error=slurm_logs/sceneweaver_%j.err
#SBATCH --time=3:00:00
#SBATCH --partition=a100
#SBATCH --gres=gpu:a100:1

set -euo pipefail

# Defaults are portable; override with env vars if needed.
PROJECT_ROOT="${PROJECT_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)}"
ENV_PATH="${ENV_PATH:-}"
WAN_LOCAL_MODEL="${WAN_LOCAL_MODEL:-}"
CONDA_SH="${CONDA_SH:-/apps/python/3.12-conda/etc/profile.d/conda.sh}"
USE_MODULES="${USE_MODULES:-1}"
USE_OFFLINE_MODE="${USE_OFFLINE_MODE:-0}"
DEVICE="${DEVICE:-cuda}"
PYTHON_BIN="${PYTHON_BIN:-}"
DOWNLOAD_MODEL="${DOWNLOAD_MODEL:-0}"
MODEL_REPO="${MODEL_REPO:-Wan-AI/Wan2.2-TI2V-5B}"
MODEL_DIR="${MODEL_DIR:-${PROJECT_ROOT}/models/$(basename "${MODEL_REPO}")}"
# Keep this narrow to avoid full snapshot downloads.
MODEL_INCLUDE="${MODEL_INCLUDE:-*.safetensors *.json *.txt tokenizer* *.model}"

# Optional HPC module setup (kept off by default for local/pool PCs).
if [ "${USE_MODULES}" = "1" ] && command -v module >/dev/null 2>&1; then
  module purge
  module load python/3.12-conda
  module load cuda/12.4.1
fi

mkdir -p "${PROJECT_ROOT}/slurm_logs" "${PROJECT_ROOT}/outputs" "${PROJECT_ROOT}/.hf"
cd "${PROJECT_ROOT}"

# Optional conda activation.
if [ -n "${ENV_PATH}" ]; then
  if [ -f "${CONDA_SH}" ]; then
    # shellcheck disable=SC1090
    source "${CONDA_SH}"
  fi
  if ! command -v conda >/dev/null 2>&1; then
    echo "Conda command not found. Set CONDA_SH correctly or activate env before running."
    exit 1
  fi
  conda activate "${ENV_PATH}"
fi

export PYTHONUNBUFFERED=1
export HF_HOME="${HF_HOME:-${PROJECT_ROOT}/.hf}"

# Online mode is default for pool PCs; enable offline only if local model/cache exists.
if [ "${USE_OFFLINE_MODE}" = "1" ]; then
  export HF_HUB_OFFLINE=1
  export TRANSFORMERS_OFFLINE=1
  export DIFFUSERS_OFFLINE=1
fi

# Override any of these at runtime, e.g.:
# STORYLINE="..." TOTAL_MINUTES=2 bash run_story_pipeline.sh
STORYLINE="${STORYLINE:-A race starts between a rabbit and a tortoise, rabbit sprints early, then slows, tortoise steadily advances and wins.}"
TOTAL_MINUTES="${TOTAL_MINUTES:-5}"
WINDOW_SECONDS="${WINDOW_SECONDS:-10}"
VIDEO_MODEL_ID="${VIDEO_MODEL_ID:-${WAN_LOCAL_MODEL:-Wan-AI/Wan2.0-T2V-14B}}"
EMBEDDING_BACKEND="${EMBEDDING_BACKEND:-none}"
DRY_RUN="${DRY_RUN:-0}"
AUTO_FALLBACK_DRY_RUN="${AUTO_FALLBACK_DRY_RUN:-1}"
RUN_STAMP="$(date +%Y%m%d_%H%M%S)"
OUTPUT_DIR="${OUTPUT_DIR:-outputs/story_run_${RUN_STAMP}}"

if [ -z "${PYTHON_BIN}" ]; then
  if command -v python >/dev/null 2>&1; then
    PYTHON_BIN="python"
  elif command -v python3 >/dev/null 2>&1; then
    PYTHON_BIN="python3"
  else
    echo "No Python interpreter found on PATH (python/python3)."
    exit 1
  fi
fi

# Optional selective model download (disabled by default).
# Example:
# DOWNLOAD_MODEL=1 MODEL_REPO=tencent/HunyuanVideo-1.5 bash run_story_pipeline.sh
if [ "${DOWNLOAD_MODEL}" = "1" ]; then
  mkdir -p "${MODEL_DIR}"
  HF_DL=""
  if command -v hf >/dev/null 2>&1; then
    HF_DL="hf"
  elif command -v huggingface-cli >/dev/null 2>&1; then
    HF_DL="huggingface-cli"
  else
    echo "No HF CLI found. Install 'huggingface_hub[cli]' or set VIDEO_MODEL_ID to a local path."
    exit 1
  fi
  # shellcheck disable=SC2206
  INCLUDE_ARR=(${MODEL_INCLUDE})
  HF_INCLUDE_ARGS=()
  for pat in "${INCLUDE_ARR[@]}"; do
    HF_INCLUDE_ARGS+=(--include "${pat}")
  done
  "${HF_DL}" download "${MODEL_REPO}" --local-dir "${MODEL_DIR}" "${HF_INCLUDE_ARGS[@]}"
  VIDEO_MODEL_ID="${MODEL_DIR}"
fi

# Preflight for real generation runtime on local/pool machines.
if [ "${DRY_RUN}" = "0" ]; then
  if ! "${PYTHON_BIN}" -c "import diffusers" >/dev/null 2>&1; then
    if [ "${AUTO_FALLBACK_DRY_RUN}" = "1" ]; then
      echo "Real generation runtime is not healthy on this machine (diffusers import failed)."
      echo "Falling back to DRY_RUN=1. Set AUTO_FALLBACK_DRY_RUN=0 to disable this behavior."
      DRY_RUN="1"
    else
      echo "Real generation runtime is not healthy on this machine (diffusers import failed)."
      echo "Use DRY_RUN=1, or run on a CUDA node where diffusers/torch stack is stable."
      exit 1
    fi
  fi
fi

CMD=("${PYTHON_BIN}" scripts/run_story_pipeline.py
  --storyline "${STORYLINE}" \
  --total_minutes "${TOTAL_MINUTES}" \
  --window_seconds "${WINDOW_SECONDS}" \
  --video_model_id "${VIDEO_MODEL_ID}" \
  --embedding_backend "${EMBEDDING_BACKEND}" \
  --device "${DEVICE}" \
  --output_dir "${OUTPUT_DIR}")

if [ "${DRY_RUN}" = "1" ]; then
  CMD+=(--dry_run)
fi

echo "PROJECT_ROOT=${PROJECT_ROOT}"
echo "OUTPUT_DIR=${OUTPUT_DIR}"
echo "DEVICE=${DEVICE}"
echo "DRY_RUN=${DRY_RUN}"
echo "VIDEO_MODEL_ID=${VIDEO_MODEL_ID}"
echo "PYTHON_BIN=${PYTHON_BIN}"
echo "DOWNLOAD_MODEL=${DOWNLOAD_MODEL}"
echo "MODEL_REPO=${MODEL_REPO}"
echo "MODEL_DIR=${MODEL_DIR}"

"${CMD[@]}"
