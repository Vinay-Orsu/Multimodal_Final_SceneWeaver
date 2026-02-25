#!/bin/bash -l
#SBATCH --job-name=sceneweaver_full
#SBATCH --output=slurm_logs/sceneweaver_%j.out
#SBATCH --error=slurm_logs/sceneweaver_%j.err
#SBATCH --time=3:00:00
#SBATCH --partition=a40
#SBATCH --gres=gpu:a40:1
set -euo pipefail


module purge
module load python
cd /home/hpc/v123be/v123be36/Multimodal_Final_SceneWeaver/
# Some cluster profile scripts assume this exists; keep nounset-safe default.
export LD_LIBRARY_PATH="${LD_LIBRARY_PATH:-}"
# Defaults are portable; override with env vars if needed.
# Under Slurm, prefer the original submit directory instead of spool staging.
DEFAULT_PROJECT_ROOT="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)}"
PROJECT_ROOT="${PROJECT_ROOT:-${DEFAULT_PROJECT_ROOT}}"
ENV_PATH="${ENV_PATH:-}"
VENV_PATH="${VENV_PATH:-}"
DEFAULT_ENV_PATH="${DEFAULT_ENV_PATH:-sceneweaver_runtime}"
WAN_LOCAL_MODEL="${WAN_LOCAL_MODEL:-/home/vault/v123be/v123be36/Wan2.1-T2V-1.3B-Diffusers}"
CONDA_SH="${CONDA_SH:-/apps/python/3.12-conda/etc/profile.d/conda.sh}"
USE_MODULES="${USE_MODULES:-1}"
PYTHON_MODULE="${PYTHON_MODULE:-python/3.12-conda}"
CUDA_MODULE="${CUDA_MODULE:-cuda/12.4.1}"
USE_OFFLINE_MODE="${USE_OFFLINE_MODE:-1}"
DEVICE="${DEVICE:-auto}"
STRICT_DEVICE="${STRICT_DEVICE:-0}"
PYTHON_BIN="${PYTHON_BIN:-}"
DOWNLOAD_MODEL="${DOWNLOAD_MODEL:-0}"
MODEL_REPO="${MODEL_REPO:-Wan-AI/Wan2.1-T2V-1.3B-Diffusers}"
MODEL_DIR="${MODEL_DIR:-${PROJECT_ROOT}/models/$(basename "${MODEL_REPO}")}"
# Keep this narrow to avoid full snapshot downloads.
MODEL_INCLUDE="${MODEL_INCLUDE:-*.safetensors *.json *.txt tokenizer* *.model}"
DINOV2_LOCAL_MODEL="${DINOV2_LOCAL_MODEL:-/home/vault/v123be/v123be36/facebook/dinov2-base}"

# Optional HPC module setup (kept off by default for local/pool PCs).
if [ "${USE_MODULES}" = "1" ] && command -v module >/dev/null 2>&1; then
  if ! module purge; then
    echo "Warning: module purge failed; continuing."
  fi
  if [ -n "${PYTHON_MODULE}" ] && ! module load "${PYTHON_MODULE}"; then
    echo "Warning: could not load Python module '${PYTHON_MODULE}'."
  fi
  if [ -n "${CUDA_MODULE}" ] && ! module load "${CUDA_MODULE}"; then
    echo "Warning: could not load CUDA module '${CUDA_MODULE}'; continuing without module load."
    echo "Set CUDA_MODULE to a valid module name, or set USE_MODULES=0 to skip modules."
  fi
fi

mkdir -p "${PROJECT_ROOT}/slurm_logs" "${PROJECT_ROOT}/outputs" "${PROJECT_ROOT}/.hf"
cd "${PROJECT_ROOT}"

# Optional default env when neither ENV_PATH nor VENV_PATH is provided.
if [ -z "${ENV_PATH}" ] && [ -z "${VENV_PATH}" ] && [ -n "${DEFAULT_ENV_PATH}" ]; then
  ENV_PATH="${DEFAULT_ENV_PATH}"
fi

# Optional runtime activation.
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
elif [ -n "${VENV_PATH}" ]; then
  if [ ! -f "${VENV_PATH}/bin/activate" ]; then
    echo "Virtualenv activate script not found at ${VENV_PATH}/bin/activate"
    exit 1
  fi
  # shellcheck disable=SC1090
  source "${VENV_PATH}/bin/activate"
fi

export PYTHONUNBUFFERED=1
# Prevent ~/.local packages from shadowing the conda env on cluster nodes.
export PYTHONNOUSERSITE=1
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
if [ -n "${WAN_LOCAL_MODEL}" ]; then
  DEFAULT_VIDEO_MODEL_ID="${WAN_LOCAL_MODEL}"
elif [ -d "${MODEL_DIR}" ]; then
  DEFAULT_VIDEO_MODEL_ID="${MODEL_DIR}"
else
  DEFAULT_VIDEO_MODEL_ID="${MODEL_REPO}"
fi
VIDEO_MODEL_ID="${VIDEO_MODEL_ID:-${DEFAULT_VIDEO_MODEL_ID}}"
DIRECTOR_MODEL_ID="${DIRECTOR_MODEL_ID:-}"
DIRECTOR_TEMPERATURE="${DIRECTOR_TEMPERATURE:-0.3}"
EMBEDDING_BACKEND="${EMBEDDING_BACKEND:-dinov2}"
EMBEDDING_MODEL_ID="${EMBEDDING_MODEL_ID:-}"
EMBEDDING_ADAPTER_CKPT="${EMBEDDING_ADAPTER_CKPT:-}"
LAST_FRAME_MEMORY="${LAST_FRAME_MEMORY:-1}"
CONTINUITY_CANDIDATES="${CONTINUITY_CANDIDATES:-2}"
ENVIRONMENT_MEMORY="${ENVIRONMENT_MEMORY:-1}"
TRANSITION_WEIGHT="${TRANSITION_WEIGHT:-0.65}"
ENVIRONMENT_WEIGHT="${ENVIRONMENT_WEIGHT:-0.35}"
SCENE_CHANGE_ENV_DECAY="${SCENE_CHANGE_ENV_DECAY:-0.25}"
DRY_RUN="0"
AUTO_FALLBACK_DRY_RUN="${AUTO_FALLBACK_DRY_RUN:-0}"
STYLE_PREFIX="${STYLE_PREFIX:-cinematic realistic, coherent motion, stable camera, high detail}"
CHARACTER_LOCK="${CHARACTER_LOCK:-one rabbit and one tortoise only; keep same appearance, size, and colors across all windows; no extra animals or humans}"
NEGATIVE_PROMPT="${NEGATIVE_PROMPT:-blurry, low quality, flicker, frame jitter, deformed anatomy, duplicate subjects, extra limbs, extra animals, wrong species, text, subtitles, watermark, logo, collage, split-screen, glitch}"
NUM_FRAMES="${NUM_FRAMES:-49}"
STEPS="${STEPS:-35}"
GUIDANCE_SCALE="${GUIDANCE_SCALE:-12.0}"
HEIGHT="${HEIGHT:-480}"
WIDTH="${WIDTH:-832}"
FPS="${FPS:-8}"
SEED="${SEED:-42}"
RUN_STAMP="$(date +%Y%m%d_%H%M%S)"
OUTPUT_DIR="${OUTPUT_DIR:-outputs/story_run_${RUN_STAMP}}"

if [ "${EMBEDDING_BACKEND}" = "dinov2" ] && [ -z "${EMBEDDING_MODEL_ID}" ]; then
  EMBEDDING_MODEL_ID="${DINOV2_LOCAL_MODEL}"
fi

if [ -z "${EMBEDDING_ADAPTER_CKPT}" ] && [ -f "${PROJECT_ROOT}/outputs/pororo_continuity_adapter.pt" ]; then
  EMBEDDING_ADAPTER_CKPT="${PROJECT_ROOT}/outputs/pororo_continuity_adapter.pt"
fi

if [ -n "${EMBEDDING_ADAPTER_CKPT}" ] && [ ! -f "${EMBEDDING_ADAPTER_CKPT}" ]; then
  echo "EMBEDDING_ADAPTER_CKPT is set but file does not exist: ${EMBEDDING_ADAPTER_CKPT}"
  exit 1
fi

if [ "${USE_OFFLINE_MODE}" = "1" ] && [ "${EMBEDDING_BACKEND}" = "dinov2" ]; then
  if [ ! -f "${EMBEDDING_MODEL_ID}/preprocessor_config.json" ]; then
    echo "Offline dinov2 embedding requires a local path with preprocessor_config.json."
    echo "Current EMBEDDING_MODEL_ID='${EMBEDDING_MODEL_ID}' is not a valid local dinov2 directory."
    exit 1
  fi
fi

# If a local model directory is provided, ensure it looks like a diffusers pipeline.
if [ -d "${VIDEO_MODEL_ID}" ] && [ ! -f "${VIDEO_MODEL_ID}/model_index.json" ]; then
  echo "Local VIDEO_MODEL_ID is not a diffusers pipeline directory: ${VIDEO_MODEL_ID}"
  echo "Missing required file: ${VIDEO_MODEL_ID}/model_index.json"
  if [ -f "${VIDEO_MODEL_ID}/config.json" ] && grep -q '"_class_name"[[:space:]]*:[[:space:]]*"WanModel"' "${VIDEO_MODEL_ID}/config.json"; then
    echo "Detected native Wan checkpoint layout (WanModel config)."
    echo "Current SceneWeaver runtime expects diffusers pipeline format for --video_model_id."
  fi
  echo "Set VIDEO_MODEL_ID to a diffusers-formatted local model directory."
  exit 1
fi

# Current runtime is text-only. Reject TI2V checkpoints to avoid low-quality/noise outputs.
if echo "${VIDEO_MODEL_ID}" | grep -qi "TI2V"; then
  echo "VIDEO_MODEL_ID appears to be a TI2V model: ${VIDEO_MODEL_ID}"
  echo "This pipeline currently provides text-only prompts and no image conditioning input."
  echo "Use a T2V model (e.g., Wan2.1-T2V-1.3B-Diffusers) for stable scene generation."
  exit 1
fi

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
  if [ "${DEVICE}" = "cuda" ] && ! "${PYTHON_BIN}" -c "import torch; assert torch.cuda.is_available()" >/dev/null 2>&1; then
    echo "DEVICE=cuda requested, but torch.cuda.is_available() is false in this environment."
    if ! "${PYTHON_BIN}" -c "import torch; print('torch_version=' + str(torch.__version__)); print('torch_cuda_build=' + str(torch.version.cuda)); print('cuda_available=' + str(torch.cuda.is_available())); print('cuda_device_count=' + str(torch.cuda.device_count()))" 2>/dev/null; then
      echo "Torch diagnostics unavailable (torch import failed in the active environment)."
    fi
    echo "Likely cause: CPU-only torch build in the active env, or a CUDA/runtime mismatch."
    if [ "${STRICT_DEVICE}" = "1" ]; then
      echo "STRICT_DEVICE=1 set; exiting."
      exit 1
    fi
    echo "Falling back to DEVICE=auto."
    DEVICE="auto"
  fi
  if ! "${PYTHON_BIN}" -c "import diffusers; assert hasattr(diffusers, 'AutoPipelineForText2Video') or hasattr(diffusers, 'DiffusionPipeline')" >/dev/null 2>&1; then
    echo "Real generation runtime is not healthy on this machine (diffusers text2video pipeline unavailable)."
    echo "Suggested fix in env: pip install -U 'diffusers>=0.30' transformers accelerate"
    exit 1
  fi
fi

SCRIPT_HELP="$("${PYTHON_BIN}" scripts/run_story_pipeline.py --help 2>/dev/null || true)"
supports_flag() {
  echo "${SCRIPT_HELP}" | grep -q -- "$1"
}

CMD=("${PYTHON_BIN}" scripts/run_story_pipeline.py
  --storyline "${STORYLINE}" \
  --total_minutes "${TOTAL_MINUTES}" \
  --window_seconds "${WINDOW_SECONDS}" \
  --video_model_id "${VIDEO_MODEL_ID}" \
  --director_temperature "${DIRECTOR_TEMPERATURE}" \
  --embedding_backend "${EMBEDDING_BACKEND}" \
  --num_frames "${NUM_FRAMES}" \
  --steps "${STEPS}" \
  --guidance_scale "${GUIDANCE_SCALE}" \
  --height "${HEIGHT}" \
  --width "${WIDTH}" \
  --fps "${FPS}" \
  --seed "${SEED}" \
  --device "${DEVICE}" \
  --output_dir "${OUTPUT_DIR}")

if supports_flag "--style_prefix"; then
  CMD+=(--style_prefix "${STYLE_PREFIX}")
fi
if supports_flag "--character_lock"; then
  CMD+=(--character_lock "${CHARACTER_LOCK}")
fi
if supports_flag "--negative_prompt"; then
  CMD+=(--negative_prompt "${NEGATIVE_PROMPT}")
fi
if supports_flag "--embedding_model_id" && [ -n "${EMBEDDING_MODEL_ID}" ]; then
  CMD+=(--embedding_model_id "${EMBEDDING_MODEL_ID}")
fi
if supports_flag "--embedding_adapter_ckpt" && [ -n "${EMBEDDING_ADAPTER_CKPT}" ]; then
  CMD+=(--embedding_adapter_ckpt "${EMBEDDING_ADAPTER_CKPT}")
fi
if supports_flag "--continuity_candidates"; then
  CMD+=(--continuity_candidates "${CONTINUITY_CANDIDATES}")
fi
if supports_flag "--last_frame_memory" && [ "${LAST_FRAME_MEMORY}" = "1" ]; then
  CMD+=(--last_frame_memory)
fi
if supports_flag "--environment_memory"; then
  if [ "${ENVIRONMENT_MEMORY}" = "1" ]; then
    CMD+=(--environment_memory)
  else
    CMD+=(--no-environment_memory)
  fi
fi
if supports_flag "--transition_weight"; then
  CMD+=(--transition_weight "${TRANSITION_WEIGHT}")
fi
if supports_flag "--environment_weight"; then
  CMD+=(--environment_weight "${ENVIRONMENT_WEIGHT}")
fi
if supports_flag "--scene_change_env_decay"; then
  CMD+=(--scene_change_env_decay "${SCENE_CHANGE_ENV_DECAY}")
fi
if [ -n "${DIRECTOR_MODEL_ID}" ]; then
  CMD+=(--director_model_id "${DIRECTOR_MODEL_ID}")
fi

echo "PROJECT_ROOT=${PROJECT_ROOT}"
echo "OUTPUT_DIR=${OUTPUT_DIR}"
echo "DEVICE=${DEVICE}"
echo "DRY_RUN=${DRY_RUN}"
echo "VIDEO_MODEL_ID=${VIDEO_MODEL_ID}"
echo "DIRECTOR_MODEL_ID=${DIRECTOR_MODEL_ID}"
echo "DIRECTOR_TEMPERATURE=${DIRECTOR_TEMPERATURE}"
echo "PYTHON_BIN=${PYTHON_BIN}"
echo "DOWNLOAD_MODEL=${DOWNLOAD_MODEL}"
echo "MODEL_REPO=${MODEL_REPO}"
echo "MODEL_DIR=${MODEL_DIR}"
echo "STYLE_PREFIX=${STYLE_PREFIX}"
echo "CHARACTER_LOCK=${CHARACTER_LOCK}"
echo "NEGATIVE_PROMPT=${NEGATIVE_PROMPT}"
echo "EMBEDDING_BACKEND=${EMBEDDING_BACKEND}"
echo "EMBEDDING_MODEL_ID=${EMBEDDING_MODEL_ID}"
echo "EMBEDDING_ADAPTER_CKPT=${EMBEDDING_ADAPTER_CKPT}"
echo "LAST_FRAME_MEMORY=${LAST_FRAME_MEMORY}"
echo "CONTINUITY_CANDIDATES=${CONTINUITY_CANDIDATES}"
echo "ENVIRONMENT_MEMORY=${ENVIRONMENT_MEMORY}"
echo "TRANSITION_WEIGHT=${TRANSITION_WEIGHT}"
echo "ENVIRONMENT_WEIGHT=${ENVIRONMENT_WEIGHT}"
echo "SCENE_CHANGE_ENV_DECAY=${SCENE_CHANGE_ENV_DECAY}"
echo "NUM_FRAMES=${NUM_FRAMES}"
echo "STEPS=${STEPS}"
echo "GUIDANCE_SCALE=${GUIDANCE_SCALE}"
echo "HEIGHT=${HEIGHT}"
echo "WIDTH=${WIDTH}"
echo "FPS=${FPS}"
echo "SEED=${SEED}"

"${CMD[@]}"
