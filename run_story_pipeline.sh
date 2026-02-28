#!/bin/bash -l
#SBATCH --job-name=sceneweaver_full
#SBATCH --output=slurm_logs/sceneweaver_%j.out
#SBATCH --error=slurm_logs/sceneweaver_%j.err
#SBATCH --time=1:50:00
#SBATCH --partition=a100
#SBATCH --gres=gpu:a100:2
#SBATCH --cpus-per-task=16

set -euo pipefail

# Some cluster profile scripts assume this exists; keep nounset-safe default.
export LD_LIBRARY_PATH="${LD_LIBRARY_PATH:-}"

# Defaults are portable; override with env vars if needed.
DEFAULT_PROJECT_ROOT="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)}"
PROJECT_ROOT="${PROJECT_ROOT:-${DEFAULT_PROJECT_ROOT}}"

ENV_PATH="${ENV_PATH:-}"
VENV_PATH="${VENV_PATH:-}"
DEFAULT_ENV_PATH="${DEFAULT_ENV_PATH:-sceneweaver311}"

WAN_LOCAL_MODEL="${WAN_LOCAL_MODEL:-${PROJECT_ROOT}/models/Wan2.1-T2V-1.3B-Diffusers}"

CONDA_SH="${CONDA_SH:-/apps/python/3.12-conda/etc/profile.d/conda.sh}"
USE_MODULES="${USE_MODULES:-0}"
PYTHON_MODULE="${PYTHON_MODULE:-python/3.12-conda}"
CUDA_MODULE="${CUDA_MODULE:-cuda/12.4.1}"

USE_OFFLINE_MODE="${USE_OFFLINE_MODE:-1}"
DEVICE="${DEVICE:-cuda}"
STRICT_DEVICE="${STRICT_DEVICE:-0}"
PYTHON_BIN="${PYTHON_BIN:-}"

DOWNLOAD_MODEL="${DOWNLOAD_MODEL:-0}"
MODEL_REPO="${MODEL_REPO:-Wan-AI/Wan2.1-T2V-1.3B-Diffusers}"
MODEL_DIR="${MODEL_DIR:-${PROJECT_ROOT}/models/$(basename "${MODEL_REPO}")}"
MODEL_INCLUDE="${MODEL_INCLUDE:-*.safetensors *.json *.txt tokenizer* *.model}"
DOWNLOAD_DIRECTOR_MODEL="${DOWNLOAD_DIRECTOR_MODEL:-0}"
DIRECTOR_MODEL_REPO="${DIRECTOR_MODEL_REPO:-Qwen/Qwen2.5-3B-Instruct}"
DIRECTOR_MODEL_DIR="${DIRECTOR_MODEL_DIR:-${PROJECT_ROOT}/models/$(basename "${DIRECTOR_MODEL_REPO}")}"
DIRECTOR_MODEL_INCLUDE="${DIRECTOR_MODEL_INCLUDE:-*.safetensors *.json *.txt tokenizer* *.model}"

DINOV2_LOCAL_MODEL="${DINOV2_LOCAL_MODEL:-${PROJECT_ROOT}/models/dinov2-base}"

# Optional HPC module setup.
if [ "${USE_MODULES}" = "1" ] && command -v module >/dev/null 2>&1; then
  module purge || true
  [ -n "${PYTHON_MODULE}" ] && module load "${PYTHON_MODULE}" || true
  [ -n "${CUDA_MODULE}" ] && module load "${CUDA_MODULE}" || true
fi

mkdir -p "${PROJECT_ROOT}/slurm_logs" "${PROJECT_ROOT}/outputs" "${PROJECT_ROOT}/.hf"
cd "${PROJECT_ROOT}"

# If an env is already active in the current shell, keep it by default.
if [ -n "${CONDA_DEFAULT_ENV:-}" ] && [ -z "${ENV_PATH}" ] && [ -z "${VENV_PATH}" ]; then
  ENV_PATH="${CONDA_DEFAULT_ENV}"
fi

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
  command -v conda >/dev/null 2>&1 || { echo "Conda command not found. Set CONDA_SH correctly or activate env before running."; exit 1; }
  conda activate "${ENV_PATH}"
elif [ -n "${VENV_PATH}" ]; then
  [ -f "${VENV_PATH}/bin/activate" ] || { echo "Virtualenv activate script not found at ${VENV_PATH}/bin/activate"; exit 1; }
  # shellcheck disable=SC1090
  source "${VENV_PATH}/bin/activate"
fi

export PYTHONUNBUFFERED=1
export PYTHONNOUSERSITE=1
export HF_HOME="${HF_HOME:-${PROJECT_ROOT}/.hf}"

if [ "${USE_OFFLINE_MODE}" = "1" ] && { [ "${DOWNLOAD_MODEL}" = "1" ] || [ "${DOWNLOAD_DIRECTOR_MODEL}" = "1" ]; }; then
  echo "USE_OFFLINE_MODE=1 conflicts with model download. Switching to online mode for this run."
  USE_OFFLINE_MODE="0"
fi

if [ "${USE_OFFLINE_MODE}" = "1" ]; then
  export HF_HUB_OFFLINE=1
  export TRANSFORMERS_OFFLINE=1
  export DIFFUSERS_OFFLINE=1
fi

# STORYLINE
if [ -z "${STORYLINE:-}" ]; then
  STORYLINE="$(cat <<'STORY_EOF'
A thirsty crow finds a pot with very little water. The crow carries small stones one by one and drops them into the pot. As more stones fall in, the water level rises gradually. At last, the crow can reach the water and drinks.
STORY_EOF
)"
fi

if [ -z "${STYLE_PREFIX:-}" ]; then
  STYLE_PREFIX="$(cat <<'STYLE_EOF'
cinematic realistic drama, natural skin tones, stable camera, coherent motion, soft depth of field, high detail, grounded everyday realism
STYLE_EOF
)"
fi

if [ -z "${CHARACTER_LOCK:-}" ]; then
  CHARACTER_LOCK="$(cat <<'CHAR_EOF'
exactly one crow with consistent appearance across all windows: glossy black feathers, medium size, sharp beak, bright eyes. Keep one clay pot and small pebbles/stones consistent in shape and color. No extra animals, no humans, no crowds, no duplicate or cloned crows.
CHAR_EOF
)"
fi

if [ -z "${GLOBAL_CONTINUITY_ANCHOR:-}" ]; then
  GLOBAL_CONTINUITY_ANCHOR="$(cat <<'ANCHOR_EOF'
All windows occur in the same outdoor courtyard. Anchor objects must stay: one clay pot on a small wooden table, scattered pebbles near the table, one tree branch where the crow perches, warm daylight. Keep identical courtyard layout, object positions, and lighting progression. No random location changes, no teleporting objects, no sky-falling stones.
ANCHOR_EOF
)"
fi

if [ -z "${NEGATIVE_PROMPT:-}" ]; then
  NEGATIVE_PROMPT="$(cat <<'NEG_EOF'
blurry, low quality, flicker, frame jitter, deformed anatomy, extra limbs, duplicate people, cloned faces, twin characters, repeated character in frame, crowd, background people, strangers, extra humans, extra children, extra women, extra men, inconsistent face, identity drift, face morph, different person, different outfit, wardrobe change, different hairstyle, different location, different room layout, different furniture, inconsistent background, scene change, teleport, background swap, text overlay, subtitles, watermark, logo, collage, split-screen, glitch
NEG_EOF
)"
fi

CAPTIONER_MODEL_ID="${CAPTIONER_MODEL_ID:-}"
CAPTIONER_DEVICE="${CAPTIONER_DEVICE:-cpu}"
CAPTIONER_STUB_FALLBACK="${CAPTIONER_STUB_FALLBACK:-1}"

# Track which knobs were explicitly provided by user/environment so presets
# only override unset values.
HAS_FPS="${FPS+x}"
HAS_NUM_FRAMES="${NUM_FRAMES+x}"
HAS_STEPS="${STEPS+x}"
HAS_CONTINUITY_CANDIDATES="${CONTINUITY_CANDIDATES+x}"
HAS_CONTINUITY_REGEN_ATTEMPTS="${CONTINUITY_REGEN_ATTEMPTS+x}"
HAS_CONTINUITY_MIN_SCORE="${CONTINUITY_MIN_SCORE+x}"
HAS_RUN_REPAIR_PASS="${RUN_REPAIR_PASS+x}"
HAS_REPAIR_CANDIDATES="${REPAIR_CANDIDATES+x}"
HAS_REPAIR_ATTEMPTS="${REPAIR_ATTEMPTS+x}"
HAS_PARALLEL_WINDOW_MODE="${PARALLEL_WINDOW_MODE+x}"

TOTAL_MINUTES="${TOTAL_MINUTES:-1}"
FPS="${FPS:-12}"
WINDOW_SECONDS="${WINDOW_SECONDS:-8}"
NUM_FRAMES="${NUM_FRAMES:-96}"
STEPS="${STEPS:-30}"
GUIDANCE_SCALE="${GUIDANCE_SCALE:-6.0}"
HEIGHT="${HEIGHT:-480}"
WIDTH="${WIDTH:-832}"
SEED="${SEED:-42}"
SEED_STRATEGY="${SEED_STRATEGY:-fixed}"

if [ -z "${DIRECTOR_MODEL_ID:-}" ] && [ -d "${DIRECTOR_MODEL_DIR}" ]; then
  DIRECTOR_MODEL_ID="${DIRECTOR_MODEL_DIR}"
else
  DIRECTOR_MODEL_ID="${DIRECTOR_MODEL_ID:-}"
fi
DIRECTOR_TEMPERATURE="${DIRECTOR_TEMPERATURE:-0.05}"

EMBEDDING_BACKEND="${EMBEDDING_BACKEND:-dinov2}"
EMBEDDING_MODEL_ID="${EMBEDDING_MODEL_ID:-${DINOV2_LOCAL_MODEL}}"
LAST_FRAME_MEMORY="${LAST_FRAME_MEMORY:-1}"
CONTINUITY_CANDIDATES="${CONTINUITY_CANDIDATES:-8}"
CONTINUITY_MIN_SCORE="${CONTINUITY_MIN_SCORE:-0.72}"
CONTINUITY_REGEN_ATTEMPTS="${CONTINUITY_REGEN_ATTEMPTS:-2}"
CRITIC_STORY_WEIGHT="${CRITIC_STORY_WEIGHT:-0.15}"
ENVIRONMENT_MEMORY="${ENVIRONMENT_MEMORY:-1}"
TRANSITION_WEIGHT="${TRANSITION_WEIGHT:-0.65}"
ENVIRONMENT_WEIGHT="${ENVIRONMENT_WEIGHT:-0.35}"
SCENE_CHANGE_ENV_DECAY="${SCENE_CHANGE_ENV_DECAY:-0.25}"

EMBEDDING_ADAPTER_CKPT="${EMBEDDING_ADAPTER_CKPT:-${PROJECT_ROOT}/outputs/pororo_continuity_adapter.pt}"

DRY_RUN="${DRY_RUN:-0}"
RUN_STAMP="$(date +%Y%m%d_%H%M%S)"
OUTPUT_DIR="${OUTPUT_DIR:-outputs/story_run_${RUN_STAMP}}"
COMBINE_WINDOWS="${COMBINE_WINDOWS:-0}"
COMBINED_VIDEO_NAME="${COMBINED_VIDEO_NAME:-story_windows_concat.mp4}"
RUN_REPAIR_PASS="${RUN_REPAIR_PASS:-0}"
REPAIR_REPLACE_ORIGINAL="${REPAIR_REPLACE_ORIGINAL:-1}"
REPAIR_CANDIDATES="${REPAIR_CANDIDATES:-8}"
REPAIR_ATTEMPTS="${REPAIR_ATTEMPTS:-3}"
REPAIR_ACCEPT_SCORE="${REPAIR_ACCEPT_SCORE:-0.86}"
REPAIR_TRANSITION_THRESHOLD="${REPAIR_TRANSITION_THRESHOLD:-0.84}"
REPAIR_CRITIC_THRESHOLD="${REPAIR_CRITIC_THRESHOLD:-0.82}"
RUN_PRESET="${RUN_PRESET:-quality}"   # quality|fast
WINDOW_SHARD_COUNT="${WINDOW_SHARD_COUNT:-1}"
WINDOW_SHARD_INDEX="${WINDOW_SHARD_INDEX:-0}"
PARALLEL_WINDOW_MODE="${PARALLEL_WINDOW_MODE:-0}"
PARALLEL_GPUS_PER_NODE="${PARALLEL_GPUS_PER_NODE:-0}"

if [ "${RUN_PRESET}" = "fast" ]; then
  [ -z "${HAS_FPS}" ] && FPS="8"
  [ -z "${HAS_NUM_FRAMES}" ] && NUM_FRAMES="64"
  [ -z "${HAS_STEPS}" ] && STEPS="16"
  [ -z "${HAS_CONTINUITY_CANDIDATES}" ] && CONTINUITY_CANDIDATES="2"
  [ -z "${HAS_CONTINUITY_REGEN_ATTEMPTS}" ] && CONTINUITY_REGEN_ATTEMPTS="1"
  [ -z "${HAS_CONTINUITY_MIN_SCORE}" ] && CONTINUITY_MIN_SCORE="0.72"
  [ -z "${HAS_RUN_REPAIR_PASS}" ] && RUN_REPAIR_PASS="0"
  [ -z "${HAS_REPAIR_CANDIDATES}" ] && REPAIR_CANDIDATES="2"
  [ -z "${HAS_REPAIR_ATTEMPTS}" ] && REPAIR_ATTEMPTS="1"
  [ -z "${HAS_PARALLEL_WINDOW_MODE}" ] && PARALLEL_WINDOW_MODE="1"
fi

# Auto-bind shard index for SLURM array jobs when not explicitly set.
if [ "${WINDOW_SHARD_COUNT}" -gt 1 ] && [ "${WINDOW_SHARD_INDEX}" = "0" ] && [ -n "${SLURM_ARRAY_TASK_ID:-}" ]; then
  WINDOW_SHARD_INDEX="${SLURM_ARRAY_TASK_ID}"
fi

# Select model id
if [ -n "${WAN_LOCAL_MODEL}" ]; then
  DEFAULT_VIDEO_MODEL_ID="${WAN_LOCAL_MODEL}"
elif [ -d "${MODEL_DIR}" ]; then
  DEFAULT_VIDEO_MODEL_ID="${MODEL_DIR}"
else
  DEFAULT_VIDEO_MODEL_ID="${MODEL_REPO}"
fi
VIDEO_MODEL_ID="${VIDEO_MODEL_ID:-${DEFAULT_VIDEO_MODEL_ID}}"

if [ "${USE_OFFLINE_MODE}" = "1" ] && [ "${EMBEDDING_BACKEND}" = "dinov2" ]; then
  if [ ! -f "${EMBEDDING_MODEL_ID}/preprocessor_config.json" ]; then
    echo "Offline dinov2 embedding requires a local path with preprocessor_config.json."
    echo "Current EMBEDDING_MODEL_ID='${EMBEDDING_MODEL_ID}' is not a valid local dinov2 directory."
    exit 1
  fi
fi

if [ -d "${VIDEO_MODEL_ID}" ] && [ ! -f "${VIDEO_MODEL_ID}/model_index.json" ]; then
  echo "Local VIDEO_MODEL_ID is not a diffusers pipeline directory: ${VIDEO_MODEL_ID}"
  echo "Missing required file: ${VIDEO_MODEL_ID}/model_index.json"
  exit 1
fi

if echo "${VIDEO_MODEL_ID}" | grep -qi "TI2V"; then
  echo "VIDEO_MODEL_ID appears to be a TI2V model: ${VIDEO_MODEL_ID}"
  echo "Use a T2V model for text-only generation."
  exit 1
fi

# fail fast if adapter path is set but missing
if [ -n "${EMBEDDING_ADAPTER_CKPT}" ] && [ ! -f "${EMBEDDING_ADAPTER_CKPT}" ]; then
  echo "ERROR: embedding adapter checkpoint not found: ${EMBEDDING_ADAPTER_CKPT}"
  exit 1
fi

if [ -z "${PYTHON_BIN}" ]; then
  if command -v python >/dev/null 2>&1; then
    PYTHON_BIN="python"
  elif command -v python3 >/dev/null 2>&1; then
    PYTHON_BIN="python3"
  else
    echo "No python/python3 found."
    exit 1
  fi
fi

# Optional selective model download.
if [ "${DOWNLOAD_MODEL}" = "1" ] || [ "${DOWNLOAD_DIRECTOR_MODEL}" = "1" ]; then
  HF_DL=""
  if command -v hf >/dev/null 2>&1; then
    HF_DL="hf"
  elif command -v huggingface-cli >/dev/null 2>&1; then
    HF_DL="huggingface-cli"
  else
    echo "No HF CLI found. Install huggingface_hub[cli] or set model paths to local directories."
    exit 1
  fi
fi

if [ "${DOWNLOAD_MODEL}" = "1" ]; then
  mkdir -p "${MODEL_DIR}"
  # shellcheck disable=SC2206
  INCLUDE_ARR=(${MODEL_INCLUDE})
  HF_INCLUDE_ARGS=()
  for pat in "${INCLUDE_ARR[@]}"; do
    HF_INCLUDE_ARGS+=(--include "${pat}")
  done
  "${HF_DL}" download "${MODEL_REPO}" --local-dir "${MODEL_DIR}" "${HF_INCLUDE_ARGS[@]}"
  VIDEO_MODEL_ID="${MODEL_DIR}"
fi

if [ "${DOWNLOAD_DIRECTOR_MODEL}" = "1" ]; then
  mkdir -p "${DIRECTOR_MODEL_DIR}"
  # shellcheck disable=SC2206
  D_INCLUDE_ARR=(${DIRECTOR_MODEL_INCLUDE})
  D_HF_INCLUDE_ARGS=()
  for pat in "${D_INCLUDE_ARR[@]}"; do
    D_HF_INCLUDE_ARGS+=(--include "${pat}")
  done
  "${HF_DL}" download "${DIRECTOR_MODEL_REPO}" --local-dir "${DIRECTOR_MODEL_DIR}" "${D_HF_INCLUDE_ARGS[@]}"
  DIRECTOR_MODEL_ID="${DIRECTOR_MODEL_DIR}"
fi

if [ "${DRY_RUN}" = "0" ]; then
  if [ "${DEVICE}" = "cuda" ] && ! "${PYTHON_BIN}" -c "import torch; assert torch.cuda.is_available()" >/dev/null 2>&1; then
    echo "DEVICE=cuda requested, but torch.cuda.is_available() is false in this environment."
    if [ "${STRICT_DEVICE}" = "1" ]; then
      echo "STRICT_DEVICE=1 set; exiting."
      exit 1
    fi
    echo "Falling back to DEVICE=auto."
    DEVICE="auto"
  fi
fi

SCRIPT_HELP="$("${PYTHON_BIN}" scripts/run_story_pipeline.py --help 2>/dev/null || true)"
supports_flag() {
  echo "${SCRIPT_HELP}" | grep -q -- "$1"
}

# Inject global continuity anchor into style context when supported.
STYLE_PREFIX_COMBINED="${STYLE_PREFIX} Environment anchor: ${GLOBAL_CONTINUITY_ANCHOR}"

CMD=("${PYTHON_BIN}" scripts/run_story_pipeline.py
  --storyline "${STORYLINE}"
  --total_minutes "${TOTAL_MINUTES}"
  --window_seconds "${WINDOW_SECONDS}"
  --video_model_id "${VIDEO_MODEL_ID}"
  --director_temperature "${DIRECTOR_TEMPERATURE}"
  --embedding_backend "${EMBEDDING_BACKEND}"
  --embedding_model_id "${EMBEDDING_MODEL_ID}"
  --num_frames "${NUM_FRAMES}"
  --steps "${STEPS}"
  --guidance_scale "${GUIDANCE_SCALE}"
  --height "${HEIGHT}"
  --width "${WIDTH}"
  --fps "${FPS}"
  --seed "${SEED}"
  --seed_strategy "${SEED_STRATEGY}"
  --device "${DEVICE}"
  --output_dir "${OUTPUT_DIR}"
)

if supports_flag "--style_prefix"; then
  CMD+=(--style_prefix "${STYLE_PREFIX_COMBINED}")
fi
if supports_flag "--character_lock"; then
  CMD+=(--character_lock "${CHARACTER_LOCK}")
fi
if supports_flag "--negative_prompt"; then
  CMD+=(--negative_prompt "${NEGATIVE_PROMPT}")
fi
if supports_flag "--embedding_adapter_ckpt" && [ -n "${EMBEDDING_ADAPTER_CKPT}" ]; then
  CMD+=(--embedding_adapter_ckpt "${EMBEDDING_ADAPTER_CKPT}")
fi
if supports_flag "--last_frame_memory" && [ "${LAST_FRAME_MEMORY}" = "1" ]; then
  CMD+=(--last_frame_memory)
fi
if supports_flag "--continuity_candidates"; then
  CMD+=(--continuity_candidates "${CONTINUITY_CANDIDATES}")
fi
if supports_flag "--continuity_min_score"; then
  CMD+=(--continuity_min_score "${CONTINUITY_MIN_SCORE}")
fi
if supports_flag "--continuity_regen_attempts"; then
  CMD+=(--continuity_regen_attempts "${CONTINUITY_REGEN_ATTEMPTS}")
fi
if supports_flag "--critic_story_weight"; then
  CMD+=(--critic_story_weight "${CRITIC_STORY_WEIGHT}")
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
if supports_flag "--window_shard_count"; then
  CMD+=(--window_shard_count "${WINDOW_SHARD_COUNT}")
fi
if supports_flag "--window_shard_index"; then
  CMD+=(--window_shard_index "${WINDOW_SHARD_INDEX}")
fi
if supports_flag "--parallel_window_mode"; then
  if [ "${PARALLEL_WINDOW_MODE}" = "1" ]; then
    CMD+=(--parallel_window_mode)
  else
    CMD+=(--no-parallel_window_mode)
  fi
fi
if supports_flag "--captioner_model_id" && [ -n "${CAPTIONER_MODEL_ID}" ]; then
  CMD+=(--captioner_model_id "${CAPTIONER_MODEL_ID}")
  CMD+=(--captioner_device "${CAPTIONER_DEVICE}")
  if [ "${CAPTIONER_STUB_FALLBACK}" = "0" ]; then
    CMD+=(--no-captioner_stub_fallback)
  fi
fi
if [ -n "${DIRECTOR_MODEL_ID}" ]; then
  CMD+=(--director_model_id "${DIRECTOR_MODEL_ID}")
fi
if [ "${DRY_RUN}" = "1" ]; then
  CMD+=(--dry_run)
fi

echo "PROJECT_ROOT=${PROJECT_ROOT}"
echo "OUTPUT_DIR=${OUTPUT_DIR}"
echo "DEVICE=${DEVICE}"
echo "VIDEO_MODEL_ID=${VIDEO_MODEL_ID}"
echo "FPS=${FPS}"
echo "NUM_FRAMES=${NUM_FRAMES}"
echo "WINDOW_SECONDS=${WINDOW_SECONDS}"
echo "SEED=${SEED}"
echo "SEED_STRATEGY=${SEED_STRATEGY}"
echo "DIRECTOR_TEMPERATURE=${DIRECTOR_TEMPERATURE}"
echo "EMBEDDING_BACKEND=${EMBEDDING_BACKEND}"
echo "EMBEDDING_MODEL_ID=${EMBEDDING_MODEL_ID}"
echo "LAST_FRAME_MEMORY=${LAST_FRAME_MEMORY}"
echo "CONTINUITY_CANDIDATES=${CONTINUITY_CANDIDATES}"
echo "CONTINUITY_MIN_SCORE=${CONTINUITY_MIN_SCORE}"
echo "CONTINUITY_REGEN_ATTEMPTS=${CONTINUITY_REGEN_ATTEMPTS}"
echo "CRITIC_STORY_WEIGHT=${CRITIC_STORY_WEIGHT}"
echo "EMBEDDING_ADAPTER_CKPT=${EMBEDDING_ADAPTER_CKPT}"
echo "RUN_PRESET=${RUN_PRESET}"
echo "WINDOW_SHARD_COUNT=${WINDOW_SHARD_COUNT}"
echo "WINDOW_SHARD_INDEX=${WINDOW_SHARD_INDEX}"
echo "PARALLEL_WINDOW_MODE=${PARALLEL_WINDOW_MODE}"
echo "PARALLEL_GPUS_PER_NODE=${PARALLEL_GPUS_PER_NODE}"
echo "COMBINE_WINDOWS=${COMBINE_WINDOWS}"
echo "RUN_REPAIR_PASS=${RUN_REPAIR_PASS}"
echo "REPAIR_REPLACE_ORIGINAL=${REPAIR_REPLACE_ORIGINAL}"
echo "REPAIR_CANDIDATES=${REPAIR_CANDIDATES}"
echo "REPAIR_ATTEMPTS=${REPAIR_ATTEMPTS}"
echo "REPAIR_ACCEPT_SCORE=${REPAIR_ACCEPT_SCORE}"
echo "REPAIR_TRANSITION_THRESHOLD=${REPAIR_TRANSITION_THRESHOLD}"
echo "REPAIR_CRITIC_THRESHOLD=${REPAIR_CRITIC_THRESHOLD}"
echo "CAPTIONER_MODEL_ID=${CAPTIONER_MODEL_ID}"
echo "CAPTIONER_DEVICE=${CAPTIONER_DEVICE}"

bash -n "${BASH_SOURCE[0]}"

if [ "${PARALLEL_GPUS_PER_NODE}" -gt 1 ]; then
  BASE_SHARDS="${WINDOW_SHARD_COUNT}"
  BASE_INDEX="${WINDOW_SHARD_INDEX}"
  if [ "${BASE_SHARDS}" -le 1 ]; then
    BASE_SHARDS="${PARALLEL_GPUS_PER_NODE}"
    BASE_INDEX=0
  fi
  pids=()
  for local_gpu in $(seq 0 $((PARALLEL_GPUS_PER_NODE - 1))); do
    global_shard_index=$((BASE_INDEX * PARALLEL_GPUS_PER_NODE + local_gpu))
    global_shard_count=$((BASE_SHARDS * PARALLEL_GPUS_PER_NODE))
    SUB_CMD=("${CMD[@]}" --window_shard_count "${global_shard_count}" --window_shard_index "${global_shard_index}")
    echo "[parallel] launch gpu=${local_gpu} shard=${global_shard_index}/${global_shard_count}"
    CUDA_VISIBLE_DEVICES="${local_gpu}" "${SUB_CMD[@]}" &
    pids+=("$!")
  done
  rc=0
  for pid in "${pids[@]}"; do
    if ! wait "${pid}"; then
      rc=1
    fi
  done
  if [ "${rc}" -ne 0 ]; then
    echo "One or more parallel shard workers failed."
    exit "${rc}"
  fi
else
  "${CMD[@]}"
fi

if [ "${RUN_REPAIR_PASS}" = "1" ] && [ "${WINDOW_SHARD_COUNT}" = "1" ] && [ "${PARALLEL_GPUS_PER_NODE}" = "0" ]; then
  REPAIR_CMD=("${PYTHON_BIN}" scripts/08_repair_windows.py
    --run_dir "${OUTPUT_DIR}"
    --embedding_backend "${EMBEDDING_BACKEND}"
    --embedding_model_id "${EMBEDDING_MODEL_ID}"
    --video_model_id "${VIDEO_MODEL_ID}"
    --dtype "${DTYPE:-bfloat16}"
    --device "${DEVICE}"
    --num_frames "${NUM_FRAMES}"
    --steps "${STEPS}"
    --guidance_scale "${GUIDANCE_SCALE}"
    --height "${HEIGHT}"
    --width "${WIDTH}"
    --fps "${FPS}"
    --candidates "${REPAIR_CANDIDATES}"
    --attempts "${REPAIR_ATTEMPTS}"
    --accept_score "${REPAIR_ACCEPT_SCORE}"
    --transition_threshold "${REPAIR_TRANSITION_THRESHOLD}"
    --critic_threshold "${REPAIR_CRITIC_THRESHOLD}")
  if [ -n "${CAPTIONER_MODEL_ID}" ]; then
    REPAIR_CMD+=(--captioner_model_id "${CAPTIONER_MODEL_ID}" --captioner_device "${CAPTIONER_DEVICE}")
    if [ "${CAPTIONER_STUB_FALLBACK}" = "0" ]; then
      REPAIR_CMD+=(--captioner_stub_fallback 0)
    fi
  fi
  if [ "${REPAIR_REPLACE_ORIGINAL}" = "1" ]; then
    REPAIR_CMD+=(--replace_original)
  fi
  "${REPAIR_CMD[@]}"
fi

# Optional: concatenate generated windows into one video.
if [ "${COMBINE_WINDOWS}" = "1" ] && [ "${WINDOW_SHARD_COUNT}" = "1" ] && [ "${PARALLEL_GPUS_PER_NODE}" = "0" ]; then
  CLIPS_DIR="${OUTPUT_DIR}/clips"
  COMBINED_VIDEO_PATH="${OUTPUT_DIR}/${COMBINED_VIDEO_NAME}"
  CONCAT_LIST="${OUTPUT_DIR}/concat_windows.txt"

  if [ -d "${CLIPS_DIR}" ] && compgen -G "${CLIPS_DIR}/window_*.mp4" > /dev/null; then
    : > "${CONCAT_LIST}"
    for clip in "${CLIPS_DIR}"/window_*.mp4; do
      printf "file '%s'\n" "$(realpath "${clip}")" >> "${CONCAT_LIST}"
    done

    if command -v ffmpeg >/dev/null 2>&1; then
      if ! ffmpeg -y -f concat -safe 0 -i "${CONCAT_LIST}" -c copy "${COMBINED_VIDEO_PATH}"; then
        ffmpeg -y -f concat -safe 0 -i "${CONCAT_LIST}" -c:v libx264 -preset veryfast -crf 18 -c:a aac "${COMBINED_VIDEO_PATH}"
      fi
      echo "Combined continuity preview: ${COMBINED_VIDEO_PATH}"
    else
      echo "ffmpeg not found; skipping combined video creation."
    fi
  else
    echo "No window clips found in ${CLIPS_DIR}; skipping combined video creation."
  fi
fi
