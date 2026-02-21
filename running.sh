#!/usr/bin/env bash
set -euo pipefail
DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${DIR}"
mkdir -p slurm_logs

# Defaults aligned to current idle availability from sinfo; override when needed.
SLURM_PARTITION="${SLURM_PARTITION:-a40}"
SLURM_GRES="${SLURM_GRES:-gpu:a40:1}"
SLURM_NODELIST="${SLURM_NODELIST:-}"

SBATCH_ARGS=(
  --partition="${SLURM_PARTITION}"
  --gres="${SLURM_GRES}"
  --export=ALL
)

if [ -n "${SLURM_NODELIST}" ]; then
  SBATCH_ARGS+=(--nodelist="${SLURM_NODELIST}")
fi

exec sbatch \
  "${SBATCH_ARGS[@]}" \
  "${DIR}/run_story_pipeline.sh" \
  "$@"
