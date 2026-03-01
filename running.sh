#!/usr/bin/env bash
set -euo pipefail

echo "=== Starting sceneweaver_full job ==="
echo "Running on node: $(hostname)"
echo "Job ID: ${SLURM_JOB_ID}"
echo "Submit directory: ${SLURM_SUBMIT_DIR}"
echo "Start time: $(date)"
echo "--------------------------------------"

# Always run from the directory where the job was submitted
cd "${SLURM_SUBMIT_DIR}"

# Create log folder if needed
mkdir -p slurm_logs

# (Optional) Activate conda / venv if you use one
# source ~/.bashrc
# conda activate your_env_name

# Print GPU info
echo "=== GPU Info ==="
nvidia-smi
echo "--------------------------------------"

# Run your actual pipeline
# Replace this with your real entry point
python your_main_script.py "$@"

echo "--------------------------------------"
echo "Job finished at: $(date)"