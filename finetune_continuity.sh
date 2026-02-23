#!/bin/bash -l
#SBATCH --job-name=ft_pororo_cont
#SBATCH --output=slurm_logs/ft_pororo_cont_%j.out
#SBATCH --error=slurm_logs/ft_pororo_cont_%j.err
#SBATCH --time=10:00:00
#SBATCH --partition=a40
#SBATCH --gres=gpu:a40:1

set -euo pipefail

PROJECT_ROOT="/home/hpc/v123be/v123be36/Multimodal_Final_SceneWeaver"
DATASET_ROOT="/home/vault/v123be/v123be36/PororoSV"
DINO_MODEL_ID="/home/vault/v123be/v123be36/facebook/dinov2-base"

source /apps/python/3.12-conda/etc/profile.d/conda.sh
conda activate sceneweaver_runtime

cd "${PROJECT_ROOT}"

python scripts/finetune_pororo_continuity.py \
  --dataset_root "${DATASET_ROOT}" \
  --dino_model_id "${DINO_MODEL_ID}" \
  --device auto \
  --epochs 6 \
  --batch_size 64 \
  --temperature 0.15 \
  --lr_projector 5e-5 \
  --weight_decay 5e-2 \
  --val_split seen \
  --save_path outputs/pororo_continuity_adapter.pt
