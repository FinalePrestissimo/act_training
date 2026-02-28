#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

DATASET_DIR="${1:-}"
GPU_ID="${2:-0}"

if [[ -z "${DATASET_DIR}" ]]; then
  echo "Usage: bash scripts/train_real.sh <dataset_dir> [gpu_id]"
  exit 1
fi

DATASET_NAME="$(basename "${DATASET_DIR}")"
CKPT_DIR="${ROOT_DIR}/checkpoint/${DATASET_NAME}"

export CUDA_VISIBLE_DEVICES="${GPU_ID}"

cd "${ROOT_DIR}"
uv run python train_real.py \
  --dataset_dir "${DATASET_DIR}" \
  --ckpt_dir "${CKPT_DIR}" \
  --policy_class ACT \
  --batch_size 8 \
  --num_epochs 6000 \
  --lr 1e-5 \
  --save_freq 2000 \
  --kl_weight 10 \
  --chunk_size 50 \
  --hidden_dim 512 \
  --dim_feedforward 3200 \
  --state_dim 14
