#!/bin/bash

# =========================
# Parse arguments
# =========================
GPU_ID=$1

if [ -z "$GPU_ID" ]; then
    echo "Usage: bash eval_expert.sh <GPU_ID>"
    echo "Example: bash eval_expert.sh 3"
    exit 1
fi

export CUDA_VISIBLE_DEVICES=$GPU_ID

echo "Using GPU: ${CUDA_VISIBLE_DEVICES}"

# =========================
# Paths (modify if needed)
# =========================
DATA_PATH="/data/mmc_lyxiang/dataset/ImageNet/train"
SAVE_DIR="/data/mmc_lyxiang/DD/MinimaxDiffusion/results/expert_confusion"
SAVE_PATH="${SAVE_DIR}/expert_confusion_imagenet-woof.pt"

mkdir -p "${SAVE_DIR}"

# =========================
# Run expert evaluation
# =========================
python eval_expert.py \
  --data-path "${DATA_PATH}" \
  --spec woof \
  --nclass 10 \
  --finetune-ipc -1 \
  --phase 0 \
  --batch-size 64 \
  --num-workers 8 \
  --save-path "${SAVE_PATH}"

echo "Expert confusion evaluation finished."
echo "Saved to: ${SAVE_PATH}"
