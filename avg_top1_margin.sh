#!/bin/bash
set -euo pipefail

export CUDA_VISIBLE_DEVICES=$1
SPEC=$2
NUM_CLASSES=10
DATA_PATH=$3

BATCH_SIZE=64
NUM_WORKERS=4
MAX_SAMPLES=50
MARGIN_ON="probs"

echo "Using GPU: ${CUDA_VISIBLE_DEVICES}"
echo "Spec: ${SPEC}"
echo "Num classes: ${NUM_CLASSES}"
echo "Data path: ${DATA_PATH}"

python avg_top1_margin.py \
  --data_path "$DATA_PATH" \
  --spec "$SPEC" \
  --num_classes "$NUM_CLASSES" \
  --batch_size "$BATCH_SIZE" \
  --num_workers "$NUM_WORKERS" \
  --margin_on "$MARGIN_ON"

