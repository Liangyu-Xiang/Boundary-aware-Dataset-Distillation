#!/bin/bash
set -euo pipefail

export CUDA_VISIBLE_DEVICES=$1
PROJECT=$2

SD_PATH="/data/mmc_lyxiang/DD/MinimaxDiffusion/pretrained_models/stable-diffusion-v1-5"
UNET_PATH="/data/mmc_lyxiang/DD/MinimaxDiffusion/pretrained_models/stable-diffusion-v1-5/unet/"
CLASS_NAMES="/data/mmc_lyxiang/DD/MinimaxDiffusion/outputs/imagewoof_class_names.txt"
OUTPUT_DIR="/data/mmc_lyxiang/DD/MinimaxDiffusion/results/sd_samples/${PROJECT}"

NUM_SAMPLES_PER_CLASS=50
BATCH_SIZE=1
STEPS=50
CFG_SCALE=7.5
RESOLUTION=512
PRECISION="fp16"
SECOND_PROMPT_MODE="random"
FIRST_HALF_WEIGHT=0.5
SECOND_HALF_WEIGHT=0.0

echo "Using GPU: ${CUDA_VISIBLE_DEVICES}"
echo "Project: ${PROJECT}"

python sample_sd_multi_label_gated.py \
  --sd_path "$SD_PATH" \
  --class_names_path "$CLASS_NAMES" \
  --output_dir "$OUTPUT_DIR" \
  --batch_size "$BATCH_SIZE" \
  --num_samples_per_class "$NUM_SAMPLES_PER_CLASS" \
    --resolution "$RESOLUTION" \

echo "Sample generation finished."
