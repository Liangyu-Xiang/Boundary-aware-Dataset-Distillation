#!/usr/bin/env bash
set -euo pipefail

SD_PATH="/data/mmc_lyxiang/DD/MinimaxDiffusion/pretrained_models/stable-diffusion-v1-5"
DATA_PATH="/data/mmc_lyxiang/dataset/ImageWoof/training_dataset/ImageWoof_ori/train"
OUTPUT_DIR="/data/mmc_lyxiang/DD/MinimaxDiffusion/logs/sd_gated_woof"
CLASS_NAMES_FILE="/data/mmc_lyxiang/DD/MinimaxDiffusion/outputs/imagewoof_class_names.txt"
NUM_GPUS=4
CUDA_VISIBLE_DEVICES="4,5,6,7"

# Auto-generate class names from folder names (sorted) if missing.
if [[ ! -f "$CLASS_NAMES_FILE" ]]; then
  mkdir -p "$(dirname "$CLASS_NAMES_FILE")"
  find "$DATA_PATH" -mindepth 1 -maxdepth 1 -type d -printf '%f\n' | sort > "$CLASS_NAMES_FILE"
fi

CUDA_VISIBLE_DEVICES="$CUDA_VISIBLE_DEVICES" \
accelerate launch --num_processes "$NUM_GPUS" train_sd_multi_label_gated.py \
  --sd_path "$SD_PATH" \
  --data_path "$DATA_PATH" \
  --class_names_path "$CLASS_NAMES_FILE" \
  --output_dir "$OUTPUT_DIR" \
  --num_classes 10 \
  --resolution 512 \
  --center_crop --random_flip \
  --train_batch_size 8 \
  --gradient_accumulation_steps 4 \
  --gradient_checkpointing \
  --mixed_precision fp16 \
  --lr 1e-5 \
  --max_grad_norm 1 \
  --lr_scheduler constant \
  --lr_warmup_steps 0 \
  --num_train_epochs 8 \
  --checkpointing_steps 500 \
  --checkpoints_total_limit 2 \
  --use_ema \
  --seed 0
