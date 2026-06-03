#!/bin/bash
GPU_LIST=$1
FIRST_GPU=$(echo $GPU_LIST | cut -d',' -f1)
export CUDA_VISIBLE_DEVICES=$GPU_LIST
MASTER_PORT=$((25670 + FIRST_GPU))
PROJECT=$2


if [ -z "$1" ]; then
    echo "Usage: bash finetune.sh <GPU_ID>"
    exit 1
fi

NUM_GPUS=$(echo $GPU_LIST | awk -F',' '{print NF}')

echo "Using GPU: ${CUDA_VISIBLE_DEVICES}"

# fine-tune the diffusion model
# torchrun --nnode=1 --master_port=${MASTER_PORT} train_dit_woof_expert.py --model DiT-XL/2 \
#      --data-path /data/mmc_lyxiang/dataset/ImageNet/train/ --ckpt pretrained_models/DiT-XL-2-256x256.pt \
#      --global-batch-size 8 --tag ${PROJECT} --ckpt-every 12000 --log-every 1500 --epochs 8 \
#      --condense --finetune-ipc -1 --results-dir ./logs/${PROJECT} --spec woof


torchrun --nnodes=1 --nproc_per_node=${NUM_GPUS} --master_port=${MASTER_PORT} expert_confusion_matrix.py \
  --model DiT-XL/2 \
  --data-path "/data/mmc_lyxiang/dataset/ImageNet/train/" \
  --ckpt "/data/mmc_lyxiang/DD/MinimaxDiffusion/pretrained_models/DiT-XL-2-256x256.pt" \
  --results-dir ./logs/confusion/${PROJECT} \
  --tag ${PROJECT} \
  --vae ema \
  --ckpt-every 12000 \
  --global-batch-size 512 \
  --epochs 1 \
  --finetune-ipc -1 \
  --spec "1k" \
  --nclass 1000 \
  --num-workers 16
  # --condense \

