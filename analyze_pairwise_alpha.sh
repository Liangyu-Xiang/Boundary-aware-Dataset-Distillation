#!/bin/bash
GPU_LIST=$1
FIRST_GPU=$(echo $GPU_LIST | cut -d',' -f1)
export CUDA_VISIBLE_DEVICES=$GPU_LIST
MASTER_PORT=$((25670 + FIRST_GPU))
GENSPEC=woof
N_CLASS=10
GLOBAL_BATCH_SIZE=8
MAX_STEPS=-1

if [ -z "$1" ]; then
    echo "Usage: bash analyze_pairwise_alpha.sh <GPU_ID>"
    exit 1
fi

NUM_GPUS=$(echo $GPU_LIST | awk -F',' '{print NF}')

echo "Using GPU: ${CUDA_VISIBLE_DEVICES}"

torchrun --nnodes=1 --nproc_per_node=${NUM_GPUS} --master_port=${MASTER_PORT} analyze_pairwise_alpha.py \
  --model DiT-XL/2 \
  --data-path "/data/mmc_lyxiang/dataset/ImageNet/train/" \
  --ckpt "/data/mmc_lyxiang/DD/MinimaxDiffusion/pretrained_models/DiT-XL-2-256x256.pt" \
  --vae ema \
  --global-batch-size ${GLOBAL_BATCH_SIZE} \
  --finetune-ipc -1 \
  --spec "${GENSPEC}" \
  --nclass ${N_CLASS} \
  --ratio-topk 2 \
  --max-steps ${MAX_STEPS}
