#!/bin/bash
GPU_LIST=$1
FIRST_GPU=$(echo $GPU_LIST | cut -d',' -f1)
export CUDA_VISIBLE_DEVICES=$GPU_LIST
MASTER_PORT=$((25670 + FIRST_GPU))
PROJECT=$2
GENSPEC=woof
IPC=50
SPEC=imagenet-${GENSPEC}
EVAL_DIR="/data/mmc_lyxiang/DD/CaO2ori/CaO2"
IMAGENET_TRAIN_PATH="/data/mmc_lyxiang/dataset/ImageNet/train/"
IMAGENET_VAL_PATH="/data/mmc_lyxiang/dataset/ImageNet/val/"
N_CLASS=10
CFG=1.0
RESULTS_DIR="./logs/${PROJECT}"
SAMPLE_SAVE_DIR="./results/dif/Exp_IPC${IPC}/${GENSPEC}/${PROJECT}"


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


torchrun --nnodes=1 --nproc_per_node=${NUM_GPUS} --master_port=${MASTER_PORT} train_dit_multi_label_gated.py \
  --model DiT-XL/2 \
  --data-path "/data/mmc_lyxiang/dataset/ImageNet/train/" \
  --ckpt "/data/mmc_lyxiang/DD/MinimaxDiffusion/pretrained_models/DiT-XL-2-256x256.pt" \
  --results-dir ./logs/${PROJECT} \
  --tag ${PROJECT} \
  --vae ema \
  --ckpt-every 12000 \
  --global-batch-size 8 \
  --epochs 16 \
  --finetune-ipc -1 \
  --spec "woof" \
  --nclass 10 \
  # --condense \


