#!/bin/bash
GPU_LIST=$1
FIRST_GPU=$(echo $GPU_LIST | cut -d',' -f1)
export CUDA_VISIBLE_DEVICES=$GPU_LIST
MASTER_PORT=$((25670 + FIRST_GPU))
PROJECT=$2

GENSPEC=woof
N_CLASS=10
TOPK=2
BAYES_ALPHA=1000
LAMBDA_BAYES=0.001

if [ -z "$1" ]; then
    echo "Usage: bash train_dit_bayes.sh <GPU_ID> <PROJECT>"
    exit 1
fi

if [ -z "$2" ]; then
    echo "Usage: bash train_dit_bayes.sh <GPU_ID> <PROJECT>"
    exit 1
fi

NUM_GPUS=$(echo $GPU_LIST | awk -F',' '{print NF}')

echo "Using GPU: ${CUDA_VISIBLE_DEVICES}"
echo "Project: ${PROJECT}"

torchrun --nnodes=1 --nproc_per_node=${NUM_GPUS} --master_port=${MASTER_PORT} train_dit_bayes.py \
  --model DiT-XL/2 \
  --data-path "/data/mmc_lyxiang/dataset/ImageNet/train/" \
  --ckpt "/data/mmc_lyxiang/DD/MinimaxDiffusion/pretrained_models/DiT-XL-2-256x256.pt" \
  --results-dir "./logs/${PROJECT}" \
  --tag ${PROJECT} \
  --vae ema \
  --log-every 1500 \
  --global-batch-size 8 \
  --epochs 16 \
  --finetune-ipc -1 \
  --spec ${GENSPEC} \
  --nclass ${N_CLASS} \
  --topk ${TOPK} \
  --bayes-alpha ${BAYES_ALPHA} \
  --lambda-bayes ${LAMBDA_BAYES} \
  --expert-model resnet18 \
  --debug-print-probs
