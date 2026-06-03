#!/bin/bash

export CUDA_VISIBLE_DEVICES=$1

IPC_LIST=(50 70 100)

CKPT="/data/mmc_lyxiang/DD/MinimaxDiffusion/logs/baseline/006-DiT-XL-2-baseline/checkpoints/0012000.pt"

echo "Using GPU: ${CUDA_VISIBLE_DEVICES}"
echo "IPC list: ${IPC_LIST[@]}"

for IPC in "${IPC_LIST[@]}"; do
    echo -e "\033[1;34m[IPC=${IPC}] Generating synthetic samples...\033[0m"

    SAVE_DIR="./results/dit-distillation/Minimax/Nette-IPC${IPC}"

    python sample.py \
        --model DiT-XL/2 \
        --image-size 256 \
        --ckpt "${CKPT}" \
        --save-dir "${SAVE_DIR}" \
        --spec nette \
        --nclass 10 \
        --num-samples ${IPC} \
        --cfg-scale 4.0

    echo -e "\033[1;32m[IPC=${IPC}] Generation finished.\033[0m"
done

echo -e "\033[1;36mAll IPC generation finished.\033[0m"
