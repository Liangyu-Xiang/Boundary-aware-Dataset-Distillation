#!/usr/bin/env bash
GPU_ID=${1:-0}
CUDA_VISIBLE_DEVICES=${GPU_ID} python BoundarySet-CCS.py \
    --dataset imagefolder \
    --data-root /data/mmc_lyxiang/dataset/ImageWoof/train/ \
    --checkpoint /data/mmc_lyxiang/DD/CaO2/data/pretrain_models/imagenet-woof_resnet18.pth \
    --num-classes 10 \
    --per-class-count 10 \
    --method boundaryset \
    --alpha 0.0001 \
    --max-step 50 \
    --batch-size 64 \
    --num-workers 8 \
    --resize-size 256 \
    --crop-size 224 \
    --output-dir /data/mmc_lyxiang/DD/MinimaxDiffusion/outputs/BoundarySet-CCS \
    --export-dir /data/mmc_lyxiang/DD/MinimaxDiffusion/outputs/BoundarySet-CCS
