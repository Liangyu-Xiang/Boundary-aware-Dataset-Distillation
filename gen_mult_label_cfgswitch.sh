#!/bin/bash

export CUDA_VISIBLE_DEVICES=$1
PROJECT=$2
LABEL_WEIGHT=${3:-0.5}
ENABLE_STAGE_CFG=${4:-1}
MULTI_COND_RATIO=${5:-1.0}

GENSPEC=woof
IPC=50
N_CLASS=10
CKPT="/data/mmc_lyxiang/DD/MinimaxDiffusion/logs/Bayes_BayesAlpha1000_LambdaBayes0.001/000-DiT-XL-2-Bayes_BayesAlpha1000_LambdaBayes0.001/checkpoints/epoch_003.pt"
SAVE_DIR="./results/Finetune/Exp_IPC${IPC}/${GENSPEC}/${PROJECT}"
SPEC="imagenet-${GENSPEC}"
EVAL_DIR="/data/mmc_lyxiang/DD/CaO2ori/CaO2"
IMAGENET_TRAIN_PATH="/data/mmc_lyxiang/dataset/ImageNet/train/"
IMAGENET_VAL_PATH="/data/mmc_lyxiang/dataset/ImageNet/val/"

echo "Using GPU: ${CUDA_VISIBLE_DEVICES}"
echo "Project: ${PROJECT}"
echo "label_weight: ${LABEL_WEIGHT}"
echo "save_dir: ${SAVE_DIR}"
echo "multi_cond_ratio: ${MULTI_COND_RATIO}"
echo "enable_stage_cfg: ${ENABLE_STAGE_CFG}"

STAGE_CFG_ARG=""
if [ "${ENABLE_STAGE_CFG}" = "0" ]; then
    STAGE_CFG_ARG="--disable-stage-cfg"
fi

python sample_mult_label_cfgswitch.py \
    --model DiT-XL/2 \
    --image-size 256 \
    --ckpt "${CKPT}" \
    --save-dir "${SAVE_DIR}" \
    --spec "${GENSPEC}" \
    --num-samples "${IPC}" \
    --nclass "${N_CLASS}" \
    --multi-cond-ratio "${MULTI_COND_RATIO}" \
    --label-weight "${LABEL_WEIGHT}" \
    ${STAGE_CFG_ARG}

echo "Sample generation finished."

cd "${EVAL_DIR}"
echo "Switched to evaluation directory: $(pwd)"

python main_validate_random.py \
    --subset "${SPEC}" \
    --arch-name "resnet18" \
    --factor 2 \
    --num-crop 5 \
    --mipc 300 \
    --ipc "${IPC}" \
    --stud-name "resnet18" \
    --re-epochs 300 \
    --train-dir "${IMAGENET_TRAIN_PATH}" \
    --val-dir "${IMAGENET_VAL_PATH}" \
    --repeat 10 \
    --syn-data-path "/data/mmc_lyxiang/DD/MinimaxDiffusion/results/Finetune/Exp_IPC${IPC}/${GENSPEC}/${PROJECT}"

echo "Evaluation finished."
