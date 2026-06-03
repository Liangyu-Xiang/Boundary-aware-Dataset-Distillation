#!/bin/bash

export CUDA_VISIBLE_DEVICES=$1
INPUT_PATH=$2
OUTPUT_PATH=${3:-/data/mmc_lyxiang/DD/MinimaxDiffusion/results/Predictions/Minimax.csv}
SPEC=${4:-woof}
N_CLASS=${5:-10}
RECURSIVE=${6:-0}
TOPK=${7:-5}
DEVICE=${8:-}

RECURSIVE_ARG=""
if [ "${RECURSIVE}" = "1" ]; then
    RECURSIVE_ARG="--recursive"
fi

DEVICE_ARG=""
if [ -n "${DEVICE}" ]; then
    DEVICE_ARG="--device ${DEVICE}"
fi

echo "Using GPU: ${CUDA_VISIBLE_DEVICES}"
echo "input_path: ${INPUT_PATH}"
echo "output_path: ${OUTPUT_PATH}"
echo "spec: ${SPEC}"
echo "nclass: ${N_CLASS}"
echo "recursive: ${RECURSIVE}"
echo "topk: ${TOPK}"

python tools/infer_expert_model.py \
    --spec "${SPEC}" \
    --nclass "${N_CLASS}" \
    --paths "${INPUT_PATH}" \
    --topk "${TOPK}" \
    --output "${OUTPUT_PATH}" \
    ${RECURSIVE_ARG} \
    ${DEVICE_ARG}

echo "Expert inference finished."
