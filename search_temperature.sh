#!/bin/bash
set -e

export CUDA_VISIBLE_DEVICES=$1
PROJECT=$2
SYN_DATA_PATH=$3

GENSPEC=woof
IPC=50
SPEC=imagenet-${GENSPEC}
ROOT_DIR="/data/mmc_lyxiang/DD/MinimaxDiffusion"
EVAL_DIR="/data/mmc_lyxiang/DD/CaO2ori/CaO2"
IMAGENET_TRAIN_PATH="/data/mmc_lyxiang/dataset/ImageNet/train/"
IMAGENET_VAL_PATH="/data/mmc_lyxiang/dataset/ImageNet/val/"
SWANLAB_PROJECT="BDD-Temperature-Search"
REPEAT=3
BASE_SEED=42
TEMPERATURES="${TEMPERATURES:-10 15 20 30}"

if [ -z "$CUDA_VISIBLE_DEVICES" ] || [ -z "$PROJECT" ] || [ -z "$SYN_DATA_PATH" ]; then
    echo "Usage: $0 <gpu_id> <project_name> <synthetic_data_path>"
    echo "Example: $0 4 temp_search /data/mmc_lyxiang/DD/MinimaxDiffusion/results/BDD/Exp_IPC50/woof/BDD_CFG_SEARCH/cfg_3p5"
    echo "Override temperatures with: TEMPERATURES=\"4 8 12 16 20\" $0 4 temp_search <path>"
    exit 1
fi

if [ ! -d "$SYN_DATA_PATH" ]; then
    echo "Synthetic data path does not exist: $SYN_DATA_PATH"
    exit 1
fi

echo "Using GPU: ${CUDA_VISIBLE_DEVICES}"
echo "Project: ${PROJECT}"
echo "Synthetic data path: ${SYN_DATA_PATH}"
echo "Temperatures: ${TEMPERATURES}"

for TEMPERATURE in $TEMPERATURES; do
    TEMP_TAG=${TEMPERATURE/./p}

    echo "========================================"
    echo "Running temperature=${TEMPERATURE}"
    echo "========================================"

    for REP in $(seq 1 "$REPEAT"); do
        SEED=$((BASE_SEED + REP - 1))

        cd "${ROOT_DIR}"
        echo "Running temperature=${TEMPERATURE}, repeat ${REP}/${REPEAT}"

        python swanlab_eval_wrapper.py \
            --work-dir "${EVAL_DIR}" \
            --swanlab-project "${SWANLAB_PROJECT}" \
            --swanlab-run "${PROJECT}_temp_${TEMP_TAG}_rep${REP}" \
            --temperature "$TEMPERATURE" \
            --ipc "$IPC" \
            --spec "$SPEC" \
            --repeat 1 \
            -- \
            python main_validate_random.py --subset "$SPEC" --arch-name "resnet18" --factor 2 \
            --num-crop 5 --mipc 300 --ipc "$IPC" --stud-name "resnet18" --re-epochs 300 \
            --train-dir "$IMAGENET_TRAIN_PATH" --val-dir "$IMAGENET_VAL_PATH" --repeat 1 \
            --seed "$SEED" \
            --temperature "$TEMPERATURE" \
            --syn-data-path "$SYN_DATA_PATH"
    done

    echo "Evaluation finished for temperature=${TEMPERATURE}."
done

echo "Temperature search finished."
