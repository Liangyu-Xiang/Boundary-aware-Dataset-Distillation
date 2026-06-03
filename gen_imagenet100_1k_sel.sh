#!/bin/bash
set -e

export CUDA_VISIBLE_DEVICES=$1
PROJECT=$2

ROOT_DIR="/data/mmc_lyxiang/DD/MinimaxDiffusion"
EVAL_DIR="/data/mmc_lyxiang/DD/CaO2ori/CaO2"
IMAGENET_TRAIN_PATH="/data/mmc_lyxiang/dataset/ImageNet/train/"
IMAGENET_VAL_PATH="/data/mmc_lyxiang/dataset/ImageNet/val/"

IPC=${IPC:-50}
CFG=${CFG:-1.0}
REPEAT=${REPEAT:-3}
RE_EPOCHS=500
TEMPERATURE=15
NUM_CANDIDATES=${NUM_CANDIDATES:-4}
CKPT=${CKPT:-"/data/mmc_lyxiang/DD/MinimaxDiffusion/logs/p_x_y_2_Gated+Epoch16+Temp2/000-DiT-XL-2-p_x_y_2_Gated+Epoch16+Temp2/checkpoints/last.pt"}

if [ -z "$CUDA_VISIBLE_DEVICES" ] || [ -z "$PROJECT" ]; then
    echo "Usage: $0 <gpu_id> <project_name>"
    echo "Optional overrides: IPC=50 CFG=1.0 REPEAT=3 NUM_CANDIDATES=4 CKPT=/path/to/ckpt.pt $0 <gpu_id> <project_name>"
    exit 1
fi

echo "Using GPU: ${CUDA_VISIBLE_DEVICES}"
echo "Project: ${PROJECT}"
echo "IPC=${IPC}, CFG=${CFG}, repeat=${REPEAT}, re_epochs=${RE_EPOCHS}, temperature=${TEMPERATURE}"
echo "Selection enabled, stage-cfg disabled, num_candidates=${NUM_CANDIDATES}"
echo "Checkpoint: ${CKPT}"

run_one_dataset() {
    local GENSPEC=$1
    local N_CLASS=$2
    local SPEC="imagenet-${GENSPEC}"
    local SAVE_DIR="${ROOT_DIR}/results/OCD/Exp_IPC${IPC}/${GENSPEC}/${PROJECT}"

    if [ "$GENSPEC" = "100" ]; then
        SPEC="imagenet-100"
    elif [ "$GENSPEC" = "1k" ]; then
        SPEC="imagenet-1k"
    fi

    echo "========================================"
    echo "Generating ${SPEC}"
    echo "Save dir: ${SAVE_DIR}"
    echo "========================================"

    cd "${ROOT_DIR}"
    python sample_mult_label_sel.py --model DiT-XL/2 --image-size 256 --ckpt "$CKPT" \
        --save-dir "$SAVE_DIR" --spec "$GENSPEC" --num-samples "$IPC" \
        --nclass "$N_CLASS" --cfg-scale "$CFG" --disable-stage-cfg \
        --num-candidates "$NUM_CANDIDATES"

    echo "Sample generation finished for ${SPEC}."

    cd "${EVAL_DIR}"
    echo "Switched to evaluation directory: $(pwd)"

    python main_validate_random.py --subset "$SPEC" --arch-name "resnet18" --factor 2 \
        --num-crop 5 --mipc 300 --ipc "$IPC" --stud-name "resnet18" --re-epochs "$RE_EPOCHS" \
        --train-dir "$IMAGENET_TRAIN_PATH" --val-dir "$IMAGENET_VAL_PATH" --repeat "$REPEAT" \
        --temperature "$TEMPERATURE" \
        --syn-data-path "$SAVE_DIR"

    echo "Evaluation finished for ${SPEC}."
}

run_one_dataset "100" 100
run_one_dataset "1k" 1000

echo "ImageNet-100 and ImageNet-1k generation/evaluation finished."
