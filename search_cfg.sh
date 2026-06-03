#!/bin/bash
set -e

export CUDA_VISIBLE_DEVICES=$1
PROJECT=$2
GENSPEC=woof
IPC=50
SPEC=imagenet-${GENSPEC}
ROOT_DIR="/data/mmc_lyxiang/DD/MinimaxDiffusion"
EVAL_DIR="/data/mmc_lyxiang/DD/CaO2ori/CaO2"
IMAGENET_PATH="/data/mmc_lyxiang/dataset/ImageNet/train/"
IMAGENET_TRAIN_PATH="/data/mmc_lyxiang/dataset/ImageNet/train/"
IMAGENET_VAL_PATH="/data/mmc_lyxiang/dataset/ImageNet/val/"
N_CLASS=10
SWANLAB_PROJECT="BDD-CFG-Search"
REPEAT=3
BASE_SEED=42
TEMPERATURE=10
CKPT="/data/mmc_lyxiang/DD/MinimaxDiffusion/logs/p_x_y_2_Gated+Epoch16+Temp2/000-DiT-XL-2-p_x_y_2_Gated+Epoch16+Temp2/checkpoints/last.pt"
# CKPT="/data/mmc_lyxiang/DD/MinimaxDiffusion/logs/dif_denoise_pairwise_alpha30_lambda_ratio0.003/000-DiT-XL-2-dif_denoise_pairwise_alpha30_lambda_ratio0.003/checkpoints/epoch_003.pt"

echo "Using GPU: ${CUDA_VISIBLE_DEVICES}"
echo "Project: ${PROJECT}"

for CFG in 3.5 4.0 4.5 5.0; do
    CFG_TAG=${CFG/./p}
    SYN_DATA_PATH="${ROOT_DIR}/results/BDD/Exp_IPC${IPC}/${GENSPEC}/${PROJECT}/cfg_${CFG_TAG}"

    echo "========================================"
    echo "Running CFG=${CFG}"
    echo "Synthetic data path: ${SYN_DATA_PATH}"
    echo "========================================"

    cd "${ROOT_DIR}"
    python sample_mult_label_sel.py --model DiT-XL/2 --image-size 256 --ckpt "$CKPT" \
        --save-dir "${SYN_DATA_PATH}" --spec "$GENSPEC" --num-samples "$IPC" \
        --nclass "$N_CLASS" --cfg-scale "$CFG" --disable-stage-cfg --disable-selection

    echo "Sample generation finished for CFG=${CFG}."

    for REP in $(seq 1 "$REPEAT"); do
        SEED=$((BASE_SEED + REP - 1))

        cd "${ROOT_DIR}"
        echo "Running evaluation repeat ${REP}/${REPEAT} with swanlab logging in ${EVAL_DIR}"

        python swanlab_eval_wrapper.py \
            --work-dir "${EVAL_DIR}" \
            --swanlab-project "${SWANLAB_PROJECT}" \
            --swanlab-run "cfg_${CFG_TAG}_rep${REP}" \
            --cfg "$CFG" \
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

    echo "Evaluation finished for CFG=${CFG}."
done

echo "CFG search finished."
