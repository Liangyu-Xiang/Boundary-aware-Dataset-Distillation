#!/bin/bash
set -euo pipefail

GPU_LIST=${1:-0}
PROJECT=${2:-expert_struct_run}
EXPERT_MODEL=${3:-resnet18}

FIRST_GPU=$(echo "${GPU_LIST}" | cut -d',' -f1)
NUM_GPUS=$(echo "${GPU_LIST}" | awk -F',' '{print NF}')
MASTER_PORT=${MASTER_PORT:-$((26670 + FIRST_GPU))}
EXPERT_MASTER_PORT=${EXPERT_MASTER_PORT:-$((27670 + FIRST_GPU))}
export CUDA_VISIBLE_DEVICES="${GPU_LIST}"

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PIPELINE_DIR="${ROOT_DIR}/expert_struct_pipeline"
VISION_DIR=${VISION_DIR:-/data/mmc_lyxiang/DD/vision}
VISION_TRAIN="${VISION_DIR}/references/classification/train.py"
IMAGEWOOF_ROOT=${IMAGEWOOF_ROOT:-"${VISION_DIR}/datasets/imagewoof_ordered_from_imagenet"}
EXPERT_OUT_DIR=${EXPERT_OUT_DIR:-"${VISION_DIR}/trained_model/imagewoof_${EXPERT_MODEL}"}
EXPERT_CKPT=${EXPERT_CKPT:-"${EXPERT_OUT_DIR}/checkpoint.pth"}

GENSPEC=${GENSPEC:-woof}
SPEC="imagenet-${GENSPEC}"
IPC=${IPC:-50}
N_CLASS=${N_CLASS:-10}
CFG=${CFG:-1.0}
NUM_CANDIDATES=${NUM_CANDIDATES:-4}
RUN_EVAL=${RUN_EVAL:-1}
RUN_EVAL_SWANLAB=${RUN_EVAL_SWANLAB:-1}
EVAL_REPEAT=${EVAL_REPEAT:-3}
EVAL_SEED=${EVAL_SEED:-42}
SWANLAB_PROJECT=${SWANLAB_PROJECT:-Boundary-aware-Dataset-Distillation}
SWANLAB_RUN=${SWANLAB_RUN:-"${PROJECT}_${EXPERT_MODEL}_ipc${IPC}_cfg${CFG}"}

IMAGENET_TRAIN_PATH=${IMAGENET_TRAIN_PATH:-/data/mmc_lyxiang/dataset/ImageNet/train/}
IMAGENET_VAL_PATH=${IMAGENET_VAL_PATH:-/data/mmc_lyxiang/dataset/ImageNet/val/}
EVAL_DIR=${EVAL_DIR:-/data/mmc_lyxiang/DD/CaO2ori/CaO2}
DIT_PRETRAIN_CKPT=${DIT_PRETRAIN_CKPT:-"${ROOT_DIR}/pretrained_models/DiT-XL-2-256x256.pt"}
RESULTS_DIR="${ROOT_DIR}/logs/${PROJECT}"
SAMPLE_SAVE_DIR="${ROOT_DIR}/results/OCD/Exp_IPC${IPC}/${GENSPEC}/${PROJECT}"

EXPERT_EPOCHS=${EXPERT_EPOCHS:-90}
EXPERT_BATCH_SIZE=${EXPERT_BATCH_SIZE:-32}
EXPERT_LR=${EXPERT_LR:-0.1}
EXPERT_WORKERS=${EXPERT_WORKERS:-8}

DIT_EPOCHS=${DIT_EPOCHS:-16}
DIT_GLOBAL_BATCH_SIZE=${DIT_GLOBAL_BATCH_SIZE:-8}
DIT_CKPT_EVERY=${DIT_CKPT_EVERY:-12000}
DIT_LOG_EVERY=${DIT_LOG_EVERY:-100}

cd "${ROOT_DIR}"

echo "Using GPUs: ${CUDA_VISIBLE_DEVICES}"
echo "Project: ${PROJECT}"
echo "Expert model: ${EXPERT_MODEL}"
echo "Expert checkpoint: ${EXPERT_CKPT}"

python "${PIPELINE_DIR}/prepare_imagewoof_links.py" \
  --class-file "${ROOT_DIR}/misc/class_woof.txt" \
  --train-src "${IMAGENET_TRAIN_PATH}" \
  --val-src "${IMAGENET_VAL_PATH}" \
  --output-root "${IMAGEWOOF_ROOT}"

if [ ! -f "${EXPERT_CKPT}" ]; then
  echo "Expert checkpoint missing; training ${EXPERT_MODEL} on ImageWoof."
  mkdir -p "${EXPERT_OUT_DIR}"
  torchrun --nnodes=1 --nproc_per_node="${NUM_GPUS}" --master_port="${EXPERT_MASTER_PORT}" "${VISION_TRAIN}" \
    --data-path "${IMAGEWOOF_ROOT}" \
    --model "${EXPERT_MODEL}" \
    --output-dir "${EXPERT_OUT_DIR}" \
    --epochs "${EXPERT_EPOCHS}" \
    --batch-size "${EXPERT_BATCH_SIZE}" \
    --lr "${EXPERT_LR}" \
    --workers "${EXPERT_WORKERS}"
fi

if [ ! -f "${EXPERT_CKPT}" ]; then
  echo "Expected expert checkpoint was not created: ${EXPERT_CKPT}" >&2
  exit 1
fi

echo "Fine-tuning DiT with ${EXPERT_MODEL}."
torchrun --nnodes=1 --nproc_per_node="${NUM_GPUS}" --master_port="${MASTER_PORT}" \
  "${PIPELINE_DIR}/train_dit_multi_label_gated_expert.py" \
  --model DiT-XL/2 \
  --data-path "${IMAGENET_TRAIN_PATH}" \
  --ckpt "${DIT_PRETRAIN_CKPT}" \
  --results-dir "${RESULTS_DIR}" \
  --tag "${PROJECT}" \
  --vae ema \
  --ckpt-every "${DIT_CKPT_EVERY}" \
  --log-every "${DIT_LOG_EVERY}" \
  --global-batch-size "${DIT_GLOBAL_BATCH_SIZE}" \
  --epochs "${DIT_EPOCHS}" \
  --finetune-ipc -1 \
  --spec "${GENSPEC}" \
  --nclass "${N_CLASS}" \
  --expert-model "${EXPERT_MODEL}" \
  --expert-ckpt "${EXPERT_CKPT}"

DIT_CKPT=$(find "${RESULTS_DIR}" -path "*/checkpoints/last.pt" -printf '%T@ %p\n' | sort -nr | head -n1 | cut -d' ' -f2-)
if [ -z "${DIT_CKPT}" ] || [ ! -f "${DIT_CKPT}" ]; then
  echo "Could not locate fine-tuned DiT checkpoint under ${RESULTS_DIR}" >&2
  exit 1
fi
echo "Using DiT checkpoint: ${DIT_CKPT}"

python "${PIPELINE_DIR}/sample_mult_label_sel_expert.py" \
  --model DiT-XL/2 \
  --image-size 256 \
  --ckpt "${DIT_CKPT}" \
  --save-dir "${SAMPLE_SAVE_DIR}" \
  --spec "${GENSPEC}" \
  --num-samples "${IPC}" \
  --nclass "${N_CLASS}" \
  --cfg-scale "${CFG}" \
  --disable-stage-cfg \
  --num-candidates "${NUM_CANDIDATES}" \
  --expert-model "${EXPERT_MODEL}" \
  --expert-ckpt "${EXPERT_CKPT}"

if [ "${RUN_EVAL}" = "1" ]; then
  if [ "${RUN_EVAL_SWANLAB}" = "1" ]; then
    for repeat_idx in $(seq 1 "${EVAL_REPEAT}"); do
      current_seed=$((EVAL_SEED + repeat_idx - 1))
      repeat_name=$(printf "%s_repeat%02d" "${SWANLAB_RUN}" "${repeat_idx}")
      EVAL_CMD=(
        python main_validate_random.py
        --subset "${SPEC}"
        --arch-name "resnet18"
        --factor 2
        --num-crop 5
        --mipc 300
        --ipc "${IPC}"
        --stud-name "resnet18"
        --re-epochs 300
        --train-dir "${IMAGENET_TRAIN_PATH}"
        --val-dir "${IMAGENET_VAL_PATH}"
        --repeat 1
        --seed "${current_seed}"
        --syn-data-path "${SAMPLE_SAVE_DIR}"
      )

      python "${ROOT_DIR}/swanlab_eval_wrapper.py" \
        --work-dir "${EVAL_DIR}" \
        --swanlab-project "${SWANLAB_PROJECT}" \
        --swanlab-run "${repeat_name}" \
        --cfg "${CFG}" \
        --ipc "${IPC}" \
        --spec "${GENSPEC}" \
        --repeat 1 \
        -- "${EVAL_CMD[@]}"
    done
  else
    EVAL_CMD=(
      python main_validate_random.py
      --subset "${SPEC}"
      --arch-name "resnet18"
      --factor 2
      --num-crop 5
      --mipc 300
      --ipc "${IPC}"
      --stud-name "resnet18"
      --re-epochs 300
      --train-dir "${IMAGENET_TRAIN_PATH}"
      --val-dir "${IMAGENET_VAL_PATH}"
      --repeat "${EVAL_REPEAT}"
      --seed "${EVAL_SEED}"
      --syn-data-path "${SAMPLE_SAVE_DIR}"
    )

    cd "${EVAL_DIR}"
    "${EVAL_CMD[@]}"
  fi
fi

echo "Pipeline finished."
