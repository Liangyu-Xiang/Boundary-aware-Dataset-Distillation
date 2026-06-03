#!/bin/bash
export CUDA_VISIBLE_DEVICES=$1
PROJECT=$2
GENSPEC="woof"
IPC=50
SPEC=imagenet-${GENSPEC}
EVAL_DIR="/data/mmc_lyxiang/DD/CaO2ori/CaO2"
IMAGENET_PATH="/data/mmc_lyxiang/dataset/ImageNet/train/"
IMAGENET_TRAIN_PATH="/data/mmc_lyxiang/dataset/ImageNet/train/"
IMAGENET_VAL_PATH="/data/mmc_lyxiang/dataset/ImageNet/val/"
N_CLASS=10
CFG=4.0

echo "Using GPU: ${CUDA_VISIBLE_DEVICES}"
echo "Project: ${PROJECT}"



# run sample generation
python sample_mult_label_sel_confusion.py --model DiT-XL/2 --image-size 256 --ckpt "/data/mmc_lyxiang/DD/MinimaxDiffusion/logs/p_x_y_2/000-DiT-XL-2-p_x_y_2/checkpoints/last.pt" \
    --save-dir ./results/dit-distillation/imagenet-10-1000-${PROJECT} --spec $GENSPEC --num-samples $IPC --nclass $N_CLASS --cfg-scale $CFG --confusion-path "/data/mmc_lyxiang/DD/MinimaxDiffusion/results/expert_confusion/expert_confusion_imagenet-woof.pt"

echo "Sample generation finished."

# =====================
# Switch to evaluation directory
# =====================

cd "${EVAL_DIR}"
echo "Switched to evaluation directory: $(pwd)"

# Datasets including: imagenet-woof (N_CLASS=10), imagenet-nette (N_CLASS=10), imagenet-100 (N_CLASS=100), imagenet-1k (N_CLASS=1000)

python main_validate.py --subset "$SPEC" --arch-name "resnet18" --factor 2 \
    --num-crop 5 --mipc 300 --ipc $IPC --stud-name "resnet18" --re-epochs 300 \
    --train-dir $IMAGENET_TRAIN_PATH --val-dir $IMAGENET_VAL_PATH --repeat 3 \
    --syn-data-path "/data/mmc_lyxiang/DD/MinimaxDiffusion/results/dit-distillation/imagenet-10-1000-${PROJECT}"

python main_validate.py --subset "$SPEC" --arch-name "resnet18" --factor 2 \
    --num-crop 5 --mipc 300 --ipc $IPC --stud-name "resnet50" --re-epochs 300 \
    --train-dir $IMAGENET_TRAIN_PATH --val-dir $IMAGENET_VAL_PATH --repeat 3 \
    --syn-data-path "/data/mmc_lyxiang/DD/MinimaxDiffusion/results/dit-distillation/imagenet-10-1000-${PROJECT}"

python main_validate.py --subset "$SPEC" --arch-name "resnet18" --factor 2 \
    --num-crop 5 --mipc 300 --ipc $IPC --stud-name "resnet101" --re-epochs 300 \
    --train-dir $IMAGENET_TRAIN_PATH --val-dir $IMAGENET_VAL_PATH --repeat 3 \
    --syn-data-path "/data/mmc_lyxiang/DD/MinimaxDiffusion/results/dit-distillation/imagenet-10-1000-${PROJECT}"

echo -e "\033[34mMulti students evaluation finished.\033[0m"