#!/bin/bash
export CUDA_VISIBLE_DEVICES=$1
GENSPEC="woof"
IPC=50
SPEC=imagenet-${GENSPEC}
EVAL_DIR="/data/mmc_lyxiang/DD/CaO2ori/CaO2"
IMAGENET_PATH="/data/mmc_lyxiang/dataset/ImageNet/train/"
IMAGENET_TRAIN_PATH="/data/mmc_lyxiang/dataset/ImageNet/train/"
IMAGENET_VAL_PATH="/data/mmc_lyxiang/dataset/ImageNet/val/"
N_CLASS=10

echo "Using GPU: ${CUDA_VISIBLE_DEVICES}"
cd "${EVAL_DIR}"
echo "Switched to evaluation directory: $(pwd)"

# Datasets including: imagenet-woof (N_CLASS=10), imagenet-nette (N_CLASS=10), imagenet-100 (N_CLASS=100), imagenet-1k (N_CLASS=1000)

python main_validate_random.py --subset "$SPEC" --arch-name "resnet18" --factor 2 \
    --num-crop 5 --mipc 300 --ipc $IPC --stud-name "resnet18" --re-epochs 300 \
    --train-dir $IMAGENET_TRAIN_PATH --val-dir $IMAGENET_VAL_PATH --repeat 10 \
    --syn-data-path "/data/mmc_lyxiang/DD/MinimaxDiffusion/results/StageCFG/Exp_IPC50/woof/StageCFG_Default/"

echo "Evaluation finished."

# run validation
# python train_kd.py -d imagenet --imagenet_dir ./results/dit-distillation/imagenet-10-1000-${PROJECT} /data/mmc_lyxiang/dataset/ImageNet/ \
#     -n resnet_ap --nclass 10 --norm_type instance --ipc 50 --tag ${PROJECT} --slct_type random --spec woof
