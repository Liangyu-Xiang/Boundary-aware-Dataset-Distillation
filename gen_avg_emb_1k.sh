#!/bin/bash
export CUDA_VISIBLE_DEVICES=$1
PROJECT=$2
GENSPEC=1k
IPC=10
SPEC=imagenet-${GENSPEC}
EVAL_DIR="/data/mmc_lyxiang/DD/CaO2ori/CaO2"
IMAGENET_PATH="/data/mmc_lyxiang/dataset/ImageNet/train/"
IMAGENET_TRAIN_PATH="/data/mmc_lyxiang/dataset/ImageNet/train/"
IMAGENET_VAL_PATH="/data/mmc_lyxiang/dataset/ImageNet/val/"
N_CLASS=1000
CFG=4.0

echo "Using GPU: ${CUDA_VISIBLE_DEVICES}"
echo "Project: ${PROJECT}"


# run sample generation
# for ((PHASE=1; PHASE<NUM_PHASE; PHASE++)); do
python sample_mult_label_sel_confusion_1k.py --model DiT-XL/2 --image-size 256 --ckpt "/data/mmc_lyxiang/DD/MinimaxDiffusion/logs/1K_p_x_y_2+Batch8+lr1e-4+Epoch8/Epoch0.pt" \
    --save-dir ./results/dit-distillation/Exp_IPC${IPC}/${GENSPEC}/${PROJECT} --spec $GENSPEC --num-samples $IPC --nclass $N_CLASS --cfg-scale $CFG --class-from 0 --class-to 1000
# done

echo "Sample generation finished."

# --confusion-path "/data/mmc_lyxiang/DD/MinimaxDiffusion/results/expert_confusion/expert_confusion_imagenet-woof.pt"

# =====================
# Switch to evaluation directory
# =====================

# cd "${EVAL_DIR}"
# echo "Switched to evaluation directory: $(pwd)"

# # Datasets including: imagenet-woof (N_CLASS=10), imagenet-nette (N_CLASS=10), imagenet-100 (N_CLASS=100), imagenet-1k (N_CLASS=1000)

# python main_validate_random.py --subset "$SPEC" --arch-name "resnet18" --factor 2 \
#     --num-crop 5 --mipc 300 --ipc $IPC --stud-name "resnet18" --re-epochs 300 \
#     --train-dir $IMAGENET_TRAIN_PATH --val-dir $IMAGENET_VAL_PATH --repeat 5 \
#     --syn-data-path "/data/mmc_lyxiang/DD/MinimaxDiffusion/results/dit-distillation/Exp_IPC${IPC}/${GENSPEC}/${PROJECT}"

# echo "Evaluation finished."

# run validation
# python train_kd.py -d imagenet --imagenet_dir ./results/dit-distillation/imagenet-10-1000-${PROJECT} /data/mmc_lyxiang/dataset/ImageNet/ \
#     -n resnet_ap --nclass 10 --norm_type instance --ipc 50 --tag ${PROJECT} --slct_type random --spec woof