#!/bin/bash
export CUDA_VISIBLE_DEVICES=$1
PROJECT=$2
GENSPEC=woof
IPC=50
SPEC=imagenet-${GENSPEC}
EVAL_DIR="/data/mmc_lyxiang/DD/CaO2ori/CaO2"
IMAGENET_PATH="/data/mmc_lyxiang/dataset/ImageNet/train/"
IMAGENET_TRAIN_PATH="/data/mmc_lyxiang/dataset/ImageNet/train/"
IMAGENET_VAL_PATH="/data/mmc_lyxiang/dataset/ImageNet/val/"
N_CLASS=10
CFG=1.0
CKPT="/data/mmc_lyxiang/DD/MinimaxDiffusion/logs/p_x_y_2_Gated+Epoch16+Temp2/000-DiT-XL-2-p_x_y_2_Gated+Epoch16+Temp2/checkpoints/last.pt"
# CKPT="/data/mmc_lyxiang/DD/MinimaxDiffusion/logs/dif_denoise_pairwise_alpha30_lambda_ratio0.003/000-DiT-XL-2-dif_denoise_pairwise_alpha30_lambda_ratio0.003/checkpoints/epoch_003.pt"


echo "Using GPU: ${CUDA_VISIBLE_DEVICES}"
echo "Project: ${PROJECT}"


# run sample generation
python sample_mult_label_sel.py --model DiT-XL/2 --image-size 256 --ckpt "$CKPT" \
    --save-dir ./results/OCD/Exp_IPC${IPC}/${GENSPEC}/${PROJECT} --spec $GENSPEC --num-samples $IPC --nclass $N_CLASS --cfg-scale $CFG --disable-stage-cfg --disable-selection

echo "Sample generation finished."

# --confusion-path "/data/mmc_lyxiang/DD/MinimaxDiffusion/results/expert_confusion/expert_confusion_imagenet-woof.pt"

# run evaluation
cd "${EVAL_DIR}"
echo "Switched to evaluation directory: $(pwd)"

python main_validate_random.py --subset "$SPEC" --arch-name "resnet18" --factor 2 \
    --num-crop 5 --mipc 300 --ipc $IPC --stud-name "resnet18" --re-epochs 300 \
    --train-dir $IMAGENET_TRAIN_PATH --val-dir $IMAGENET_VAL_PATH --repeat 3 \
    --syn-data-path "/data/mmc_lyxiang/DD/MinimaxDiffusion/results/OCD/Exp_IPC${IPC}/${GENSPEC}/${PROJECT}"

echo "Evaluation finished."

# run validation
# python train_kd.py -d imagenet --imagenet_dir ./results/dit-distillation/imagenet-10-1000-${PROJECT} /data/mmc_lyxiang/dataset/ImageNet/ \
#     -n resnet_ap --nclass 10 --norm_type instance --ipc 50 --tag ${PROJECT} --slct_type random --spec woof
