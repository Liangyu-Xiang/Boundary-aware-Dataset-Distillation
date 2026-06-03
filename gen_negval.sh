export CUDA_VISIBLE_DEVICES=$1
PROJECT=$2
CFG_SCALE=$3

echo "Using GPU: ${CUDA_VISIBLE_DEVICES}"
echo "Project: ${PROJECT}"
echo "CFG Scale: ${CFG_SCALE}"

# run sample generation
python sample_neg_label.py --model DiT-XL/2 --image-size 256 --ckpt /data/mmc_lyxiang/DD/MinimaxDiffusion/logs/run-0/001-DiT-XL-2-minimax/checkpoints/0012000.pt \
    --save-dir ./results/dit-distillation/imagenet-10-1000-${PROJECT} --spec woof --cfg-scale ${CFG_SCALE}

# run validation
python train.py -d imagenet --imagenet_dir ./results/dit-distillation/imagenet-10-1000-${PROJECT} /data/mmc_lyxiang/dataset/ImageNet/ \
    -n resnet_ap --nclass 10 --norm_type instance --ipc 100 --tag ${PROJECT} --slct_type random --spec woof
