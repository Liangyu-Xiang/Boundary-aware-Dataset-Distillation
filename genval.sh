export CUDA_VISIBLE_DEVICES=$1
PROJECT=$2

echo "Using GPU: ${CUDA_VISIBLE_DEVICES}"
echo "Project: ${PROJECT}"

# run sample generation
python sample.py --model DiT-XL/2 --image-size 256 --ckpt "/data/mmc_lyxiang/DD/MinimaxDiffusion/logs/p_x_y_2/000-DiT-XL-2-p_x_y_2/checkpoints/last.pt" \
    --save-dir ./results/dit-distillation/imagenet-10-1000-${PROJECT} --spec woof --nclass 10 --num-samples 10

# run validation
python train.py -d imagenet --imagenet_dir ./results/dit-distillation/imagenet-10-1000-${PROJECT} /data/mmc_lyxiang/dataset/ImageNet/ \
    -n resnet_ap --nclass 100 --norm_type instance --ipc 50 --tag ${PROJECT} --slct_type random --spec woof