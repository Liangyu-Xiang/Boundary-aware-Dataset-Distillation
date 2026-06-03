export CUDA_VISIBLE_DEVICES=$1
PROJECT=$2

if [ -z "$1" ]; then
    echo "Usage: bash finetune.sh <GPU_ID>"
    exit 1
fi

IPC=50

MASTER_PORT=$((25670 + $1))
echo "Using GPU: ${CUDA_VISIBLE_DEVICES}"

# fine-tune the diffusion model
torchrun --nnode=1 --master_port=${MASTER_PORT} train_dit.py --model DiT-XL/2 \
     --data-path /data/mmc_lyxiang/dataset/ImageNet/train/ --ckpt pretrained_models/DiT-XL-2-256x256.pt \
     --global-batch-size 8 --tag ${PROJECT} --ckpt-every 12000 --log-every 1500 --epochs 8 \
     --condense --finetune-ipc -1 --results-dir ./logs/${PROJECT} --spec woof --condense

# run sample generation
python sample.py --model DiT-XL/2 --image-size 256 --ckpt ./logs/${PROJECT}/000-DiT-XL-2-${PROJECT}/checkpoints/0012000.pt \
    --save-dir ./results/dit-distillation/imagenet-10-1000-${PROJECT} --spec woof --num-samples 50

# run validation
python train.py -d imagenet --imagenet_dir ./results/dit-distillation/imagenet-10-1000-${PROJECT} /data/mmc_lyxiang/dataset/ImageNet/ \
    -n resnet_ap --nclass 10 --norm_type instance --ipc 100 --tag ${PROJECT} --slct_type random --spec woof
