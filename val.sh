export CUDA_VISIBLE_DEVICES=$1
PROJECT=$2

echo "Using GPU: ${CUDA_VISIBLE_DEVICES}"
echo "Project: ${PROJECT}"

# run validation
# python train_EDLexpert.py -d imagenet --imagenet_dir /data/mmc_lyxiang/dataset/ImageNet/train/ /data/mmc_lyxiang/dataset/ImageNet/ \
#     -n resnet --nclass 10 --norm_type instance --ipc 50 --tag ${PROJECT} --slct_type random --spec woof --depth 18

python train.py -d imagenet --imagenet_dir //data/mmc_lyxiang/DD/MinimaxDiffusion/results/Rebuttal/Exp_IPC50/woof/rebuttal_ablation/0.7-1.0/ /data/mmc_lyxiang/dataset/ImageNet/ \
    -n resnet --depth 18 --nclass 10 --norm_type instance --ipc 50 --tag ${PROJECT} --slct_type random --spec woof --mixup vanilla
# python train.py -d imagenet --imagenet_dir /data/mmc_lyxiang/dataset/ImageNet/ \
#     -n resnet --depth 18 --nclass 10 --norm_type instance --tag ${PROJECT} --slct_type random --spec woof --save_ckpt True
