export CUDA_VISIBLE_DEVICES=4


# run sample generation
python random_sample_woof.py --imagenet-dir /data/mmc_lyxiang/dataset/ImageNet/train/ \
    --dst-dir /data/mmc_lyxiang/DD/MinimaxDiffusion/results/dit-distillation/imagenet-10-1000-random_sample_ipc100+seed42/ --ipc 100 --seed 42

