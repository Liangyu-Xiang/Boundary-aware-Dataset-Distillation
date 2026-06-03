export CUDA_VISIBLE_DEVICES=$1
PROJECT=$2

echo "Using GPU: ${CUDA_VISIBLE_DEVICES}"
echo "Project: ${PROJECT}"

# run sample generation
python gen_score_sel.py \
  --model DiT-XL/2 \
  --image-size 256 \
  --ckpt /data/mmc_lyxiang/DD/MinimaxDiffusion/logs/baseline/006-DiT-XL-2-baseline/checkpoints/0012000.pt \
  --save-dir ./results/dit-distillation/imagenet-10-1000-${PROJECT} \
  --spec woof \
  --teacher-ckpt /data/mmc_lyxiang/KD/EKD/output/Evidential_Teacher/ResNet18_ImageNet/student_best \
  --vae mse \
  --num-sampling-steps 50 \
  --cfg-scale 2.5 \
  --batch-size 1 \
  --candidates-per-class 500 \
  --core-per-class 50 \
  --topr-factor 5 \
  --mix-prob 0.5 \
  --mix-alpha 0.7 \
  --lambda1 0.5 \
  --lambda2 0.5

# run validation
python train.py -d imagenet --imagenet_dir ./results/dit-distillation/imagenet-10-1000-${PROJECT} /data/mmc_lyxiang/dataset/ImageNet/ \
    -n resnet_ap --nclass 10 --norm_type instance --ipc 100 --tag ${PROJECT} --slct_type random --spec woof
