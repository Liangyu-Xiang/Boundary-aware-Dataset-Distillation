export CUDA_VISIBLE_DEVICES=$1
PROJECT=$2

echo "Using GPU: ${CUDA_VISIBLE_DEVICES}"
echo "Project: ${PROJECT}"

# run sample generation
python sample_film.py --model DiT-XL/2 --image-size 256 --ckpt "/data/mmc_lyxiang/DD/MinimaxDiffusion/logs/film_bound_dis/000-DiT-XL-2-film_bound_dis/checkpoints/0003000.pt" \
    --save-dir ./results/dit-distillation/imagenet-10-1000-${PROJECT} --spec woof --nclass 10 --num-samples 50 --cfg-scale 4.0