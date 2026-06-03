export CUDA_VISIBLE_DEVICES=$1
PROJECT=$2

echo "Using GPU: ${CUDA_VISIBLE_DEVICES}"
echo "Project: ${PROJECT}"

# run sample generation
python sample.py --model DiT-XL/2 --image-size 256 --ckpt "/data/mmc_lyxiang/DD/MinimaxDiffusion/logs/Bayes_BayesAlpha1000_LambdaBayes0.001/000-DiT-XL-2-Bayes_BayesAlpha1000_LambdaBayes0.001/checkpoints/epoch_007.pt" \
    --save-dir ./results/FT_TEST/${PROJECT} --spec woof --nclass 10 --num-samples 10 --cfg-scale 4.0