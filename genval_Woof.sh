export CUDA_VISIBLE_DEVICES=$1
GEN_DIR="/data/mmc_lyxiang/DD/MinimaxDiffusion"
EVAL_DIR="/data/mmc_lyxiang/DD/CaO2ori/CaO2"
IMAGENET_PATH="/data/mmc_lyxiang/dataset/ImageNet/train/"
IMAGENET_TRAIN_PATH="/data/mmc_lyxiang/dataset/ImageNet/train/"
IMAGENET_VAL_PATH="/data/mmc_lyxiang/dataset/ImageNet/val/"
N_CLASS=10
IPC_LIST=(50 20 70 100 10)
STUDENTS=("resnet18" "resnet50" "resnet101")

echo "Using GPU: ${CUDA_VISIBLE_DEVICES}"

# run sample generation
for IPC in "${IPC_LIST[@]}"; do

    SAVE_DIR="${GEN_DIR}/results/dit-distillation/Minimax/imagenet-10-1000-Woof-IPC${IPC}"

    # cd "${GEN_DIR}" || exit 1

    # if [ -d "${SAVE_DIR}" ] && [ "$(ls -A "${SAVE_DIR}")" ]; then
    #     echo -e "\033[33m[IPC=${IPC}] Found existing samples in ${SAVE_DIR}, skip generation.\033[0m"
    # else
    #     echo -e "\033[1;34m[IPC=${IPC}] Generating synthetic samples...\033[0m"

    #     python sample.py --model DiT-XL/2 --image-size 256 --ckpt "/data/mmc_lyxiang/DD/MinimaxDiffusion/logs/p_x_y_2/000-DiT-XL-2-p_x_y_2/checkpoints/last.pt" \
    #     --save-dir "${SAVE_DIR}" --spec woof --nclass 10 --num-samples ${IPC}
    # fi
    

    cd "${EVAL_DIR}" || exit 1

    for STUD in "${STUDENTS[@]}"; do
        echo -e "\033[1;36m[IPC=${IPC}] Evaluating student: ${STUD}\033[0m"

        python main_validate_random.py --subset "imagenet-woof" --arch-name "resnet18" --factor 2 \
            --num-crop 5 --mipc 300 --ipc ${IPC} --stud-name ${STUD} --re-epochs 300 \
            --train-dir $IMAGENET_TRAIN_PATH --val-dir $IMAGENET_VAL_PATH --repeat 10 \
            --syn-data-path "${SAVE_DIR}"
    
    done

    echo -e "\033[1;32m[IPC=${IPC}] Evaluation finished.\033[0m"

done

echo "Evaluation finished."