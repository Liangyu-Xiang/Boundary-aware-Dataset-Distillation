

python mix_datasets.py \
  --dataset-a /data/mmc_lyxiang/DD/MinimaxDiffusion/results/dit-distillation/Exp_IPC25/nette/p_x_y_2+Weight0.8+Candidate4+WeightAlign+CFG4/ \
  --dataset-b /data/mmc_lyxiang/dataset/CaO2/CaO2_Distilled_Data/nette-ipc50/ \
  --dst-root /data/mmc_lyxiang/DD/MinimaxDiffusion/results/dit-distillation/Exp_IPC50/nette/Mix0.5CaO2+p_x_y_2+Weight0.8+Candidate4+WeightAlign+CFG4/ \
  --total-per-class 50 \
  --ratio-a 0.5 \
  --seed 42