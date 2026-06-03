#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Merge multiple ImageNet-style datasets into one unified dataset.
Paths are defined as program hyperparameters (NOT from command line).
"""

import os
import shutil
from collections import defaultdict
from typing import Optional

# =========================
# 🔧 超参数区（你只需要改这里）
# =========================

# 多个数据集路径（每个只包含部分类别）
DATASET_PATHS = [
   '/data/mmc_lyxiang/DD/MinimaxDiffusion/results/dit-distillation/Exp_IPC50/1k/p_x_y_2_Gated+lr1e-4+Steps96000+Weight0.5-1+Gap0.5+Candidate4+WeightAlign+CFG1/p_x_y_2_Gated+lr1e-4+Steps96000+Weight0.5-1+Gap0.5+Candidate4+WeightAlign+CFG1/FROM_757_TO_804/',
   '/data/mmc_lyxiang/DD/MinimaxDiffusion/results/dit-distillation/Exp_IPC50/1k/p_x_y_2_Gated+lr1e-4+Steps96000+Weight0.5-1+Gap0.5+Candidate4+WeightAlign+CFG1/p_x_y_2_Gated+lr1e-4+Steps96000+Weight0.5-1+Gap0.5+Candidate4+WeightAlign+CFG1/PHASE3/',
   '/data/mmc_lyxiang/DD/MinimaxDiffusion/results/dit-distillation/Exp_IPC50/1k/p_x_y_2_Gated+lr1e-4+Steps96000+Weight0.5-1+Gap0.5+Candidate4+WeightAlign+CFG1/p_x_y_2_Gated+lr1e-4+Steps96000+Weight0.5-1+Gap0.5+Candidate4+WeightAlign+CFG1/FROM_853_TO_900/',
   '/data/mmc_lyxiang/DD/MinimaxDiffusion/results/dit-distillation/Exp_IPC50/1k/p_x_y_2_Gated+lr1e-4+Steps96000+Weight0.5-1+Gap0.5+Candidate4+WeightAlign+CFG1/p_x_y_2_Gated+lr1e-4+Steps96000+Weight0.5-1+Gap0.5+Candidate4+WeightAlign+CFG1/FROM_950_TO_999/',
   '/data/mmc_lyxiang/DD/MinimaxDiffusion/results/dit-distillation/Exp_IPC50/1k/p_x_y_2_Gated+lr1e-4+Steps96000+Weight0.5-1+Gap0.5+Candidate4+WeightAlign+CFG1/p_x_y_2_Gated+lr1e-4+Steps96000+Weight0.5-1+Gap0.5+Candidate4+WeightAlign+CFG1/FROM_805_TO_852/',
   '/data/mmc_lyxiang/DD/MinimaxDiffusion/results/dit-distillation/Exp_IPC50/1k/p_x_y_2_Gated+lr1e-4+Steps96000+Weight0.5-1+Gap0.5+Candidate4+WeightAlign+CFG1/p_x_y_2_Gated+lr1e-4+Steps96000+Weight0.5-1+Gap0.5+Candidate4+WeightAlign+CFG1/FROM_901_TO_949/',
   '/data/mmc_lyxiang/DD/MinimaxDiffusion/results/dit-distillation/Exp_IPC50/1k/p_x_y_2_Gated+lr1e-4+Steps96000+Weight0.5-1+Gap0.5+Candidate4+WeightAlign+CFG1/PHASE0/',
   '/data/mmc_lyxiang/DD/MinimaxDiffusion/results/dit-distillation/Exp_IPC50/1k/p_x_y_2_Gated+lr1e-4+Steps96000+Weight0.5-1+Gap0.5+Candidate4+WeightAlign+CFG1/PHASE1/',
   '/data/mmc_lyxiang/DD/MinimaxDiffusion/results/dit-distillation/Exp_IPC50/1k/p_x_y_2_Gated+lr1e-4+Steps96000+Weight0.5-1+Gap0.5+Candidate4+WeightAlign+CFG1/FROM383_TO421/',
   '/data/mmc_lyxiang/DD/MinimaxDiffusion/results/dit-distillation/Exp_IPC50/1k/p_x_y_2_Gated+lr1e-4+Steps96000+Weight0.5-1+Gap0.5+Candidate4+WeightAlign+CFG1/FROM422_TO460/',
]

# 融合后的输出路径
OUTPUT_ROOT = '/data/mmc_lyxiang/DD/MinimaxDiffusion/results/dit-distillation/Exp_IPC50/1k/p_x_y_2_Gated+lr1e-4+Steps96000+Weight0.5-1+Gap0.5+Candidate4+WeightAlign+CFG1+Merged/'

# 每个类别最多样本数（IPC），None 表示不限制
MAX_PER_CLASS: Optional[int] = 50

# 文件合并方式：copy | symlink | hardlink
LINK_MODE = 'symlink'   # 强烈推荐 symlink（省空间）

# 只允许 ImageNet 风格类别（nxxxxxxx）
ONLY_IMAGENET_STYLE = True

# 是否打印详细日志
VERBOSE = True


# =========================
# 内部常量
# =========================
IMG_EXTENSIONS = ('.jpg', '.jpeg', '.png', '.bmp', '.webp')


# =========================
# 工具函数
# =========================
def is_image(fname: str) -> bool:
    return fname.lower().endswith(IMG_EXTENSIONS)


def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


def link_file(src: str, dst: str, mode: str):
    if mode == "copy":
        shutil.copy2(src, dst)
    elif mode == "symlink":
        os.symlink(src, dst)
    elif mode == "hardlink":
        os.link(src, dst)
    else:
        raise ValueError(f"Unsupported link mode: {mode}")


# =========================
# 融合主逻辑
# =========================
def merge_datasets(
    dataset_paths,
    output_root,
    max_per_class,
    link_mode,
    only_imagenet_style=True,
    verbose=True,
):
    ensure_dir(output_root)

    class_count = defaultdict(int)   # cls -> total samples

    for root in dataset_paths:
        root = os.path.normpath(root)
        if not os.path.isdir(root):
            print(f"[WARN] Skip non-existent path: {root}")
            continue

        if verbose:
            print(f"[INFO] Scanning dataset: {root}")

        for cls in os.listdir(root):
            if only_imagenet_style and not cls.startswith('n'):
                continue

            cls_dir = os.path.join(root, cls)
            if not os.path.isdir(cls_dir):
                continue

            out_cls_dir = os.path.join(output_root, cls)
            ensure_dir(out_cls_dir)

            for fname in os.listdir(cls_dir):
                if not is_image(fname):
                    continue

                if max_per_class is not None and class_count[cls] >= max_per_class:
                    break

                src = os.path.join(cls_dir, fname)

                base, ext = os.path.splitext(fname)
                new_name = f"{base}_m{class_count[cls]}{ext}"
                dst = os.path.join(out_cls_dir, new_name)

                if os.path.exists(dst):
                    continue

                link_file(src, dst, link_mode)
                class_count[cls] += 1

    # =========================
    # 输出统计
    # =========================
    print("\n========== MERGE SUMMARY ==========")
    total_images = 0
    for cls in sorted(class_count.keys()):
        cnt = class_count[cls]
        total_images += cnt
        print(f"{cls}: {cnt}")

    print("----------------------------------")
    print(f"Total classes: {len(class_count)}")
    print(f"Total images : {total_images}")
    print(f"Output root  : {output_root}")


# =========================
# 程序入口
# =========================
if __name__ == "__main__":
    merge_datasets(
        dataset_paths=DATASET_PATHS,
        output_root=OUTPUT_ROOT,
        max_per_class=MAX_PER_CLASS,
        link_mode=LINK_MODE,
        only_imagenet_style=ONLY_IMAGENET_STYLE,
        verbose=VERBOSE,
    )
