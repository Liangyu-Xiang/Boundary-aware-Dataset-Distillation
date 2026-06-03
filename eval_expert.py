#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Evaluate an expert classifier (e.g. ResNet18) on the TRAINING set and compute:

1) Soft confusion:
   confusion[y, k] = E_{x | y} [ p_expert(k | x) ]

2) Class-wise weights:
   weights[y, k] = confusion[y, k] / sum_{j != y} confusion[y, j]

The script saves confusion + weights to a .pt file and prints
hardest-to-distinguish class rankings for each class.
"""

import os
import argparse
from typing import List, Dict, Tuple
import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import transforms

from misc.utils import load_model
from data import ImageFolder


# ------------------------------------------------------------
# Core computation
# ------------------------------------------------------------

@torch.no_grad()
def compute_soft_confusion(
    model: torch.nn.Module,
    loader: DataLoader,
    num_classes: int,
    device: torch.device,
):
    """
    Returns:
        confusion: [C, C] tensor
        counts:    [C] tensor (number of samples per class)
    """
    sum_probs = torch.zeros(num_classes, num_classes, device=device)
    counts = torch.zeros(num_classes, device=device)

    model.eval()
    for batch in loader:
        # Compatible with your ImageFolder return format
        # training script: (x_dit, x_expert, ry, y)
        if len(batch) == 4:
            _, x_expert, ry, _ = batch
        else:
            raise RuntimeError("Unexpected dataset return format.")

        x_expert = x_expert.to(device, non_blocking=True)
        ry = ry.to(device, non_blocking=True).long()

        logits = model(x_expert)
        probs = torch.softmax(logits, dim=1)

        for y in range(num_classes):
            mask = (ry == y)
            if mask.any():
                sum_probs[y] += probs[mask].sum(dim=0)
                counts[y] += mask.sum()

    confusion = torch.zeros_like(sum_probs)
    for y in range(num_classes):
        if counts[y] > 0:
            confusion[y] = sum_probs[y] / counts[y]

    return confusion.cpu(), counts.cpu()


def confusion_to_weights(confusion: torch.Tensor) -> torch.Tensor:
    """
    Convert confusion to class-wise weights.

    weights[y, k] indicates how hard it is to distinguish k from y.
    """
    weights = confusion.clone()
    idx = torch.arange(weights.shape[0])
    weights[idx, idx] = 0.0
    row_sum = weights.sum(dim=1, keepdim=True)
    weights = torch.where(row_sum > 0, weights / row_sum, weights)
    return weights


def rank_hardest_classes(
    weights: torch.Tensor,
    class_names: List[str],
    topk: int = 9,
) -> Dict[str, List[Tuple[str, float]]]:
    """
    For each class y, rank other classes by difficulty.
    """
    C = weights.shape[0]
    results = {}
    for y in range(C):
        row = weights[y].clone()
        row[y] = 0.0
        order = torch.argsort(row, descending=True)
        results[class_names[y]] = [
            (class_names[k], float(row[k]))
            for k in order[:topk]
        ]
    return results

def center_crop_arr(pil_image, image_size):
    """
    Center cropping implementation from ADM.
    https://github.com/openai/guided-diffusion/blob/8fb3ad9197f16bbc40620447b2742e13458d2831/guided_diffusion/image_datasets.py#L126
    """
    while min(*pil_image.size) >= 2 * image_size:
        pil_image = pil_image.resize(
            tuple(x // 2 for x in pil_image.size), resample=Image.BOX
        )

    scale = image_size / min(*pil_image.size)
    pil_image = pil_image.resize(
        tuple(round(x * scale) for x in pil_image.size), resample=Image.BICUBIC
    )

    arr = np.array(pil_image)
    crop_y = (arr.shape[0] - image_size) // 2
    crop_x = (arr.shape[1] - image_size) // 2
    return Image.fromarray(arr[crop_y: crop_y + image_size, crop_x: crop_x + image_size])


# ------------------------------------------------------------
# Main
# ------------------------------------------------------------

def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # ------------------------------
    # Load expert model
    # ------------------------------
    expert_model = load_model(
        model_name="resnet18",
        dataset=args.spec,
        pretrained=True,
        classes=range(args.nclass),
    ).to(device)
    expert_model.eval()
    for p in expert_model.parameters():
        p.requires_grad = False

    # ------------------------------
    # Expert preprocessing
    # ------------------------------
    expert_transform = transforms.Compose(
        [
            transforms.Resize(224 // 7 * 8, antialias=True),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
        ]
    )
    transform = transforms.Compose([
        transforms.Lambda(lambda pil_image: center_crop_arr(pil_image, args.image_size)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], inplace=True)
    ])
    # ------------------------------
    # Dataset / Loader
    # ------------------------------
    dataset = ImageFolder(
        args.data_path,
        transform=transform,
        expert_transform=expert_transform,
        nclass=args.nclass,
        ipc=args.finetune_ipc,
        spec=args.spec,
        phase=args.phase,
        seed=args.seed,
        return_origin=True,
    )

    if hasattr(dataset, "class_names") and dataset.class_names:
        class_names = list(dataset.class_names)
    else:
        class_names = [str(i) for i in range(args.nclass)]

    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=False,
    )

    print(f"Dataset size: {len(dataset)}")

    # ------------------------------
    # Compute confusion & weights
    # ------------------------------
    confusion, counts = compute_soft_confusion(
        expert_model, loader, args.nclass, device
    )
    weights = confusion_to_weights(confusion)

    # ------------------------------
    # Save results
    # ------------------------------
    os.makedirs(os.path.dirname(args.save_path), exist_ok=True)
    torch.save(
        {
            "confusion": confusion,
            "weights": weights,
            "counts": counts,
            "class_names": class_names,
            "args": vars(args),
        },
        args.save_path,
    )

    print(f"\n✅ Saved expert confusion & weights to:\n{args.save_path}")

    # ------------------------------
    # Print ranking
    # ------------------------------
    hardest = rank_hardest_classes(
        weights, class_names, topk=min(args.nclass - 1, args.topk)
    )

    print("\nHardest-to-distinguish classes (by expert):")
    for y, pairs in hardest.items():
        msg = ", ".join([f"{k}:{w:.4f}" for k, w in pairs])
        print(f"  [{y}] -> {msg}")


# ------------------------------------------------------------
# CLI
# ------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--data-path", type=str, required=True,
                        help="Training dataset root")
    parser.add_argument("--spec", type=str, default="none",
                        help="Subset spec (e.g. imagenet-woof)")
    parser.add_argument("--nclass", type=int, default=10)
    parser.add_argument("--finetune-ipc", type=int, default=1000)
    parser.add_argument("--phase", type=int, default=0)
    parser.add_argument("--image-size", type=int, choices=[256, 512], default=256)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=0)

    parser.add_argument("--save-path", type=str, required=True)
    parser.add_argument("--topk", type=int, default=9)

    args = parser.parse_args()
    main(args)
