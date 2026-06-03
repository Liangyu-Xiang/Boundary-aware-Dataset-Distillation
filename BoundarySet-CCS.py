#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Boundary-aware coreset selection
Reproduction of the core selection procedure in:
"Mind the Boundary: Coreset Selection via Reconstructing the Decision Boundary"

Supports:
    1) BoundarySet
    2) BoundarySet-CCS

Main steps:
    1. Load a pretrained classifier (e.g., ResNet18)
    2. Estimate distance-to-boundary for each sample using PGD-like updates
    3. Select a coreset according to:
        - boundaryset: choose smallest distances
        - boundaryset_ccs: coverage-centric sampling over distance bins
    4. Save selected indices / paths / distances

Example:
    python boundary_coreset_select.py \
        --dataset cifar10 \
        --data-root ./data \
        --checkpoint ./resnet18_cifar10.pth \
        --num-classes 10 \
        --per-class-count 50 \
        --method boundaryset_ccs \
        --alpha 0.002 \
        --max-step 10 \
        --batch-size 128 \
        --num-workers 4 \
        --output-dir ./boundaryset_out \
        --export-dir ./boundaryset_samples

For ImageFolder:
    python boundary_coreset_select.py \
        --dataset imagefolder \
        --data-root /path/to/train \
        --checkpoint ./resnet18_imagenet_subset.pth \
        --num-classes 1000 \
        --per-class-count 50 \
        --method boundaryset \
        --alpha 0.0001 \
        --max-step 50 \
        --batch-size 64 \
        --num-workers 8 \
        --output-dir ./boundaryset_out \
        --export-dir ./boundaryset_samples
"""

import os
import json
import math
import random
import argparse
import shutil
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Tuple, Optional

import numpy as np

import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, transforms, models
from PIL import Image


# =========================
# Utilities
# =========================

def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def save_json(obj: Dict[str, Any], path: str) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)


def robust_load_checkpoint(model: nn.Module, ckpt_path: str, device: torch.device) -> None:
    """
    Robustly load checkpoints saved in different styles:
        - pure state_dict
        - {'state_dict': ...}
        - {'model': ...}
        - {'net': ...}
        - DDP keys with 'module.'
    """
    try:
        ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    except TypeError:
        ckpt = torch.load(ckpt_path, map_location=device)

    if isinstance(ckpt, dict):
        if "state_dict" in ckpt:
            state_dict = ckpt["state_dict"]
        elif "model" in ckpt:
            state_dict = ckpt["model"]
        elif "net" in ckpt:
            state_dict = ckpt["net"]
        else:
            state_dict = ckpt
    else:
        state_dict = ckpt

    new_state_dict = {}
    for k, v in state_dict.items():
        if k.startswith("module."):
            k = k[len("module."):]
        new_state_dict[k] = v

    missing, unexpected = model.load_state_dict(new_state_dict, strict=False)
    print(f"[Checkpoint] Loaded from: {ckpt_path}")
    if missing:
        print(f"[Checkpoint] Missing keys ({len(missing)}): {missing[:20]}")
    if unexpected:
        print(f"[Checkpoint] Unexpected keys ({len(unexpected)}): {unexpected[:20]}")


def forward_logits(model: nn.Module, x: torch.Tensor) -> torch.Tensor:
    """
    Normalize model outputs to logits for compatibility with models returning (logits, aux).
    """
    out = model(x)
    if isinstance(out, (tuple, list)):
        return out[0]
    if isinstance(out, dict) and "logits" in out:
        return out["logits"]
    return out


# =========================
# Dataset wrappers
# =========================

class IndexedDataset(Dataset):
    """
    Wrap a dataset to return:
        image, label, index, meta
    meta:
        - for ImageFolder: original file path
        - for CIFAR: None
    """
    def __init__(self, base_dataset: Dataset, dataset_type: str):
        self.base_dataset = base_dataset
        self.dataset_type = dataset_type

    def __len__(self) -> int:
        return len(self.base_dataset)

    def __getitem__(self, idx: int):
        item = self.base_dataset[idx]
        if isinstance(item, (tuple, list)) and len(item) >= 2:
            img, label = item[0], item[1]
        else:
            raise ValueError("Dataset must return (img, label, ...) per sample.")
        meta = None

        if self.dataset_type == "imagefolder":
            # ImageFolder stores samples as [(path, class_idx), ...]
            path, _ = self.base_dataset.samples[idx]
            meta = path

        return img, label, idx, meta


def build_dataset(args) -> IndexedDataset:
    """
    Build train dataset for selection.
    """
    if args.dataset.lower() in ["cifar10", "cifar100"]:
        # Use standard ImageNet-style normalization for ResNet18 if your model was trained this way.
        # If your own training used different normalization, modify here accordingly.
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(
                mean=args.mean,
                std=args.std
            )
        ])

        if args.dataset.lower() == "cifar10":
            base = datasets.CIFAR10(
                root=args.data_root,
                train=True,
                download=args.download,
                transform=transform
            )
        else:
            base = datasets.CIFAR100(
                root=args.data_root,
                train=True,
                download=args.download,
                transform=transform
            )

        return IndexedDataset(base, dataset_type=args.dataset.lower())

    elif args.dataset.lower() == "imagefolder":
        transform = transforms.Compose([
            transforms.Resize(args.resize_size) if args.resize_size > 0 else transforms.Lambda(lambda x: x),
            transforms.CenterCrop(args.crop_size) if args.crop_size > 0 else transforms.Lambda(lambda x: x),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=args.mean,
                std=args.std
            )
        ])
        base = datasets.ImageFolder(root=args.data_root, transform=transform)
        return IndexedDataset(base, dataset_type="imagefolder")

    else:
        raise ValueError(f"Unsupported dataset: {args.dataset}")


# =========================
# Model
# =========================

def build_model(num_classes: int) -> nn.Module:
    model = models.resnet18(weights=None)
    in_features = model.fc.in_features
    model.fc = nn.Linear(in_features, num_classes)
    return model


# =========================
# Distance-to-boundary
# =========================

@torch.no_grad()
def initial_prediction(model: nn.Module, x: torch.Tensor) -> torch.Tensor:
    logits = forward_logits(model, x)
    pred = logits.argmax(dim=1)
    return pred


def compute_boundary_distances(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    alpha: float,
    max_step: int,
    clamp_min: Optional[float] = None,
    clamp_max: Optional[float] = None,
    use_sign: bool = True,
) -> Tuple[np.ndarray, List[Optional[str]], np.ndarray]:
    """
    For each sample (x, y), estimate the smallest number of PGD-like steps needed
    to cross the decision boundary.

    Distance d(x, y):
        - smallest k in [0, max_step] such that prediction != y after k updates
        - if never crossed, distance = max_step

    Returns:
        distances: np.ndarray, shape [N]
        metas: list of meta info (file path for ImageFolder, else None)
        labels: np.ndarray, shape [N]
    """
    model.eval()
    criterion = nn.CrossEntropyLoss(reduction="mean")

    dataset_len = len(loader.dataset)
    distances = np.full(dataset_len, fill_value=max_step, dtype=np.int32)
    metas: List[Optional[str]] = [None for _ in range(dataset_len)]
    labels_arr = np.full(dataset_len, fill_value=-1, dtype=np.int32)

    for batch_idx, batch in enumerate(loader):
        x, y, indices, meta = batch
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        indices = indices.to(device, non_blocking=True)

        # record meta / label
        for bi, idx in enumerate(indices.tolist()):
            labels_arr[idx] = int(y[bi].item())
            metas[idx] = meta[bi] if isinstance(meta, list) else None

        # Start from clean sample
        x_adv = x.detach().clone()

        # k = 0: if already misclassified, distance = 0
        with torch.no_grad():
            pred0 = forward_logits(model, x_adv).argmax(dim=1)
        crossed = (pred0 != y)
        if crossed.any():
            crossed_idx = indices[crossed.cpu()].tolist()
            for idx in crossed_idx:
                distances[idx] = 0

        active_mask = ~crossed

        for k in range(max_step):
            if active_mask.sum().item() == 0:
                break

            x_active = x_adv[active_mask].detach().clone().requires_grad_(True)
            y_active = y[active_mask]
            idx_active = indices[active_mask]

            logits = forward_logits(model, x_active)
            loss = criterion(logits, y_active)
            grad = torch.autograd.grad(loss, x_active, only_inputs=True)[0]

            with torch.no_grad():
                if use_sign:
                    x_next = x_active + alpha * grad.sign()
                else:
                    grad_norm = grad.view(grad.size(0), -1).norm(dim=1, keepdim=True).clamp(min=1e-12)
                    grad_norm = grad_norm.view(-1, 1, 1, 1)
                    x_next = x_active + alpha * grad / grad_norm

                if clamp_min is not None or clamp_max is not None:
                    x_next = torch.clamp(x_next, min=clamp_min, max=clamp_max)

                logits_next = forward_logits(model, x_next)
                pred_next = logits_next.argmax(dim=1)
                newly_crossed = (pred_next != y_active)

                # If after this update they cross the boundary,
                # assign distance = k+1
                if newly_crossed.any():
                    selected_indices = idx_active[newly_crossed.cpu()].tolist()
                    for idx in selected_indices:
                        distances[idx] = min(distances[idx], k + 1)

                # update x_adv only for still-active samples
                active_positions = torch.where(active_mask)[0]
                x_adv[active_positions] = x_next.detach()

                # update active mask
                still_active_local = ~newly_crossed
                new_active_mask = torch.zeros_like(active_mask)
                new_active_mask[active_positions[still_active_local]] = True
                active_mask = new_active_mask

        if (batch_idx + 1) % 20 == 0 or (batch_idx + 1) == len(loader):
            done = min((batch_idx + 1) * loader.batch_size, dataset_len)
            print(f"[Distance] {done}/{dataset_len} processed")

    return distances, metas, labels_arr


# =========================
# Selection
# =========================

def select_boundaryset(
    distances: np.ndarray,
    selection_ratio: float,
    seed: int,
) -> np.ndarray:
    """
    Select samples with smallest distances.
    """
    n = len(distances)
    num_select = int(round(n * selection_ratio))
    num_select = max(1, min(num_select, n))

    rng = np.random.default_rng(seed)
    # Tie-breaking by random permutation first
    perm = rng.permutation(n)
    perm_dist = distances[perm]
    order_in_perm = np.argsort(perm_dist, kind="stable")
    selected = perm[order_in_perm[:num_select]]
    return np.sort(selected)


def select_boundaryset_count(
    distances: np.ndarray,
    count: int,
    seed: int,
) -> np.ndarray:
    n = len(distances)
    count = max(1, min(count, n))
    rng = np.random.default_rng(seed)
    perm = rng.permutation(n)
    perm_dist = distances[perm]
    order_in_perm = np.argsort(perm_dist, kind="stable")
    selected = perm[order_in_perm[:count]]
    return np.sort(selected)


def select_boundaryset_ccs(
    distances: np.ndarray,
    selection_ratio: float,
    max_step: int,
    seed: int,
) -> np.ndarray:
    """
    Coverage-centric sampling over distance bins.

    Paper-style idea:
        - divide all samples into K+1 groups by d in [0, K]
        - allocate budget approximately uniformly across remaining groups
        - when a group has fewer samples than its budget, redistribute surplus

    Here we implement a practical faithful version.
    """
    n = len(distances)
    total_budget = int(round(n * selection_ratio))
    total_budget = max(1, min(total_budget, n))

    rng = np.random.default_rng(seed)

    groups: Dict[int, List[int]] = {d: [] for d in range(max_step + 1)}
    for idx, d in enumerate(distances.tolist()):
        d = int(d)
        d = max(0, min(d, max_step))
        groups[d].append(idx)

    for d in groups:
        rng.shuffle(groups[d])

    active_keys = [d for d in range(max_step + 1) if len(groups[d]) > 0]
    selected: List[int] = []
    remaining_budget = total_budget

    while len(active_keys) > 0 and remaining_budget > 0:
        # choose the currently smallest group first
        active_keys.sort(key=lambda d: len(groups[d]))
        smallest_d = active_keys[0]

        remaining_groups = len(active_keys)
        alloc = remaining_budget // remaining_groups
        if alloc <= 0:
            alloc = 1

        take = min(len(groups[smallest_d]), alloc)

        chosen = groups[smallest_d][:take]
        selected.extend(chosen)
        groups[smallest_d] = groups[smallest_d][take:]
        remaining_budget -= take

        # remove exhausted group
        active_keys = [d for d in active_keys if len(groups[d]) > 0]

    # If budget remains due to rounding or empty groups, fill globally from unselected samples,
    # preferring smaller distance first.
    if remaining_budget > 0:
        all_indices = np.arange(n)
        selected_set = set(selected)
        remain = [i for i in all_indices.tolist() if i not in selected_set]

        # random tie-break + distance priority
        remain = np.array(remain, dtype=np.int64)
        perm = rng.permutation(len(remain))
        remain_perm = remain[perm]
        order = np.argsort(distances[remain_perm], kind="stable")
        fill = remain_perm[order[:remaining_budget]].tolist()
        selected.extend(fill)

    selected = np.array(sorted(selected), dtype=np.int64)
    if len(selected) > total_budget:
        selected = selected[:total_budget]

    return selected


def select_boundaryset_ccs_count(
    distances: np.ndarray,
    total_budget: int,
    max_step: int,
    seed: int,
) -> np.ndarray:
    n = len(distances)
    total_budget = max(1, min(total_budget, n))

    rng = np.random.default_rng(seed)

    groups: Dict[int, List[int]] = {d: [] for d in range(max_step + 1)}
    for idx, d in enumerate(distances.tolist()):
        d = int(d)
        d = max(0, min(d, max_step))
        groups[d].append(idx)

    for d in groups:
        rng.shuffle(groups[d])

    active_keys = [d for d in range(max_step + 1) if len(groups[d]) > 0]
    selected: List[int] = []
    remaining_budget = total_budget

    while len(active_keys) > 0 and remaining_budget > 0:
        active_keys.sort(key=lambda d: len(groups[d]))
        smallest_d = active_keys[0]

        remaining_groups = len(active_keys)
        alloc = remaining_budget // remaining_groups
        if alloc <= 0:
            alloc = 1

        take = min(len(groups[smallest_d]), alloc)

        chosen = groups[smallest_d][:take]
        selected.extend(chosen)
        groups[smallest_d] = groups[smallest_d][take:]
        remaining_budget -= take

        active_keys = [d for d in active_keys if len(groups[d]) > 0]

    if remaining_budget > 0:
        all_indices = np.arange(n)
        selected_set = set(selected)
        remain = [i for i in all_indices.tolist() if i not in selected_set]

        remain = np.array(remain, dtype=np.int64)
        perm = rng.permutation(len(remain))
        remain_perm = remain[perm]
        order = np.argsort(distances[remain_perm], kind="stable")
        fill = remain_perm[order[:remaining_budget]].tolist()
        selected.extend(fill)

    selected = np.array(sorted(selected), dtype=np.int64)
    if len(selected) > total_budget:
        selected = selected[:total_budget]

    return selected


def select_per_class(
    distances: np.ndarray,
    labels: np.ndarray,
    num_classes: int,
    per_class_count: int,
    method: str,
    max_step: int,
    seed: int,
) -> np.ndarray:
    selected_all: List[int] = []
    for cls in range(num_classes):
        cls_indices = np.where(labels == cls)[0]
        if len(cls_indices) == 0:
            print(f"[Select] Warning: class {cls} has 0 samples.")
            continue

        target = min(per_class_count, len(cls_indices))
        cls_dist = distances[cls_indices]
        cls_seed = seed + cls

        if method == "boundaryset":
            chosen_local = select_boundaryset_count(cls_dist, target, cls_seed)
        else:
            chosen_local = select_boundaryset_ccs_count(cls_dist, target, max_step, cls_seed)

        selected_all.extend(cls_indices[chosen_local].tolist())

        if len(cls_indices) < per_class_count:
            print(f"[Select] Warning: class {cls} has only {len(cls_indices)} samples.")

    return np.array(sorted(selected_all), dtype=np.int64)


# =========================
# Saving results
# =========================

def save_results(
    output_dir: str,
    selected_indices: np.ndarray,
    distances: np.ndarray,
    labels: np.ndarray,
    metas: List[Optional[str]],
    args
) -> None:
    ensure_dir(output_dir)

    np.save(os.path.join(output_dir, "selected_indices.npy"), selected_indices)
    np.save(os.path.join(output_dir, "distances.npy"), distances)
    np.save(os.path.join(output_dir, "labels.npy"), labels)

    summary = {
        "dataset": args.dataset,
        "data_root": args.data_root,
        "checkpoint": args.checkpoint,
        "num_classes": args.num_classes,
        "selection_ratio": args.selection_ratio,
        "num_selected": int(len(selected_indices)),
        "method": args.method,
        "per_class_count": args.per_class_count,
        "alpha": args.alpha,
        "max_step": args.max_step,
        "batch_size": args.batch_size,
        "seed": args.seed,
        "distance_histogram": {
            str(i): int((distances == i).sum()) for i in range(args.max_step + 1)
        },
        "selected_distance_histogram": {
            str(i): int((distances[selected_indices] == i).sum()) for i in range(args.max_step + 1)
        },
        "selected_class_histogram": {
            str(i): int((labels[selected_indices] == i).sum()) for i in range(args.num_classes)
        },
    }
    save_json(summary, os.path.join(output_dir, "summary.json"))

    # Save a readable txt file
    txt_path = os.path.join(output_dir, "selected_indices.txt")
    with open(txt_path, "w", encoding="utf-8") as f:
        for idx in selected_indices.tolist():
            f.write(f"{idx}\n")

    # If ImageFolder, also save selected file paths
    if any(m is not None for m in metas):
        path_txt = os.path.join(output_dir, "selected_paths.txt")
        with open(path_txt, "w", encoding="utf-8") as f:
            for idx in selected_indices.tolist():
                if metas[idx] is not None:
                    f.write(f"{metas[idx]}\n")

        path_json = os.path.join(output_dir, "selected_items.json")
        items = []
        for idx in selected_indices.tolist():
            items.append({
                "index": int(idx),
                "label": int(labels[idx]),
                "distance": int(distances[idx]),
                "path": metas[idx]
            })
        save_json({"items": items}, path_json)

    # Save full per-sample metadata
    all_items = []
    for i in range(len(distances)):
        all_items.append({
            "index": int(i),
            "label": int(labels[i]),
            "distance": int(distances[i]),
            "meta": metas[i]
        })
    save_json({"items": all_items}, os.path.join(output_dir, "all_items.json"))

    print(f"[Save] Results saved to: {output_dir}")


def export_selected_samples(
    dataset: IndexedDataset,
    selected_indices: np.ndarray,
    labels: np.ndarray,
    metas: List[Optional[str]],
    export_dir: str,
) -> None:
    ensure_dir(export_dir)

    has_paths = any(m is not None for m in metas)
    counters = defaultdict(int)

    if has_paths:
        for idx in selected_indices.tolist():
            src = metas[idx]
            label = int(labels[idx])
            if src is None:
                print(f"[Export] Warning: missing path for index {idx}, label {label}.")
                continue

            class_name = os.path.basename(os.path.dirname(src))
            ext = os.path.splitext(src)[1]
            dst_dir = os.path.join(export_dir, class_name)
            ensure_dir(dst_dir)
            dst_name = f"{counters[class_name]:05d}{ext}"
            dst = os.path.join(dst_dir, dst_name)
            shutil.copy2(src, dst)
            counters[class_name] += 1
    else:
        base = dataset.base_dataset
        if not hasattr(base, "data"):
            raise ValueError("Dataset has no file paths or raw data to export.")
        for idx in selected_indices.tolist():
            label = int(labels[idx])
            class_name = f"class_{label}"
            dst_dir = os.path.join(export_dir, class_name)
            ensure_dir(dst_dir)
            img_arr = base.data[idx]
            img = Image.fromarray(img_arr)
            dst_name = f"{counters[class_name]:05d}.png"
            img.save(os.path.join(dst_dir, dst_name))
            counters[class_name] += 1

    print(f"[Export] Selected samples saved to: {export_dir}")


# =========================
# Main
# =========================

def parse_args():
    parser = argparse.ArgumentParser("Boundary-aware coreset selection")

    # data
    parser.add_argument("--dataset", type=str, required=True,
                        choices=["cifar10", "cifar100", "imagefolder"])
    parser.add_argument("--data-root", type=str, required=True)
    parser.add_argument("--download", action="store_true",
                        help="Only used for CIFAR datasets")

    # model
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--num-classes", type=int, required=True)

    # selection
    parser.add_argument("--selection-ratio", type=float, default=None,
                        help="Fraction of samples to keep, e.g. 0.3 (used when per-class-count is 0)")
    parser.add_argument("--per-class-count", type=int, default=50,
                        help="Number of samples to select per class; set to 0 to use selection-ratio")
    parser.add_argument("--method", type=str, default="boundaryset_ccs",
                        choices=["boundaryset", "boundaryset_ccs"])
    parser.add_argument("--alpha", type=float, required=True,
                        help="Step size for distance-to-boundary estimation")
    parser.add_argument("--max-step", type=int, required=True,
                        help="Maximum number of update steps")
    parser.add_argument("--use-sign", action="store_true", default=True,
                        help="Use sign(grad) update, matching the paper's Eq. (1) style")
    parser.add_argument("--no-use-sign", dest="use_sign", action="store_false")

    # data loader
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--num-workers", type=int, default=4)

    # transforms for imagefolder
    parser.add_argument("--resize-size", type=int, default=0,
                        help="0 means disabled")
    parser.add_argument("--crop-size", type=int, default=0,
                        help="0 means disabled")

    # normalization
    parser.add_argument("--mean", type=float, nargs=3,
                        default=[0.485, 0.456, 0.406])
    parser.add_argument("--std", type=float, nargs=3,
                        default=[0.229, 0.224, 0.225])

    # numeric range clamp for adversarial updating
    parser.add_argument("--clamp-min", type=float, default=None)
    parser.add_argument("--clamp-max", type=float, default=None)

    # system
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--output-dir", type=str, required=True)
    parser.add_argument("--export-dir", type=str, required=True,
                        help="Directory to store selected samples")

    return parser.parse_args()


def main():
    args = parse_args()
    set_seed(args.seed)
    cudnn.benchmark = True

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"[Device] Using device: {device}")

    ensure_dir(args.output_dir)
    if args.per_class_count <= 0 and args.selection_ratio is None:
        raise ValueError("selection-ratio is required when per-class-count is 0.")

    # Build dataset
    dataset = build_dataset(args)
    print(f"[Data] Dataset size: {len(dataset)}")

    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=False
    )

    # Build model
    model = build_model(args.num_classes)
    robust_load_checkpoint(model, args.checkpoint, device)
    model.to(device)
    model.eval()

    # Compute distances
    distances, metas, labels = compute_boundary_distances(
        model=model,
        loader=loader,
        device=device,
        alpha=args.alpha,
        max_step=args.max_step,
        clamp_min=args.clamp_min,
        clamp_max=args.clamp_max,
        use_sign=args.use_sign,
    )

    print("[Distance] Done.")
    unique_d, counts_d = np.unique(distances, return_counts=True)
    print("[Distance] Histogram:")
    for d, c in zip(unique_d.tolist(), counts_d.tolist()):
        print(f"    d={d}: {c}")

    # Select coreset
    if args.per_class_count > 0:
        selected_indices = select_per_class(
            distances=distances,
            labels=labels,
            num_classes=args.num_classes,
            per_class_count=args.per_class_count,
            method=args.method,
            max_step=args.max_step,
            seed=args.seed
        )
    else:
        if args.method == "boundaryset":
            selected_indices = select_boundaryset(
                distances=distances,
                selection_ratio=args.selection_ratio,
                seed=args.seed
            )
        else:
            selected_indices = select_boundaryset_ccs(
                distances=distances,
                selection_ratio=args.selection_ratio,
                max_step=args.max_step,
                seed=args.seed
            )

    print(f"[Select] Method: {args.method}")
    print(f"[Select] Selected {len(selected_indices)} / {len(distances)} samples")

    selected_dist = distances[selected_indices]
    unique_sd, counts_sd = np.unique(selected_dist, return_counts=True)
    print("[Select] Selected distance histogram:")
    for d, c in zip(unique_sd.tolist(), counts_sd.tolist()):
        print(f"    d={d}: {c}")

    # Save
    save_results(
        output_dir=args.output_dir,
        selected_indices=selected_indices,
        distances=distances,
        labels=labels,
        metas=metas,
        args=args
    )

    export_selected_samples(
        dataset=dataset,
        selected_indices=selected_indices,
        labels=labels,
        metas=metas,
        export_dir=args.export_dir
    )

    print("[Done]")


if __name__ == "__main__":
    main()
