import os
import random
import argparse

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms

from misc.utils import load_model


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def maybe_subset(dataset, max_samples=None, seed=42):
    if max_samples is None or len(dataset) <= max_samples:
        return dataset
    rng = np.random.default_rng(seed)
    indices = rng.choice(len(dataset), size=max_samples, replace=False)
    return Subset(dataset, indices.tolist())


@torch.no_grad()
def compute_avg_stats(model, dataloader, device, margin_on="probs"):
    total_top1 = 0.0
    total_margin = 0.0
    total_count = 0

    for images, _ in dataloader:
        images = images.to(device, non_blocking=True)
        logits = model(images)
        probs = torch.softmax(logits, dim=1)

        top2_probs = torch.topk(probs, k=2, dim=1).values
        top1_prob = top2_probs[:, 0]

        if margin_on == "logits":
            top2_vals = torch.topk(logits, k=2, dim=1).values
            margin = top2_vals[:, 0] - top2_vals[:, 1]
        else:
            margin = top2_probs[:, 0] - top2_probs[:, 1]

        batch_size = images.size(0)
        total_top1 += top1_prob.sum().item()
        total_margin += margin.sum().item()
        total_count += batch_size

    avg_top1 = total_top1 / max(1, total_count)
    avg_margin = total_margin / max(1, total_count)
    return avg_top1, avg_margin, total_count


def main(args):
    set_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    if not os.path.isdir(args.data_path):
        raise FileNotFoundError(f"data_path not found: {args.data_path}")

    transform = transforms.Compose([
        transforms.Resize(224 // 7 * 8, antialias=True),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        ),
    ])

    dataset = datasets.ImageFolder(args.data_path, transform=transform)
    dataset = maybe_subset(dataset, args.max_samples, seed=args.seed)
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
    )

    model = load_model(
        model_name="resnet18",
        dataset=args.spec,
        pretrained=True,
        classes=range(args.num_classes),
    )
    model.to(device)
    model.eval()

    avg_top1, avg_margin, total_count = compute_avg_stats(
        model, dataloader, device, margin_on=args.margin_on
    )

    print(f"Total samples: {total_count}")
    print(f"Average top1 probability: {avg_top1:.6f}")
    print(f"Average top-2 margin ({args.margin_on}): {avg_margin:.6f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str, required=True)
    parser.add_argument("--spec", type=str, required=True, help="expert dataset spec, e.g. woof")
    parser.add_argument("--num_classes", type=int, required=True)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--max_samples", type=int, default=None)
    parser.add_argument("--margin_on", type=str, choices=["probs", "logits"], default="probs")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    main(args)
