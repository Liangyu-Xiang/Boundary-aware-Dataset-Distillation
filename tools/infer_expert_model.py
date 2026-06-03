import argparse
import sys
import csv

from pathlib import Path

import torch
from PIL import Image
from torchvision import transforms


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from misc.utils import load_model


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}


def load_subset_metadata(spec):
    misc_dir = REPO_ROOT / "misc"
    with open(misc_dir / "class_indices.txt", "r") as fp:
        all_ids = [line.strip() for line in fp]
    with open(misc_dir / "class_names.txt", "r") as fp:
        all_names = [line.strip() for line in fp]

    if spec == "woof":
        subset_file = misc_dir / "class_woof.txt"
    elif spec == "nette":
        subset_file = misc_dir / "class_nette.txt"
    elif spec == "100":
        subset_file = misc_dir / "class100.txt"
    elif spec == "1k":
        subset_file = misc_dir / "class_indices.txt"
    else:
        raise ValueError(f"Unsupported dataset spec '{spec}'.")

    with open(subset_file, "r") as fp:
        subset_ids = [line.strip() for line in fp]

    subset_names = []
    for class_id in subset_ids:
        imagenet_idx = all_ids.index(class_id)
        subset_names.append(all_names[imagenet_idx])
    return subset_ids, subset_names


def build_image_transform():
    return transforms.Compose(
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


def collect_images_from_path(path, recursive):
    if path.is_file():
        if path.suffix.lower() not in IMAGE_EXTENSIONS:
            raise ValueError(f"Unsupported image extension: {path}")
        return [path]
    if path.is_dir():
        pattern = "**/*" if recursive else "*"
        return sorted(
            file_path
            for file_path in path.glob(pattern)
            if file_path.is_file() and file_path.suffix.lower() in IMAGE_EXTENSIONS
        )
    raise FileNotFoundError(f"Path not found: {path}")


def get_input_paths(args):
    input_paths = list(args.paths or [])
    if args.image_list:
        with open(args.image_list, "r") as fp:
            input_paths.extend(line.strip() for line in fp if line.strip())

    if not input_paths:
        raise ValueError("No input paths provided. Use --paths or --image-list.")
    return input_paths


def resolve_input_paths(input_paths, recursive):
    resolved = []
    seen = set()
    for input_path in input_paths:
        path = Path(input_path).expanduser()
        if not path.is_absolute():
            path = (Path.cwd() / path).resolve()
        for image_path in collect_images_from_path(path, recursive):
            if image_path not in seen:
                resolved.append(image_path)
                seen.add(image_path)
    if not resolved:
        raise ValueError("No image files found in the provided paths.")
    return resolved


def infer_one_image(model, image_path, transform, device, topk):
    image = Image.open(image_path).convert("RGB")
    tensor = transform(image).unsqueeze(0).to(device)
    with torch.no_grad():
        logits = model(tensor)
        probs = torch.softmax(logits, dim=1)
        entropy = -(probs * torch.log(probs.clamp_min(1e-12))).sum(dim=1)
        scores, indices = torch.topk(probs, k=topk, dim=1)
    return scores[0].cpu(), indices[0].cpu(), entropy[0].cpu()


def build_output_rows(image_paths, predictions, subset_ids, subset_names):
    rows = []
    for image_path, (scores, indices, _) in zip(image_paths, predictions):
        row = {"image_name": image_path.name}
        for rank, (score, pred_idx) in enumerate(zip(scores.tolist(), indices.tolist()), start=1):
            row[f"top{rank}_prob"] = f"{score:.8f}"
            row[f"top{rank}_class_name"] = subset_names[pred_idx]
            row[f"top{rank}_class_id"] = subset_ids[pred_idx]
            row[f"top{rank}_subset_idx"] = pred_idx
        rows.append(row)
    return rows


def compute_summary(predictions):
    top1_probs = torch.tensor([scores[0].item() for scores, _, _ in predictions])
    entropies = torch.tensor([entropy.item() for _, _, entropy in predictions])
    return {
        "mean_top1_prob": top1_probs.mean().item(),
        "mean_entropy": entropies.mean().item(),
    }


def save_predictions_csv(rows, output_path, topk, command_paths, image_list, summary):
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["image_name"]
    for rank in range(1, topk + 1):
        fieldnames.append(f"top{rank}_prob")
    for rank in range(1, topk + 1):
        fieldnames.extend([f"top{rank}_class_name", f"top{rank}_class_id", f"top{rank}_subset_idx"])
    with open(output_path, "w", newline="") as fp:
        if command_paths:
            fp.write(f"# input_paths: {';'.join(command_paths)}\n")
        if image_list:
            fp.write(f"# image_list: {image_list}\n")
        fp.write(f"# mean_top1_prob: {summary['mean_top1_prob']:.8f}\n")
        fp.write(f"# mean_entropy: {summary['mean_entropy']:.8f}\n")
        writer = csv.DictWriter(fp, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main(args):
    device = torch.device(args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu"))
    input_paths = get_input_paths(args)
    command_paths = list(args.paths or [])
    image_paths = resolve_input_paths(input_paths, args.recursive)
    subset_ids, subset_names = load_subset_metadata(args.spec)

    if args.nclass != len(subset_ids):
        raise ValueError(
            f"--nclass={args.nclass} does not match spec '{args.spec}' class count {len(subset_ids)}."
        )

    model = load_model(
        model_name=args.model_name,
        dataset=args.spec,
        pretrained=True,
        classes=range(args.nclass),
    ).to(device)
    model.eval()
    for param in model.parameters():
        param.requires_grad = False

    transform = build_image_transform()
    topk = min(args.topk, args.nclass)
    output_path = Path(args.output).expanduser()
    if not output_path.is_absolute():
        output_path = (Path.cwd() / output_path).resolve()

    print(f"Using device: {device}")
    print(f"Loaded expert model: {args.model_name} for spec={args.spec}, nclass={args.nclass}")
    print(f"Images to process: {len(image_paths)}")

    predictions = []
    for image_path in image_paths:
        scores, indices, entropy = infer_one_image(model, image_path, transform, device, topk)
        predictions.append((scores, indices, entropy))

    rows = build_output_rows(image_paths, predictions, subset_ids, subset_names)
    summary = compute_summary(predictions)
    save_predictions_csv(rows, output_path, topk, command_paths, args.image_list, summary)
    print(f"Saved top{topk} probabilities to: {output_path}")
    print(f"Mean top1 probability: {summary['mean_top1_prob']:.8f}")
    print(f"Mean entropy: {summary['mean_entropy']:.8f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--paths", nargs="*", default=None, help="Image files or directories to run inference on.")
    parser.add_argument("--image-list", type=str, default=None, help="Text file containing one image file or directory per line.")
    parser.add_argument("--spec", type=str, required=True, choices=["woof", "nette", "100", "1k"])
    parser.add_argument("--nclass", type=int, required=True, help="Number of expert classes for the selected spec.")
    parser.add_argument("--model-name", type=str, default="resnet18")
    parser.add_argument("--topk", type=int, default=5)
    parser.add_argument("--output", type=str, default="expert_top5_probs.csv")
    parser.add_argument("--recursive", action="store_true", help="Recursively scan directories for images.")
    parser.add_argument("--device", type=str, default=None, help="Override device, e.g. cpu or cuda:0.")
    args = parser.parse_args()
    main(args)
