import argparse
import os
import random
import shutil
from pathlib import Path
from typing import List


def load_woof_classes(list_path: Path) -> List[str]:
    with list_path.open("r") as fp:
        classes = [line.strip() for line in fp if line.strip()]
    if not classes:
        raise ValueError(f"No classes found in {list_path}")
    return classes


def collect_images(class_dir: Path) -> List[Path]:
    return [path for path in class_dir.iterdir() if path.is_file()]


def copy_ipc_images(
    src_root: Path,
    dst_root: Path,
    class_ids: List[str],
    ipc: int,
    seed: int,
) -> None:
    rng = random.Random(seed)
    dst_root.mkdir(parents=True, exist_ok=True)

    for class_id in class_ids:
        src_class_dir = src_root / class_id
        if not src_class_dir.is_dir():
            print(f"⚠️  Skipping missing class directory: {src_class_dir}")
            continue

        all_images = collect_images(src_class_dir)
        if not all_images:
            print(f"⚠️  No images found for class {class_id}, skipping.")
            continue

        if ipc > len(all_images):
            print(
                f"⚠️  Requested {ipc} images for {class_id}, "
                f"but only {len(all_images)} available. Using all images."
            )
            selected = all_images
        else:
            selected = rng.sample(all_images, ipc)

        dst_class_dir = dst_root / class_id
        dst_class_dir.mkdir(parents=True, exist_ok=True)

        for image_path in selected:
            shutil.copy2(image_path, dst_class_dir / image_path.name)

        print(f"✅  Copied {len(selected)} images for class {class_id}.")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Randomly sample ImageWoof classes from an ImageNet training set."
    )
    parser.add_argument(
        "--imagenet-dir",
        required=True,
        type=Path,
        help="Path to ImageNet training set (directories per class).",
    )
    parser.add_argument(
        "--dst-dir",
        required=True,
        type=Path,
        help="Destination directory for the sampled dataset.",
    )
    parser.add_argument(
        "--ipc",
        required=True,
        type=int,
        help="Number of images per class to copy.",
    )
    parser.add_argument(
        "--class-list",
        type=Path,
        default=Path("./misc/class_woof.txt"),
        help="Path to file containing ImageWoof class IDs (default: ./misc/class_woof.txt).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Random seed for sampling.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    imagenet_dir: Path = args.imagenet_dir
    dst_dir: Path = args.dst_dir
    ipc: int = args.ipc
    class_list_path: Path = args.class_list

    if not imagenet_dir.is_dir():
        raise FileNotFoundError(f"ImageNet directory not found: {imagenet_dir}")

    class_ids = load_woof_classes(class_list_path)
    copy_ipc_images(imagenet_dir, dst_dir, class_ids, ipc, args.seed)
    print("🎯 Sampling completed.")


if __name__ == "__main__":
    main()
