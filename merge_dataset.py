#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import shutil
from pathlib import Path
from typing import List, Dict, Tuple


def list_common_classes(root_a: Path, root_b: Path) -> List[str]:
    """Return sorted list of common class directories."""
    classes_a = {p.name for p in root_a.iterdir() if p.is_dir()}
    classes_b = {p.name for p in root_b.iterdir() if p.is_dir()}
    return sorted(classes_a & classes_b)


def gather_images(class_dir: Path) -> List[Path]:
    """Gather all files in a class directory."""
    return [p for p in class_dir.iterdir() if p.is_file()]


def copy_all_images(
    images: List[Path],
    dst_dir: Path,
    prefix: str,
) -> int:
    """Copy all images to destination directory with prefix."""
    dst_dir.mkdir(parents=True, exist_ok=True)
    for idx, src in enumerate(images):
        dst_name = f"{prefix}_{idx:06d}_{src.name}"
        shutil.copy2(src, dst_dir / dst_name)
    return len(images)


def merge_datasets(
    dataset_a: Path,
    dataset_b: Path,
    dst_root: Path,
) -> Dict[str, Tuple[int, int]]:
    """
    Merge two datasets by class using ALL samples.
    Returns statistics: class_name -> (count_a, count_b)
    """
    stats: Dict[str, Tuple[int, int]] = {}

    dst_root.mkdir(parents=True, exist_ok=True)
    dataset_name = dst_root.name

    common_classes = list_common_classes(dataset_a, dataset_b)
    if not common_classes:
        raise RuntimeError("No common class directories found between the two datasets.")

    for class_name in common_classes:
        class_dir_a = dataset_a / class_name
        class_dir_b = dataset_b / class_name
        dst_class_dir = dst_root / class_name

        images_a = gather_images(class_dir_a)
        images_b = gather_images(class_dir_b)

        count_a = copy_all_images(images_a, dst_class_dir, prefix="A")
        count_b = copy_all_images(images_b, dst_class_dir, prefix="B")

        stats[class_name] = (count_a, count_b)

        print(
            f"✅ {dataset_name}/{class_name}: "
            f"{count_a} images from dataset A, "
            f"{count_b} images from dataset B"
        )

    return stats


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Merge two datasets by class using ALL samples."
    )
    parser.add_argument(
        "--dataset-a", required=True, type=Path, help="Path to dataset A"
    )
    parser.add_argument(
        "--dataset-b", required=True, type=Path, help="Path to dataset B"
    )
    parser.add_argument(
        "--dst-root", required=True, type=Path, help="Output merged dataset root"
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    stats = merge_datasets(
        dataset_a=args.dataset_a,
        dataset_b=args.dataset_b,
        dst_root=args.dst_root,
    )

    print("\n🎯 Merge completed. Summary:")
    for cls, (na, nb) in stats.items():
        print(f" - {cls}: datasetA={na}, datasetB={nb}")


if __name__ == "__main__":
    main()
