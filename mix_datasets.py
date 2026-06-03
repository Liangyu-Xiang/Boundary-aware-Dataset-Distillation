import argparse
import random
import shutil
from pathlib import Path
from typing import Dict, List, Tuple


def list_common_classes(root_a: Path, root_b: Path) -> List[str]:
    classes_a = {entry.name for entry in root_a.iterdir() if entry.is_dir()}
    classes_b = {entry.name for entry in root_b.iterdir() if entry.is_dir()}
    return sorted(classes_a & classes_b)


def gather_images(class_dir: Path) -> List[Path]:
    return [path for path in class_dir.iterdir() if path.is_file()]


def select_samples(images: List[Path], count: int, rng: random.Random) -> List[Path]:
    if count > len(images):
        raise ValueError("Requested more samples than available images.")
    if count == len(images):
        return images.copy()
    return rng.sample(images, count)


def determine_counts(
    available_a: int,
    available_b: int,
    total_needed: int,
    ratio_a: float,
) -> Tuple[int, int]:
    if available_a + available_b < total_needed:
        raise ValueError(
            f"Insufficient images: need {total_needed}, but only "
            f"{available_a + available_b} available."
        )

    target_a = int(round(total_needed * ratio_a))
    target_b = total_needed - target_a

    count_a = min(target_a, available_a)
    count_b = min(target_b, available_b)

    remaining = total_needed - (count_a + count_b)
    prefer_a = ratio_a >= 0.5

    while remaining > 0:
        extra_a = available_a - count_a
        extra_b = available_b - count_b

        if extra_a <= 0 and extra_b <= 0:
            raise ValueError("Unable to satisfy total-per-class requirement.")

        choose_a = False
        if extra_a > extra_b:
            choose_a = True
        elif extra_b > extra_a:
            choose_a = False
        else:
            choose_a = prefer_a

        if choose_a and extra_a > 0:
            count_a += 1
            remaining -= 1
        elif extra_b > 0:
            count_b += 1
            remaining -= 1
        elif extra_a > 0:
            count_a += 1
            remaining -= 1
        else:
            raise ValueError("Unable to allocate remaining samples.")

    return count_a, count_b


def copy_samples(samples: List[Path], dst_dir: Path, prefix: str) -> None:
    dst_dir.mkdir(parents=True, exist_ok=True)
    for idx, src in enumerate(samples):
        dst_name = f"{prefix}_{idx:05d}_{src.name}"
        shutil.copy2(src, dst_dir / dst_name)


def mix_datasets(
    dataset_a: Path,
    dataset_b: Path,
    dst_root: Path,
    total_per_class: int,
    ratio_a: float,
    seed: int,
) -> Dict[str, Tuple[int, int]]:
    rng = random.Random(seed)
    stats: Dict[str, Tuple[int, int]] = {}

    dst_root.mkdir(parents=True, exist_ok=True)
    dataset_name = dst_root.name

    common_classes = list_common_classes(dataset_a, dataset_b)
    if not common_classes:
        raise ValueError("No common class directories found between the two datasets.")

    for class_name in common_classes:
        class_dir_a = dataset_a / class_name
        class_dir_b = dataset_b / class_name

        images_a = gather_images(class_dir_a)
        images_b = gather_images(class_dir_b)

        try:
            num_a, num_b = determine_counts(
                len(images_a), len(images_b), total_per_class, ratio_a
            )
        except ValueError as exc:
            raise ValueError(f"Class {class_name}: {exc}") from exc

        selected_a = select_samples(images_a, num_a, rng)
        selected_b = select_samples(images_b, num_b, rng)

        dst_class_dir = dst_root / class_name
        copy_samples(selected_a, dst_class_dir, "A")
        copy_samples(selected_b, dst_class_dir, "B")
        stats[class_name] = (len(selected_a), len(selected_b))
        print(
            f"✅  {dataset_name}/{class_name}: "
            f"from datasetA {len(selected_a)} images, datasetB {len(selected_b)} images."
        )

    return stats


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Mix two datasets by class with a given ratio.")
    parser.add_argument("--dataset-a", required=True, type=Path)
    parser.add_argument("--dataset-b", required=True, type=Path)
    parser.add_argument("--dst-root", required=True, type=Path)
    parser.add_argument("--total-per-class", required=True, type=int)
    parser.add_argument("--ratio-a", required=True, type=float,
                        help="Fraction of samples per class drawn from dataset A (between 0 and 1).")
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    dataset_a: Path = args.dataset_a
    dataset_b: Path = args.dataset_b
    dst_root: Path = args.dst_root
    total_per_class: int = args.total_per_class
    ratio_a: float = args.ratio_a
    seed: int = args.seed

    if not (0.0 <= ratio_a <= 1.0):
        raise ValueError("ratio_a must be between 0 and 1.")

    stats = mix_datasets(dataset_a, dataset_b, dst_root, total_per_class, ratio_a, seed)

    print("\n🎯 Mixing completed. Summary:")
    for class_name, (count_a, count_b) in stats.items():
        print(f" - {class_name}: datasetA={count_a}, datasetB={count_b}")


if __name__ == "__main__":
    main()
