#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Rename class folders to canonical synset IDs with case-insensitive matching.

Default mapping is for ImageWoof 10 classes:
synset_id, English name, Chinese name.
"""
import argparse
import os
import shutil
from typing import Dict, List, Tuple


DEFAULT_MAPPING: List[Tuple[str, str, str]] = [
    ("n02086240", "Shih-Tzu", "西施犬"),
    ("n02087394", "Rhodesian ridgeback", "罗得西亚脊背犬"),
    ("n02088364", "beagle", "比格犬"),
    ("n02089973", "English foxhound", "英国猎狐犬"),
    ("n02093754", "Border terrier", "边境梗"),
    ("n02096294", "Australian terrier", "澳大利亚梗"),
    ("n02099601", "golden retriever", "金毛寻回犬"),
    ("n02105641", "Old English sheepdog", "古代英国牧羊犬"),
    ("n02111889", "Samoyed", "萨摩耶犬"),
    ("n02115641", "Canis dingo", "澳洲野犬（丁狗）"),
]


def normalize_name(name: str) -> str:
    return " ".join(name.strip().lower().replace("_", " ").split())


def build_alias_map(mapping: List[Tuple[str, str, str]]) -> Dict[str, str]:
    alias_to_target: Dict[str, str] = {}
    for synset, english, chinese in mapping:
        target = synset
        for alias in (synset, english, chinese):
            alias_to_target[normalize_name(alias)] = target
    return alias_to_target


def rename_classes(root: str, dry_run: bool = False) -> None:
    alias_map = build_alias_map(DEFAULT_MAPPING)
    entries = [d for d in os.listdir(root) if os.path.isdir(os.path.join(root, d))]
    for name in sorted(entries):
        src = os.path.join(root, name)
        key = normalize_name(name)
        target_name = alias_map.get(key)
        if target_name is None:
            print(f"[Skip] No mapping for: {name}")
            continue
        if name == target_name:
            print(f"[Keep] {name}")
            continue
        dst = os.path.join(root, target_name)
        if dry_run:
            print(f"[DryRun] {name} -> {target_name}")
            continue
        if os.path.exists(dst):
            # Merge into existing folder.
            for item in os.listdir(src):
                shutil.move(os.path.join(src, item), os.path.join(dst, item))
            os.rmdir(src)
            print(f"[Merge] {name} -> {target_name}")
        else:
            os.rename(src, dst)
            print(f"[Rename] {name} -> {target_name}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Rename class folders by mapping.")
    parser.add_argument("--root", type=str, required=True, help="Dataset root with class subfolders.")
    parser.add_argument("--dry-run", action="store_true", help="Print actions without renaming.")
    args = parser.parse_args()

    if not os.path.isdir(args.root):
        raise ValueError(f"Root not found or not a directory: {args.root}")
    rename_classes(args.root, dry_run=args.dry_run)


if __name__ == "__main__":
    main()
