import argparse
import os


def read_classes(path):
    with open(path, "r", encoding="utf-8") as handle:
        return [line.strip() for line in handle if line.strip()]


def link_split(src_root, dst_root, classes):
    os.makedirs(dst_root, exist_ok=True)
    for class_idx, class_name in enumerate(classes):
        src = os.path.join(src_root, class_name)
        dst = os.path.join(dst_root, f"{class_idx:03d}_{class_name}")
        if not os.path.isdir(src):
            raise FileNotFoundError(f"Missing source class directory: {src}")
        if os.path.lexists(dst):
            continue
        os.symlink(src, dst)


def main(args):
    classes = read_classes(args.class_file)
    link_split(args.train_src, os.path.join(args.output_root, "train"), classes)
    link_split(args.val_src, os.path.join(args.output_root, "val"), classes)
    print(f"Prepared ImageWoof links at {args.output_root} ({len(classes)} classes).")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--class-file", default="./misc/class_woof.txt")
    parser.add_argument("--train-src", default="/data/mmc_lyxiang/dataset/ImageNet/train")
    parser.add_argument("--val-src", default="/data/mmc_lyxiang/dataset/ImageNet/val")
    parser.add_argument("--output-root", required=True)
    main(parser.parse_args())
