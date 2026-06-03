import os
import re
import shutil
from typing import Iterable, Tuple

# ======== 配置区域 ========

# 源目录（包含多个类别子文件夹）
SRC_ROOT = "/data/mmc_lyxiang/DD/MinimaxDiffusion/results/dit-distillation/imagenet-10-1000-average_embed/"

# 目标目录（会自动创建）
DST_ROOT = "/data/mmc_lyxiang/DD/MinimaxDiffusion/results/dit-distillation/imagenet-10-1000-average_embed_ipc50/"

# 每个类别需要保留的图像数量
NUM_IMAGES_PER_CLASS = 50

# 允许的图像扩展名
IMAGE_EXTENSIONS: Tuple[str, ...] = (".png", ".jpg", ".jpeg")

# =========================


def _natural_key(text: str) -> Iterable[Tuple[int, str]]:
    """
    自然排序 key：数字部分按数值排序，其他按字符串排序。
    """
    for chunk in re.split(r"(\d+)", text):
        if chunk.isdigit():
            yield int(chunk), ""
        else:
            yield 0, chunk.lower()


def copy_first_n_images(src_root: str, dst_root: str, num_images: int) -> None:
    os.makedirs(dst_root, exist_ok=True)
    class_names = sorted(
        entry for entry in os.listdir(src_root)
        if os.path.isdir(os.path.join(src_root, entry))
    )

    for class_name in class_names:
        class_src = os.path.join(src_root, class_name)
        class_dst = os.path.join(dst_root, class_name)
        os.makedirs(class_dst, exist_ok=True)

        image_files = [
            file_name for file_name in os.listdir(class_src)
            if file_name.lower().endswith(IMAGE_EXTENSIONS)
        ]
        image_files.sort(key=lambda name: tuple(_natural_key(name)))

        selected_files = image_files[:num_images]
        if not selected_files:
            print(f"⚠️ 类别 {class_name} 没有可复制的图像，已跳过。")
            continue

        for file_name in selected_files:
            src_path = os.path.join(class_src, file_name)
            dst_path = os.path.join(class_dst, file_name)
            shutil.copy2(src_path, dst_path)
        print(f"✅ 类别 {class_name}: 复制 {len(selected_files)} 张图像到目标目录。")

    print("\n🎯 所有类别处理完成！")


if __name__ == "__main__":
    copy_first_n_images(SRC_ROOT, DST_ROOT, NUM_IMAGES_PER_CLASS)
