import os
import random
import shutil
from glob import glob

def rebuild_dataset_uniform_per_class(
    src_root,      # 原始数据集路径
    dst_root,      # 新数据集路径
    samples_per_class=1000,  # ✅ 每个类别统一采样数量 N
    seed=0,
    copy_mode=True  # True=复制数据，False=硬链接（节省空间）
):
    random.seed(seed)
    os.makedirs(dst_root, exist_ok=True)

    class_names = sorted(os.listdir(src_root))

    for cls in class_names:
        src_cls_dir = os.path.join(src_root, cls)
        if not os.path.isdir(src_cls_dir):
            continue

        # ✅ 读取该类下所有图片
        imgs = []
        for ext in ["*.jpg", "*.png", "*.jpeg", "*.bmp", "*.webp"]:
            imgs.extend(glob(os.path.join(src_cls_dir, ext)))

        num_src = len(imgs)

        if num_src == 0:
            print(f"⚠️ Empty class: {cls}, skipped!")
            continue

        # ✅ 欠采样 or 过采样（允许重复）
        if num_src >= samples_per_class:
            chosen_imgs = random.sample(imgs, samples_per_class)
        else:
            chosen_imgs = random.choices(imgs, k=samples_per_class)

        # ✅ 创建目标类别文件夹
        dst_cls_dir = os.path.join(dst_root, cls)
        os.makedirs(dst_cls_dir, exist_ok=True)

        # ✅ 写入新数据集
        for i, src_img in enumerate(chosen_imgs):
            new_name = f"{i:06d}_" + os.path.basename(src_img)
            dst_img = os.path.join(dst_cls_dir, new_name)

            if copy_mode:
                shutil.copy2(src_img, dst_img)
            else:
                os.link(src_img, dst_img)

        print(f"✅ {cls}: {num_src} → {samples_per_class}")

    print("\n🎯 New balanced dataset construction finished!")

src_root = "/data/old_dataset"
dst_root = "/data/new_balanced_dataset"

rebuild_dataset_uniform_per_class(
    src_root=src_root,
    dst_root=dst_root,
    samples_per_class=1000,   # ✅ 你只需要改这一行
    seed=42,
    copy_mode=True
)
