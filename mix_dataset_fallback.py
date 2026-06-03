import os
import random
import shutil
from glob import glob

def mix_two_datasets_with_fallback(
    src_root_A,        # ✅ 充足数据集（base）
    src_root_B,        # ✅ 稀缺数据集（rare）
    dst_root,          # ✅ 输出新数据集路径
    samples_per_class=1000,   # ✅ 每个类别最终样本数 N
    mix_ratio=0.3,     # ✅ B 数据集所占比例 r（如 0.3 表示 30% 来自 B）
    seed=0,
    copy_mode=True    # True=复制，False=硬链接（省空间）
):
    assert 0.0 <= mix_ratio <= 1.0

    random.seed(seed)
    os.makedirs(dst_root, exist_ok=True)

    class_names = sorted(os.listdir(src_root_A))

    for cls in class_names:
        dir_A = os.path.join(src_root_A, cls)
        dir_B = os.path.join(src_root_B, cls)

        if not os.path.isdir(dir_A):
            print(f"⚠️ Skip {cls}, not found in A")
            continue

        # ✅ 读取 A 类别图片
        imgs_A = []
        for ext in ["*.jpg", "*.png", "*.jpeg", "*.JPEG", "*.bmp", "*.webp"]:
            imgs_A.extend(glob(os.path.join(dir_A, ext)))

        # ✅ 读取 B 类别图片（可能为空）
        imgs_B = []
        if os.path.isdir(dir_B):
            for ext in ["*.jpg", "*.png", "*.jpeg", "*.JPEG", "*.bmp", "*.webp"]:
                imgs_B.extend(glob(os.path.join(dir_B, ext)))

        NA = len(imgs_A)
        NB = len(imgs_B)

        if NA == 0:
            print(f"❌ Fatal: class {cls} has 0 samples in dataset A!")
            continue

        # ✅ 目标混合数量
        target_B = int(samples_per_class * mix_ratio)
        target_A = samples_per_class - target_B

        # ===============================
        # ✅ 先从 B 取
        # ===============================

        if NB >= target_B:
            chosen_B = random.sample(imgs_B, target_B)
            lack = 0
        else:
            chosen_B = imgs_B
            lack = target_B - NB

        # ===============================
        # ✅ 不足部分由 A 补充
        # ===============================

        final_A_num = target_A + lack

        if NA >= final_A_num:
            chosen_A = random.sample(imgs_A, final_A_num)
        else:
            chosen_A = random.choices(imgs_A, k=final_A_num)

        final_imgs = chosen_B + chosen_A
        random.shuffle(final_imgs)

        # ✅ 创建目标目录
        dst_cls_dir = os.path.join(dst_root, cls)
        os.makedirs(dst_cls_dir, exist_ok=True)

        # ✅ 写入新数据集
        for i, src_img in enumerate(final_imgs):
            new_name = f"{i:06d}_" + os.path.basename(src_img)
            dst_img = os.path.join(dst_cls_dir, new_name)

            if copy_mode:
                shutil.copy2(src_img, dst_img)
            else:
                os.link(src_img, dst_img)

        print(
            f"✅ {cls}: "
            f"A={NA}, B={NB} → "
            f"B_used={len(chosen_B)}, A_used={len(chosen_A)} → "
            f"Final={len(final_imgs)}"
        )

    print("\n🎯 Mixed dataset construction finished!")

src_root_A = "/data/mmc_lyxiang/dataset/CaO2/CaO2_Distilled_Data/woof-ipc50/"   # ✅ 充足数据集
src_root_B = "/data/mmc_lyxiang/DD/MinimaxDiffusion/results/dit-distillation/high_data_u_from_ImageNetWoof/"   # ✅ 稀缺数据集
dst_root   = "/data/mmc_lyxiang/DD/MinimaxDiffusion/results/dit-distillation/mixed_dataset+High_data_u_from_ImageNetWoof+CaO2/"   # ✅ 输出路径

mix_two_datasets_with_fallback(
    src_root_A=src_root_A,
    src_root_B=src_root_B,
    dst_root=dst_root,
    samples_per_class=50,   # ✅ 每类最终 50
    mix_ratio=0.5,            # ✅ 50% 来自小数据集
    seed=42,
    copy_mode=True
)
