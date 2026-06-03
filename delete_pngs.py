import os

# ===== 手动配置 =====
SRC_ROOT = "/data/mmc_lyxiang/DD/MinimaxDiffusion/results/dit-distillation/imagenet-10-1000-neg_label-0.5"  # 源文件夹路径
THRESHOLD = 100  # 删除文件名 >= 该值的图片
# ====================

def delete_large_index_pngs(src_root, threshold=100):
    deleted_count = 0
    skipped_count = 0

    # 遍历所有子文件夹（类别）
    for class_name in sorted(os.listdir(src_root)):
        class_dir = os.path.join(src_root, class_name)
        if not os.path.isdir(class_dir):
            continue

        print(f"\n📁 正在处理类别: {class_name}")

        for file_name in os.listdir(class_dir):
            if not file_name.lower().endswith(".png"):
                continue

            try:
                file_id = int(os.path.splitext(file_name)[0])
            except ValueError:
                skipped_count += 1
                continue  # 跳过非数字命名的文件

            if file_id >= threshold:
                file_path = os.path.join(class_dir, file_name)
                os.remove(file_path)
                deleted_count += 1
                print(f"🗑️ 删除: {file_name}")

    print(f"\n✅ 删除完成！共删除 {deleted_count} 个文件，跳过 {skipped_count} 个非整数命名文件。")


if __name__ == "__main__":
    delete_large_index_pngs(SRC_ROOT, THRESHOLD)
