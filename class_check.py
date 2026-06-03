import os
from collections import defaultdict

# =========================
# 配置
# =========================
CLASS_TXT = './misc/class_indices.txt'

DATASET_PATHS = [
   '/data/mmc_lyxiang/DD/MinimaxDiffusion/results/dit-distillation/Exp_IPC10/1k/p_x_y_2_Gated+lr1e-4+Batch8+Epoch1+Weight0.8-1+Gap0.2+Confusion+CFG4/',
]

EXPECTED_NUM_PER_CLASS = 10
EXTENSIONS = ('.jpg', '.jpeg', '.png', '.bmp', '.webp')


# =========================
# 工具函数
# =========================
def load_all_classes(txt_path):
    with open(txt_path, 'r') as fp:
        return [line.strip() for line in fp if line.strip()]


def count_images(dir_path):
    if not os.path.isdir(dir_path):
        return 0
    return sum(
        1 for f in os.listdir(dir_path)
        if f.lower().endswith(EXTENSIONS)
    )

def compress_indices_to_ranges(indices):
    """
    将 [0,1,2,5,6,9] -> ['0-2', '5-6', '9']
    """
    if not indices:
        return []

    indices = sorted(indices)
    ranges = []

    start = prev = indices[0]
    for idx in indices[1:]:
        if idx == prev + 1:
            prev = idx
        else:
            if start == prev:
                ranges.append(f'{start}')
            else:
                ranges.append(f'{start}-{prev}')
            start = prev = idx

    # last range
    if start == prev:
        ranges.append(f'{start}')
    else:
        ranges.append(f'{start}-{prev}')

    return ranges


# =========================
# 主检查逻辑
# =========================
def analyze_merged_dataset(paths, all_classes):
    # 规范化路径，避免结尾 / 导致键不一致
    paths = [os.path.normpath(p) for p in paths]

    # 统计结构
    class_to_count = defaultdict(int)  # cls -> total num
    class_to_path_count = defaultdict(lambda: defaultdict(int))  # cls -> path -> num

    # 扫描所有路径
    for p in paths:
        if not os.path.isdir(p):
            continue

        for cls in os.listdir(p):
            if not cls.startswith('n'):
                continue

            cls_dir = os.path.join(p, cls)
            if not os.path.isdir(cls_dir):
                continue

            num = count_images(cls_dir)
            if num <= 0:
                continue

            class_to_count[cls] += num
            class_to_path_count[cls][p] += num

    found_classes = set(class_to_count.keys())
    cls_to_idx = {cls: idx for idx, cls in enumerate(all_classes)}

    # =========================
    # 1️⃣ 完全缺失类别（保持原样：只输出 index）
    # =========================
    missing_class_indices = [
        idx for idx, cls in enumerate(all_classes)
        if cls not in found_classes
    ]

    # =========================
    # 2️⃣ 样本不足类别（输出 index + 每个路径下的“完整类别目录路径”）
    # =========================
    insufficient = []
    excessive = []

    for cls, total_num in class_to_count.items():
        idx = cls_to_idx.get(cls, None)
        if idx is None:
            # 如果你的数据里出现了不在 class_indices.txt 的类，这里直接跳过或你也可以打印告警
            continue

        if total_num < EXPECTED_NUM_PER_CLASS:
            insufficient.append((idx, cls, total_num))
        elif total_num > EXPECTED_NUM_PER_CLASS:
            excessive.append((idx, cls, total_num))

    # =========================
    # 输出
    # =========================
    print('\n========== MERGED DATASET CHECK ==========')

    # Missing
    if missing_class_indices:
        ranges = compress_indices_to_ranges(missing_class_indices)
        print(f'[❌] Missing classes: {len(missing_class_indices)}')
        print('Index ranges:')
        print(', '.join(ranges))
    else:
        print('[✅] No missing classes')

    # Insufficient (with full class dir paths)
    if insufficient:
        # 按 total_num 升序，优先看最缺的
        insufficient.sort(key=lambda x: x[2])

        print(f'\n[❌] Classes with insufficient samples (< {EXPECTED_NUM_PER_CLASS}): {len(insufficient)}')
        for idx, cls, total_num in insufficient:
            print(f'  - Class index: {idx}')
            print(f'    Total num : {total_num}')
            print(f'    Per-dir breakdown (full path):')

            # 对每个 path 的计数降序输出
            items = sorted(class_to_path_count[cls].items(), key=lambda kv: kv[1], reverse=True)
            for p, cnt in items:
                full_cls_dir = os.path.join(p, cls)
                print(f'      {full_cls_dir}: {cnt}')
    else:
        print('\n[✅] All classes meet sample requirement')

    # Excessive（可选）
    if excessive:
        print(f'\n[ℹ️] Classes with excessive samples (> {EXPECTED_NUM_PER_CLASS}): {len(excessive)}')
        for idx, cls, total_num in sorted(excessive, key=lambda x: x[2], reverse=True)[:10]:
            print(f'  - {idx}: {total_num}')


# =========================
# 入口
# =========================
if __name__ == '__main__':
    all_classes = load_all_classes(CLASS_TXT)
    analyze_merged_dataset(DATASET_PATHS, all_classes)
