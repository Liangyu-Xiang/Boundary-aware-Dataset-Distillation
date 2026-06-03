import os
import shutil

# =========================
# 配置区（你只需要改这里）
# =========================
FULL_DATASET_ROOT = "/data/mmc_lyxiang/dataset/SRe2L/sre2l_in1k_rn18_4k_ipc200/"     # 含 new000~new999
TARGET_ROOT = "/data/mmc_lyxiang/dataset/SRe2L/synthetic_imagenette_IPC200/"

CLASS_INDICES_PATH = "./misc/class_indices.txt"   # nxxxxx, 顺序对应 new000~new999
CLASS_NETTE_PATH = "./misc/class_nette.txt"       # ImageNette synset list

# 是否真的拷贝（False=只打印，True=执行）
DO_COPY = True


def read_lines(path):
    with open(path, "r") as f:
        return [line.strip() for line in f if line.strip()]


def main():
    os.makedirs(TARGET_ROOT, exist_ok=True)

    # 1. 读取 class_indices：index → synset
    index_to_synset = read_lines(CLASS_INDICES_PATH)

    # sanity check
    assert len(index_to_synset) >= 1000, \
        f"class_indices.txt length seems wrong: {len(index_to_synset)}"

    # 2. 读取 ImageNette synset list
    nette_synsets = set(read_lines(CLASS_NETTE_PATH))

    print(f"[INFO] ImageNette classes: {len(nette_synsets)}")

    # 3. 建立 synset → newXXX 映射
    selected = []
    for idx, synset in enumerate(index_to_synset):
        if synset in nette_synsets:
            new_dir = f"new{idx:03d}"
            src = os.path.join(FULL_DATASET_ROOT, new_dir)
            dst = os.path.join(TARGET_ROOT, synset)
            selected.append((src, dst))

    print(f"[INFO] Matched classes: {len(selected)}")

    # 4. 执行拷贝
    for src, dst in selected:
        if not os.path.exists(src):
            print(f"[WARN] Missing source dir: {src}")
            continue

        if os.path.exists(dst):
            print(f"[SKIP] Target exists: {dst}")
            continue

        print(f"[COPY] {src} → {dst}")
        if DO_COPY:
            shutil.copytree(src, dst)

    print("[DONE] ImageNette subset extraction finished.")


if __name__ == "__main__":
    main()
