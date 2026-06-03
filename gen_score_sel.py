"""
Generate–Score–Select: Sample with DiT, score by EDL (Dirichlet), select by Top-r + k-center.
"""
import os, json, math, argparse
import numpy as np
import torch
import shutil
import torch.nn.functional as F
from torchvision.utils import save_image
from tqdm import tqdm

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

from diffusion import create_diffusion
from diffusers.models import AutoencoderKL
from download import find_model
from models import DiT_models
from resnet import resnet18  # 你已有的教师骨干（下方假设其head输出Dirichlet α）
from torchvision.transforms import v2

# -------------------------------
# Utils
# -------------------------------

def set_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)

@torch.no_grad()
def vae_decode(vae, latents):
    # Stable Diffusion VAE 0.18215 标准缩放
    images = vae.decode(latents / 0.18215).sample
    return images

def ensure_dir(path):
    os.makedirs(path, exist_ok=True)
    return path

# ----- EDL/Dirichlet 打分相关 -----

@torch.no_grad()
def dirichlet_from_teacher(teacher, x):
    """
    假设 teacher(x) 直接输出 Dirichlet α (B, K)。
    若你的 teacher 输出不是 α，请在这里改为 softplus(head)+1 的形式。
    """
    logits = teacher(x)[0]  # (B, K)
    alpha = torch.exp(logits) + torch.exp(torch.tensor(-1.43))
    return alpha

@torch.no_grad()
def evidential_score_from_alpha(alpha, y=None, lambda1=0.5, lambda2=0.5, eps=1e-8):
    """
    s(x) = λ1 * H( p_hat ) / S  +  λ2 * ||p_hat - onehot(y)|| / S
    其中 p_hat = α / S, S = sum α
    """
    S = alpha.sum(dim=-1, keepdim=True)                    # (B,1)
    p = alpha / (S + eps)                                  # (B,K)
    # 熵
    entropy = -(p * (p.clamp_min(eps).log())).sum(dim=-1, keepdim=True)  # (B,1)
    # 与 one-hot 的冲突度
    if y is not None:
        y_onehot = F.one_hot(y, num_classes=p.shape[-1]).float()
        conflict = ((p - y_onehot) ** 2).sum(dim=-1, keepdim=True).sqrt()
    else:
        # 若无标签，也可用 (p - max_onehot) 近似
        top = torch.argmax(p, dim=-1)
        y_onehot = F.one_hot(top, num_classes=p.shape[-1]).float()
        conflict = ((p - y_onehot) ** 2).sum(dim=-1, keepdim=True).sqrt()

    s = lambda1 * (entropy / (S + eps)) + lambda2 * (conflict / (S + eps))
    return s.squeeze(-1), p.squeeze(1) if p.dim() == 3 else p  # (B,), (B,K)

class FeatureHook:
    def __init__(self, module):
        self.feat = None
        self.h = module.register_forward_hook(self.hook)
    def hook(self, m, inp, out):
        # ResNet18 的 avgpool 输出是 (B, 512, 1, 1)
        self.feat = torch.flatten(out, 1).detach()
    def close(self):
        self.h.remove()

@torch.no_grad()
def k_center_select(features: np.ndarray, m: int) -> list:
    """
    简洁的 k-center 贪心选择。features: (N, D) numpy。
    返回选择的索引列表（长度 m）。
    """
    N = features.shape[0]
    if m >= N:
        return list(range(N))
    centers = [np.random.randint(N)]
    dist = np.linalg.norm(features - features[centers[0]][None, :], axis=1)
    for _ in range(1, m):
        idx = int(np.argmax(dist))
        centers.append(idx)
        dist = np.minimum(dist, np.linalg.norm(features - features[idx][None, :], axis=1))
    return centers

def save_metadata(meta_path, items):
    """
    items: list of dict
    """
    with open(meta_path, 'w', encoding='utf-8') as f:
        for it in items:
            f.write(json.dumps(it, ensure_ascii=False) + "\n")

# -------------------------------
# 主流程
# -------------------------------

def main(args):
    set_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # ----- class 列表与选择子集 -----
    with open('./misc/class_indices.txt', 'r') as fp:
        all_classes = [line.strip() for line in fp.readlines()]
    if args.spec == 'woof':
        file_list = './misc/class_woof.txt'
    elif args.spec == 'nette':
        file_list = './misc/class_nette.txt'
    else:
        file_list = './misc/class100.txt'
    with open(file_list, 'r') as fp:
        sel_classes = [line.strip() for line in fp.readlines()]

    phase = max(0, args.phase)
    cls_from = args.nclass * phase
    cls_to = args.nclass * (phase + 1)
    sel_classes = sel_classes[cls_from:cls_to]
    class_labels = [all_classes.index(c) for c in sel_classes]

    # ----- 加载 DiT & Diffusion & VAE -----
    latent_size = args.image_size // 8
    dit = DiT_models[args.model](input_size=latent_size, num_classes=args.num_classes).to(device)
    ckpt_path = args.ckpt or f"DiT-XL-2-{args.image_size}x{args.image_size}.pt"
    dit.load_state_dict(find_model(ckpt_path), strict=False)
    dit.eval()
    diffusion = create_diffusion(str(args.num_sampling_steps))

    vae_path = f"./pretrained_models/stabilityai/sd-vae-ft-{args.vae}"
    vae = AutoencoderKL.from_pretrained(vae_path).to(device)

    # ----- 加载 EDL 教师 -----
    teacher = resnet18(pretrained=False).to(device)
    if args.teacher_ckpt is not None:
        state_dict = torch.load(args.teacher_ckpt, map_location=device)
        # 你的权重格式里可能是 {"model": ...}
        if isinstance(state_dict, dict) and "model" in state_dict:
            state_dict = state_dict["model"]
        teacher.load_state_dict(state_dict, strict=False)
    teacher.eval()

    # 挂一个特征钩子（以 ResNet18 为例：avgpool 输出作为特征）
    # 如果你的 EDL 模型结构不同，请改到合适层
    # 下面假设 teacher.avgpool 存在：
    feat_hook = FeatureHook(teacher.avgpool if hasattr(teacher, "avgpool") else list(teacher.children())[-2])

    # ----- 目录 -----
    save_root = ensure_dir(args.save_dir)
    meta_root = ensure_dir(os.path.join(save_root, "_meta"))

    # -----------------------------
    # Generate – Score – Select
    # -----------------------------
    B = args.batch_size
    dit_cfg = args.cfg_scale
    woof_indices = torch.tensor([155, 159, 162, 167, 182, 193, 207, 229, 258, 273], device=device)
    confusion_matrix = np.load("/data/mmc_lyxiang/DD/MinimaxDiffusion/results/test/confusion_matrix_epoch_0.npy")
    confusion_matrix = torch.tensor(confusion_matrix, device=device)
    for class_label, sel_class in zip(class_labels, sel_classes):
        print(f"[Class] {sel_class} ({class_label})")

        # 1) Generate: 生成候选池
        cand_imgs, cand_scores, cand_feats, cand_soft = [], [], [], []
        cand_paths = []
        cls_dir = ensure_dir(os.path.join(save_root, sel_class))

        # 核心数据集保存路径
        core_root = ensure_dir(save_root + "_core")
        core_dir = ensure_dir(os.path.join(core_root, sel_class))

        n_to_generate = args.candidates_per_class
        n_batches = math.ceil(n_to_generate / B)


        for bi in tqdm(range(n_batches), desc=f"Generate {sel_class}"):
            cur_bs = B if (bi < n_batches - 1 or n_to_generate % B == 0) else (n_to_generate % B)
            # 初始噪声
            z = torch.randn(cur_bs, 4, latent_size, latent_size, device=device)

            # 类条件混合（随概率触发），用于“靠边界”
            y_main = torch.full((cur_bs,), class_label, device=device, dtype=torch.long)
            if np.random.rand() >= args.mix_prob:
                y_use = y_main.unsqueeze(0)  # (1, cur_bs)
                cfg_use = dit_cfg
                confusion_rate = None

            # 随机选一个非本类的类作为对手类
            mask = woof_indices != torch.tensor(class_label, device=device)
            candidates = woof_indices[mask]
            random_indices = torch.randperm(len(candidates), device=device)[:1]
            y_other = candidates[random_indices]
            # DiT 里用“条件混合”需要你在 forward_with_cfg 内支持；这里用简单策略：batch 内一半换成other
            # 也可通过传额外的条件嵌入实现，这里保持简单：随机替换一部分标签
            y_mixed = y_main.clone().unsqueeze(0)
            y_use = torch.cat([y_mixed, y_other.unsqueeze(0)], dim=0)  # (2, cur_bs)
            class_indices = (woof_indices == class_label).nonzero(as_tuple=True)[0].item()
            confusion_rate = confusion_matrix[class_indices, random_indices] / (confusion_matrix[class_indices, random_indices] + confusion_matrix[random_indices, class_indices])

            cfg_use = max(2.0, min(dit_cfg, 3.0))  # 降 CFG 提升熵

            # classifier-free guidance 拼 batch
            z_in = torch.cat([z, z], 0)
            y_null = torch.full((cur_bs,), 1000, device=device, dtype=torch.long)  # null token=1000
            y_in = torch.cat([y_use, y_null.unsqueeze(0)], 0)
            if confusion_rate:
                y = torch.cat([y, confusion_rate.unsqueeze(0)], 0)# 最后一个元素是混淆率作为权重
            model_kwargs = dict(y=y_in, cfg_scale=cfg_use)

            # 扩散采样
            latents = diffusion.p_sample_loop(
                dit.forward_with_cfg, z_in.shape, z_in, clip_denoised=False,
                model_kwargs=model_kwargs, progress=False, device=device
            )
            latents, _ = latents.chunk(2, dim=0)  # 去掉 null 分支
            images = vae_decode(vae, latents)     # (cur_bs, 3, H, W)

            # 2) Score: 用 EDL teacher 打分 & 特征
            # 先做 teacher 前向（保证与训练时同样的预处理/归一化；这里假设已是 [-1,1]）
            # 如需 ImageNet 标准化，请在此处添加 normalize
            teacher_transform = v2.Compose([
                        v2.Resize(256),
                        v2.CenterCrop(224),
                        v2.Normalize(mean=[0.485, 0.456, 0.406],
                                    std=[0.229, 0.224, 0.225])
                    ])
            input = teacher_transform(images)
            alpha = dirichlet_from_teacher(teacher, input)
            s, p_soft = evidential_score_from_alpha(alpha, y=y_main, lambda1=args.lambda1, lambda2=args.lambda2)

            # 抽取特征（来自 feat_hook）
            _ = teacher(input)  # 触发 hook
            feats = feat_hook.feat  # (cur_bs, D)

            # 记录到候选池（为了节省显存，转 CPU）
            for j in range(cur_bs):
                # 临时保存图片到候选目录（可选）：也可以先不落盘，仅元数据+tensor
                fname = f"cand_{bi*B + j + args.total_shift}.png"
                fpath = os.path.join(cls_dir, fname)
                save_image(images[j], fpath, normalize=True, value_range=(-1, 1))

                cand_paths.append(fpath)
                cand_scores.append(float(s[j].detach().cpu()))
                cand_feats.append(feats[j].detach().cpu().numpy())
                cand_soft.append(p_soft[j].detach().cpu().numpy())

        # 3) Select: Top-r + k-center
        cand_scores_np = np.array(cand_scores)
        idx_sorted = np.argsort(-cand_scores_np)  # 降序
        r = max(args.core_per_class * args.topr_factor, args.core_per_class)  # Top-r
        idx_top = idx_sorted[:r]

        feats_top = np.stack([cand_feats[i] for i in idx_top], axis=0)
        pick_local = k_center_select(feats_top, args.core_per_class)
        selected_idx = [int(idx_top[i]) for i in pick_local]

        # 保存核心集的元数据（包含软标签、分数、路径）
        # === Save Selected Core Images ===
        core_root = ensure_dir(save_root + "_core")
        core_dir = ensure_dir(os.path.join(core_root, sel_class))
        os.makedirs(core_dir, exist_ok=True)

        print(f"Saving {len(selected_idx)} core samples for class [{sel_class}] → {core_dir}")
        for new_i, i in enumerate(selected_idx):
            src_path = cand_paths[i]          # 原始候选图片路径
            dst_path = os.path.join(core_dir, f"{new_i:04d}.png")
            shutil.copy(src_path, dst_path)   # 拷贝文件（保留像素）

        print(f"[{sel_class}] Core subset saved. ({len(selected_idx)} images)")

    # 清理 hook
    feat_hook.close()
    print("Done Generate–Score–Select.")

# -------------------------------
# CLI
# -------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # 你的原始参数
    parser.add_argument("--model", type=str, choices=list(DiT_models.keys()), default="DiT-XL/2")
    parser.add_argument("--vae", type=str, choices=["ema", "mse"], default="mse")
    parser.add_argument("--image-size", type=int, choices=[256, 512], default=256)
    parser.add_argument("--num-classes", type=int, default=1000)
    parser.add_argument("--cfg-scale", type=float, default=4.0)
    parser.add_argument("--num-sampling-steps", type=int, default=50)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--ckpt", type=str, default=None)
    parser.add_argument("--spec", type=str, default='none')
    parser.add_argument("--save-dir", type=str, default='../logs/test')
    parser.add_argument("--total-shift", type=int, default=0)
    parser.add_argument("--nclass", type=int, default=10)
    parser.add_argument("--phase", type=int, default=0)

    # 新增：Generate–Score–Select 所需
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--candidates-per-class", type=int, default=2000, help="候选池大小 K_y")
    parser.add_argument("--core-per-class", type=int, default=50, help="最终核心集 m_y")
    parser.add_argument("--topr-factor", type=int, default=10, help="Top-r = factor * core_per_class")

    # 类条件混合（靠边界）
    parser.add_argument("--mix-prob", type=float, default=0.5, help="以该概率启用类条件混合")
    parser.add_argument("--mix-alpha", type=float, default=0.7, help="α∈[0,1]，越小越多用对手类标签（简单近似）")

    # EDL teacher & 打分权重
    parser.add_argument("--teacher-ckpt", type=str, default=None, help="EDL 教师权重路径（输出Dirichlet α）")
    parser.add_argument("--lambda1", type=float, default=0.5, help="s(x) 中熵项系数")
    parser.add_argument("--lambda2", type=float, default=0.5, help="s(x) 中冲突度项系数")

    args = parser.parse_args()
    main(args)
