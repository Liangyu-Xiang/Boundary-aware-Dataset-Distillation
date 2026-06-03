import sys
import shutil
import torch
import torch.nn as nn
# the first flag below was False when we tested this script but True makes A100 training a lot faster:
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torchvision import transforms
from torchvision.transforms import v2
import torch.nn.functional as F
import numpy as np
from collections import OrderedDict, defaultdict
from PIL import Image
from copy import deepcopy
from glob import glob
from time import time
import argparse
import logging
import os
import train_models.resnet as RN
from data import ImageFolder
from models import DiT_models
from download import find_model
from diffusion import create_diffusion
from diffusers.models import AutoencoderKL
from resnet import resnet18
from data import transform_imagenet


def pgd_boundary_distance(model,x,y,eps=1/255,alpha=0.5/255,max_steps=20,clip_min=0.0,clip_max=1.0,norm="linf",random_start=False, preprocess=None):
    device = x.device
    B = x.shape[0]
    x = (x+1) / 2 # [-1, 1] to [0, 1]
    if random_start:
        if norm == "linf":
            x_adv = x + torch.empty_like(x).uniform_(-eps, eps)
        else:
            x_adv = x.clone()
    else:
        x_adv = x.clone().detach()
    
    x_adv = x_adv.clamp(clip_min, clip_max)

    # distances=0 表示还没攻破；一旦被攻破就记录k
    distances = torch.zeros(B, dtype=torch.long, device=device)
    boundary_class = torch.full((B,), -1, dtype=torch.long, device=device)

    for k in range(1, max_steps + 1):
        x_adv.requires_grad_(True)
        if preprocess:
            x_input = preprocess(x_adv)
        else:
            x_input = x_adv
        logits = model(x_input)
        y = torch.as_tensor(y, dtype=torch.long, device=logits.device)
        loss = F.cross_entropy(logits, y, reduction="sum")

        grad = torch.autograd.grad(loss, x_adv)[0]

        with torch.no_grad():
            if norm == "linf":
                x_adv = x_adv + alpha * grad.sign()
                # 投影到 x 的 eps-ball 内
                delta = torch.clamp(x_adv - x, min=-eps, max=eps)
                x_adv = (x + delta).clamp(clip_min, clip_max)
            elif norm == "l2":
                # L2 版本（可选）
                grad_norm = grad.view(B, -1).norm(p=2, dim=1).view(B, 1, 1, 1)
                grad_normalized = grad / (grad_norm + 1e-8)
                x_adv = x_adv + alpha * grad_normalized
                delta = x_adv - x
                delta_norm = delta.view(B, -1).norm(p=2, dim=1).view(B, 1, 1, 1)
                factor = torch.clamp(delta_norm, max=eps) / (delta_norm + 1e-8)
                delta = delta * factor
                x_adv = (x + delta).clamp(clip_min, clip_max)
            else:
                raise ValueError(f"Unsupported norm: {norm}")
    
            # 当前预测
            preds = logits.argmax(dim=1)
            misclassified = (preds != y) & (distances == 0)

            # 第一次被攻破的样本记录下步数k
            distances[misclassified] = k

             # ⭐ 记录边界类别（首次错分的类别）
            boundary_class[misclassified] = preds[misclassified]

            # 如果已经全部被攻破，可以提前结束
            if (distances > 0).all():
                break
    
    # 对那些 max_steps 内都没攻破的样本，设为 max_steps+1
    distances[distances == 0] = max_steps + 1
    boundary_class[boundary_class == -1] = -1

    return distances, boundary_class, x_adv.detach()

def geometric_boundary_distance(
    model,
    x,
    y,
    preprocess=None,
    eps: float = 1e-6,
):
    """
    连续几何边界距离（替代 PGD 版本）

    返回:
        distances: (B,) 连续边界距离
        boundary_class: (B,) 最近的竞争类别
        x: 原图（保持接口一致）
    """

    device = x.device
    B = x.shape[0]
    x = x.detach()
    # ✅ [-1,1] → [0,1]
    x = (x + 1) / 2
    x = x.clamp(0, 1)
    x.requires_grad_(True)

    if preprocess:
        x_input = preprocess(x)
    else:
        x_input = x

    # ===============================
    # 1️⃣ 前向：计算 logits
    # ===============================
    logits = model(x_input)              # (B, C)
    y = torch.as_tensor(y, dtype=torch.long, device=logits.device)


    evidence = F.softplus(logits)
    alpha = evidence + 1.0 
    S = alpha.sum(dim=1, keepdim=True)
    probs = alpha / S

    # ===============================
    # 2️⃣ 找最近竞争类别
    # ===============================
    with torch.no_grad():
        tmp = logits.clone()
        tmp[torch.arange(B), y] = -1e9
        boundary_class = tmp.argmax(dim=1)

    # ===============================
    # 3️⃣ 构造 margin 函数 g(x)
    # ===============================
    f_y = logits[torch.arange(B), y]
    f_b = logits[torch.arange(B), boundary_class]
    g = f_y - f_b                         # (B,)

    # ===============================
    # 4️⃣ 计算 ∇x g(x)
    # ===============================
    grad = torch.autograd.grad(
        outputs=g.sum(),
        inputs=x,
        create_graph=False
    )[0]

    grad_norm = grad.view(B, -1).norm(p=2, dim=1) + eps

    # ===============================
    # ✅✅✅ 5️⃣ 连续几何距离（核心公式）
    # ===============================
    distances = (g.abs() / grad_norm).detach()

    return distances, boundary_class.detach(), x.detach(), probs.detach()

def edl_entropy_boundary_distance(
    model,
    x,
    y,
    preprocess=None,
    eps: float = 1e-6,
):
    """
    ✅ 使用 E_p[ H(P(y|π)) ] 作为边界距离的 EDL 版本

    返回:
        distances: (B,)  EDL 期望熵边界距离（越大越靠近边界）
        boundary_class: (B,) 最近竞争类别（基于 logits）
        probs: (B, C)    Dirichlet 后验期望概率
        alpha: (B, C)    Dirichlet 参数
        x: 原图
    """

    device = x.device
    B = x.shape[0]
    x = x.detach()

    # ✅ [-1,1] → [0,1]
    x = (x + 1) / 2
    x = x.clamp(0, 1)
    x.requires_grad_(True)

    if preprocess:
        x_input = preprocess(x)
    else:
        x_input = x

    # ===============================
    # 1️⃣ 前向：EDL 输出
    # ===============================
    logits = model(x_input)              # (B, C)

    # ✅ softplus 证据
    evidence = F.softplus(logits)
    alpha = evidence + 1.0               # (B, C)
    alpha0 = alpha.sum(dim=1, keepdim=True)  # (B, 1)

    # ✅ Dirichlet 期望概率
    probs = alpha / alpha0               # (B, C)

    y = torch.as_tensor(y, dtype=torch.long, device=logits.device)

    # ===============================
    # 2️⃣ 最近竞争类别（用于统计，不影响熵）
    # ===============================
    with torch.no_grad():
        tmp = logits.clone()
        tmp[torch.arange(B), y] = -1e9
        boundary_class = tmp.argmax(dim=1)

    # ===============================
    # ✅✅✅ 3️⃣ EDL 期望熵边界距离（你的新公式）
    # ===============================
    # digamma(α_k + 1) - digamma(α_0 + 1)
    term = torch.digamma(alpha + 1.0) - torch.digamma(alpha0 + 1.0)

    # E_p[ H(P(y|π)) ]
    distances = - torch.sum((alpha / alpha0) * term, dim=1)

    distances = distances.detach()      # (B,)

    return (
        distances,                      # ✅ 新的“概率边界距离”
        boundary_class.detach(),        # (B,)
        probs.detach(),                 # (B, C)
        alpha.detach(),                 # (B, C)
        x.detach()
    )


def center_crop_arr(pil_image, image_size):
    """
    Center cropping implementation from ADM.
    https://github.com/openai/guided-diffusion/blob/8fb3ad9197f16bbc40620447b2742e13458d2831/guided_diffusion/image_datasets.py#L126
    """
    while min(*pil_image.size) >= 2 * image_size:
        pil_image = pil_image.resize(
            tuple(x // 2 for x in pil_image.size), resample=Image.BOX
        )

    scale = image_size / min(*pil_image.size)
    pil_image = pil_image.resize(
        tuple(round(x * scale) for x in pil_image.size), resample=Image.BICUBIC
    )

    arr = np.array(pil_image)
    crop_y = (arr.shape[0] - image_size) // 2
    crop_x = (arr.shape[1] - image_size) // 2
    return Image.fromarray(arr[crop_y: crop_y + image_size, crop_x: crop_x + image_size])


if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"

    expert_model = RN.ResNet('imagenet',
                          18,
                          10,
                          norm_type='instance',
                          size=224,
                          nch=3).to(device)
    expert_path = "/data/mmc_lyxiang/DD/MinimaxDiffusion/results/imagenet10/resnet18in_EDL_NLL_noKL_cut/model_best.pth.tar"
    state_dict = torch.load(expert_path)["state_dict"]
    # expert_model = resnet18(pretrained=False).to(device)
    # expert_path = "/data/mmc_lyxiang/KD/EKD/output/Evidential_Teacher/ResNet18_ImageNet/student_best"
    # state_dict = torch.load(expert_path)["model"]
    expert_model.load_state_dict(state_dict)
    expert_model.eval()
    for p in expert_model.parameters():
        p.requires_grad = False  # 不计算分类器的梯度

    expert_transform, _ =  transform_imagenet(augment=False,
                                size=224,
                                from_tensor=True)
    
    data_path = "/data/mmc_lyxiang/DD/MinimaxDiffusion/results/dit-distillation/imagenet-10-1000-average_embed/"
    data_path = "/data/mmc_lyxiang/DD/MinimaxDiffusion/results/dit-distillation/high_data_u_from_ImageNetWoof/"
    transform = transforms.Compose([
        transforms.Lambda(lambda pil_image: center_crop_arr(pil_image, 256)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], inplace=True)
    ])
    dataset = ImageFolder(data_path, transform=transform, nclass=10,
                          ipc=-1, spec='woof', phase=0,
                          seed=0, return_origin=True)
    loader = DataLoader(
        dataset,
        batch_size=1,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
        drop_last=True
    )

    all_distances = []
    all_wrong_probs = []
    all_wrong_distances = []
    total_samples = 0
    correct_samples = 0
    save_root = "/data/mmc_lyxiang/DD/MinimaxDiffusion/results/dit-distillation/test/"
    os.makedirs(save_root, exist_ok=True)



    for x, ry, y, idx in loader:     
            
            ry = ry.numpy()
            x = x.to(device)
            y = y.to(device)

            # bound_dis, bound_class, _, probs = geometric_boundary_distance(expert_model, x, ry, preprocess=expert_transform)
            bound_dis, bound_class, probs, alpha, _ = edl_entropy_boundary_distance(
                expert_model, x, ry, preprocess=expert_transform
            )
            d_film = (bound_dis - 1.0) / (1.5 - 1.0 + 1e-6)
            d_film = torch.clamp(d_film, 0.0, 1.0) 
            exp_bound_dis = torch.exp(bound_dis / 0.5)
            

            print("EXP BOUND DISTANCE:", exp_bound_dis.item())
            print("BOUND DSTANCE:", bound_dis.item())
            pred = probs.argmax(dim=1)
            ry_tensor = torch.from_numpy(ry).long().to(device)
            wrong_mask = pred != ry_tensor

            # ✅ 统计分类准确率
            total_samples += ry_tensor.numel()
            correct_samples += (pred == ry_tensor).sum().item()

            wrong_bound_dis = exp_bound_dis[wrong_mask]
            wrong_probs = probs[wrong_mask]

            all_wrong_distances.extend(wrong_bound_dis.cpu().tolist())
            all_distances.extend(d_film.cpu().tolist())
            all_wrong_probs.extend(wrong_probs.cpu().tolist())

            # 2️⃣ 按 batch 内选 Top-K 高不确定样本（或阈值）
            threshold = 1.0   # 例如取 top 10%
            high_mask = d_film >= threshold

            idx = idx.numpy()  # dataset 下标

            for i in range(len(idx)):
                if high_mask[i]:
                    dataset_idx = idx[i]
                    img_path, _ = dataset.samples[dataset_idx]

                    # ✅ 复制到新目录（按原始类别分文件夹）
                    class_name = dataset.classes[ry[i]]
                    save_dir = os.path.join(save_root, class_name)
                    os.makedirs(save_dir, exist_ok=True)

                    save_path = os.path.join(save_dir, os.path.basename(img_path))
                    shutil.copy(img_path, save_path)
    
    # print(all_distances)
    accuracy = correct_samples / (total_samples + 1e-8)
    print(f"\n✅ Classification Accuracy: {accuracy * 100:.2f}%")
    print(f"✅ Correct: {correct_samples} / Total: {total_samples}")
