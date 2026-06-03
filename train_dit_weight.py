"""
Fine-tuning DiT with minimax criteria.
"""
import sys
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


#################################################################################
#                             Training Helper Functions                         #
#################################################################################

@torch.no_grad()
def update_ema(ema_model, model, decay=0.9999):
    """
    Step the EMA model towards the current model.
    """
    ema_params = OrderedDict(ema_model.named_parameters())
    model_params = OrderedDict(model.named_parameters())

    for name, param in model_params.items():
        # TODO: Consider applying only to params that require_grad to avoid small numerical changes of pos_embed
        ema_params[name].mul_(decay).add_(param.data, alpha=1 - decay)


def requires_grad(model, flag=True):
    """
    Set requires_grad flag for all parameters in a model.
    """
    for p in model.parameters():
        p.requires_grad = flag


def cleanup():
    """
    End DDP training.
    """
    dist.destroy_process_group()


def create_logger(logging_dir):
    """
    Create a logger that writes to a log file and stdout.
    """
    if dist.get_rank() == 0:  # real logger
        logger = logging.getLogger(__name__)
        logger.propagate = False
        logger.setLevel(logging.INFO)
        ch = logging.StreamHandler(stream=sys.stdout)
        ch.setLevel(logging.INFO)
        formatter = logging.Formatter('[%(asctime)s] %(message)s')
        ch.setFormatter(formatter)
        logger.addHandler(ch)
        if logging_dir:
            fh = logging.FileHandler(f'{logging_dir}/logs.txt')
            fh.setLevel(logging.INFO)
            fh.setFormatter(formatter)
            logger.addHandler(fh)
    else:  # dummy logger (does nothing)
        logger = logging.getLogger(__name__)
        logger.addHandler(logging.NullHandler())
    return logger


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


def cosine_similarity(ta, tb):
    bs1, bs2 = ta.shape[0], tb.shape[0]
    frac_up = torch.matmul(ta, tb.T)
    frac_down = torch.norm(ta, dim=-1).view(bs1, 1).repeat(1, bs2) * \
                torch.norm(tb, dim=-1).view(1, bs2).repeat(bs1, 1)
    return frac_up / frac_down


def mark_difffit_trainable(model, is_bitfit=False):
    """
    Mark the parameters that require updating by difffit.
    """
    if is_bitfit:
        trainable_names = ['bias']
    else:
        # Advise by GPT
        # trainable_names = ["bias", "y_embed", "scale", "adapter"]

        # Original
        trainable_names = ["bias", "norm", "gamma", "y_embed"]

    for par_name, par_tensor in model.named_parameters():
        par_tensor.requires_grad = any([kw in par_name for kw in trainable_names])
    return model

def dempster_combination(belief, u):
    num,K = belief.shape
    # b_copy = belief.copy()
    # combined_belief = torch.zeros(K)
    # conflict_term = 0.0
    b_1, u_1 = belief[:1,:], u[:1, :]
    combined_belief, combined_u = belief[:1,:], u[:1, :]
    # b_2, u_2 = belief[1], u[1]
    for i in range(num - 1):
        b_2, u_2 = belief[i+1:i+2,:], u[i+1]
        combined_belief = b_1 * b_2 + b_1 * u_2 + b_2 * u_1
        conflict_term = b_1.T @ b_2
        mask = torch.eye(conflict_term.shape[0], device=belief.device)
        conflict_term = conflict_term * (1 - mask)
        conflict_term = conflict_term.sum()
        combined_belief = combined_belief / (1 - conflict_term)
        combined_u = u_1 * u_2 / (1 - conflict_term)
        b_1, u_1 = combined_belief, combined_u
    return combined_belief, combined_u
        

def evidential_weight(alpha, y):
    S = alpha.sum(dim=-1, keepdim=True)
    p = alpha / S
    entropy = -(p * (p.clamp_min(1e-8).log())).sum(dim=-1, keepdim=True)
    # optional: Euclidean distance to one-hot
    y_onehot = F.one_hot(y, num_classes=p.shape[-1]).float()
    conflict = ((p - y_onehot)**2).sum(dim=-1, keepdim=True).sqrt()
    s = 0.5 * (entropy / S) + 0.5 * (conflict / S)
    s = (s / s.mean()).detach()   # normalize for stability
    return s

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


#################################################################################
#                                  Training Loop                                #
#################################################################################

def main(args):
    """
    Fine-tune a DiT model.
    """
    assert torch.cuda.is_available(), "Training currently requires at least one GPU."

    # Setup DDP:
    dist.init_process_group("nccl")
    assert args.global_batch_size % dist.get_world_size() == 0, f"Batch size must be divisible by world size."
    rank = dist.get_rank()
    device = rank % torch.cuda.device_count()
    seed = args.global_seed * dist.get_world_size() + rank
    torch.manual_seed(seed)
    torch.cuda.set_device(device)
    print(f"Starting rank={rank}, seed={seed}, world_size={dist.get_world_size()}.")

    # Setup an experiment folder:
    if rank == 0:
        os.makedirs(args.results_dir, exist_ok=True)  # Make results folder (holds all experiment subfolders)
        experiment_index = len(glob(f"{args.results_dir}/*"))
        model_string_name = args.model.replace("/", "-")  # e.g., DiT-XL/2 --> DiT-XL-2 (for naming folders)
        experiment_dir = f"{args.results_dir}/{experiment_index:03d}-{model_string_name}"  # Create an experiment folder
        if args.tag:
            experiment_dir += f"-{args.tag}"
        checkpoint_dir = f"{experiment_dir}/checkpoints"  # Stores saved model checkpoints
        os.makedirs(checkpoint_dir, exist_ok=True)
        logger = create_logger(experiment_dir)
        logger.info(f"Experiment directory created at {experiment_dir}")
    else:
        logger = create_logger(None)

    # Create model:
    assert args.image_size % 8 == 0, "Image size must be divisible by 8 (for the VAE encoder)."
    latent_size = args.image_size // 8
    model = DiT_models[args.model](
        input_size=latent_size,
        num_classes=args.num_classes
    )
    # Load pretrained model:
    ckpt_path = args.ckpt
    state_dict = find_model(ckpt_path)
    model.load_state_dict(state_dict, strict=False)
    # GPT: Freeze normalization and positional embeddings
    # for name, module in model.named_modules():
    #     if isinstance(module, torch.nn.LayerNorm):
    #         module.eval()
    #         for p in module.parameters():
    #             p.requires_grad = False
    # for name, param in model.named_parameters():
    #     if "pos_embed" in name or "norm" in name:
    #         param.requires_grad = False
    # Note that parameter initialization is done within the DiT constructor
    ema = deepcopy(model).to(device)  # Create an EMA of the model for use after training
    requires_grad(ema, False)
    diffusion = create_diffusion(timestep_respacing="")  # default: 1000 steps, linear noise schedule
    vae_path = f"./pretrained_models/stabilityai/sd-vae-ft-{args.vae}"
    vae = AutoencoderKL.from_pretrained(vae_path).to(device)
    vae.eval()
    logger.info(f"DiT Parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Setup optimizer (we used default Adam betas=(0.9, 0.999) and a constant learning rate of 1e-3 in the Difffit paper):
    model = mark_difffit_trainable(model)
    model = DDP(model.to(device), device_ids=[rank])
    params_to_optimize = [p for p in model.parameters() if p.requires_grad]
    total_params = sum(p.numel() for p in params_to_optimize)
    print(f"Number of Trainable Parameters: {total_params * 1.e-6:.2f} M")
    opt = torch.optim.AdamW(params_to_optimize, lr=1e-3, weight_decay=0)


    # Load expert model (e.g. ResNet18)
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

    # transform for experts trained by EKD scripts
    # expert_transform = v2.Compose([
    #                     v2.Resize(256),
    #                     v2.CenterCrop(224),
    #                     v2.Normalize(mean=[0.485, 0.456, 0.406],
    #                                 std=[0.229, 0.224, 0.225])
    #                 ])
    # transform for experts trained by train.py in this project
    expert_transform, _ =  transform_imagenet(augment=False,
                                size=224,
                                from_tensor=True)

    # Setup data:
    transform = transforms.Compose([
        transforms.Lambda(lambda pil_image: center_crop_arr(pil_image, args.image_size)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], inplace=True)
    ])
    dataset = ImageFolder(args.data_path, transform=transform, nclass=args.nclass,
                          ipc=args.finetune_ipc, spec=args.spec, phase=args.phase,
                          seed=0, return_origin=True)
    sampler = DistributedSampler(
        dataset,
        num_replicas=dist.get_world_size(),
        rank=rank,
        shuffle=True,
        seed=args.global_seed
    )
    loader = DataLoader(
        dataset,
        batch_size=int(args.global_batch_size // dist.get_world_size()),
        shuffle=False,
        sampler=sampler,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs * len(loader))
    logger.info(f"Dataset contains {len(dataset):,} images ({args.data_path})")

    # Prepare models for training:
    update_ema(ema, model.module, decay=0)  # Ensure EMA is initialized with synced weights
    requires_grad(ema, False)
    model.train()  # important! This enables embedding dropout for classifier-free guidance
    ema.eval()  # EMA model should always be in eval mode

    # Variables for monitoring/logging purposes:
    train_steps = 0
    log_steps = 0
    running_loss, running_loss_pos, running_loss_neg = 0, 0, 0
    start_time = time()
    real_memory = defaultdict(list)
    pseudo_memory = defaultdict(list)
    bound_label = [1000] * int(args.global_batch_size // dist.get_world_size())
    all_distances = []

    output_idx_to_class_label = []

    logger.info(f"Training for {args.epochs} epochs...")
    for epoch in range(args.epochs):
        sampler.set_epoch(epoch)
        logger.info(f"Beginning epoch {epoch}...")
        for x, ry, y in loader:     
            
        
            ry = ry.numpy()
            x = x.to(device)
            y = y.to(device)

            bound_dis, bound_class, _, probs = geometric_boundary_distance(expert_model, x, ry, preprocess=expert_transform)
            bound_dis_clamped = torch.clamp(bound_dis.float(), min=0.0, max=5.0)
            # bound_dis_pgd, bound_class_pgd, _ = pgd_boundary_distance(expert_model, x, ry, preprocess=expert_transform)
            # dis_gap = bound_dis_pgd.float() - bound_dis.float()
            # all_distances.append(bound_dis.cpu().numpy())

            with torch.no_grad():
                # input = x
                # logits = expert_model(expert_transform(input))
                # probs = F.softmax(logits, dim=-1)
                
                # evidence = torch.exp(logits)
                # alpha = evidence + torch.exp(torch.tensor(-1.43))
                # weights = evidential_weight(alpha, y)
                # top2_probs, top2_indices = probs.topk(2, dim=-1) 
                # top1_prob = top2_probs[:, 0]
                # top1_class = top2_indices[:, 0]
                # top2_prob = top2_probs[:, 1]
                # top2_class = top2_indices[:, 1]
                # in_class_mask = (top1_class == labels) & (top1_prob > top1_thresh)
                # in_class_indices = torch.nonzero(in_class_mask).squeeze(1)
                # inter_class_mask = (top1_class == labels) & ((top1_prob - top2_prob) < top2_gap_thresh)
                # inter_class_indices = torch.nonzero(inter_class_mask).squeeze(1)
                # ry_tensor = torch.tensor(ry, device=device)
                # wrong_mask = top1_class != ry_tensor
                # wrong_indices = torch.nonzero(wrong_mask).squeeze(1)
                # bound_label = torch.full_like(top2_class, 1000)
                # bound_mask = (top1_prob < 0.8) & (top2_prob > 0.1)
                # bound_label[bound_mask] = top2_class[bound_mask]
                # wrong_mask = torch.zeros_like(bound_label, dtype=torch.bool)
                # if wrong_indices.shape[0] > 0:
                #     wrong_mask[wrong_indices] = True
                # bound_label[wrong_mask] = top1_class[wrong_mask]
                # label_weight = torch.full_like(bound_label, 0.5, dtype=torch.float32)
                bound_label = [dataset.original_labels[idx.item()] for idx in bound_class]
                bound_label = torch.tensor(bound_label, device=device)

                label_weight = 0.5 * torch.exp( - 0.7 * (bound_dis_clamped - 0.25) )


            with torch.no_grad():
                # Map input images to latent space + normalize latents:
                x = vae.encode(x).latent_dist.sample().mul_(0.18215)
            t = torch.randint(0, diffusion.num_timesteps, (x.shape[0],), device=device)
            # y = torch.cat((y, bound_label, label_weight))
            y = torch.stack([y, bound_label, label_weight])
            model_kwargs = dict(y=y)
            loss_dict = diffusion.training_losses(model, x, t, model_kwargs)
            pseudo_embeddings = loss_dict["output"]
            loss = loss_dict["loss"].mean()

            # Calculate minimax criteria
            pos_match_loss = torch.tensor(0.).to(device)
            neg_match_loss = torch.tensor(0.).to(device)
            if args.condense:
                ry_set = set(ry)
                num_ry = len(ry_set)
                for c in ry_set:
                    
                    if len(pseudo_memory[c]):
                        pos_embeddings = torch.cat(real_memory[c]).flatten(start_dim=1)
                        neg_embeddings = torch.cat(pseudo_memory[c]).flatten(start_dim=1)
                        # Representativeness constraint
                        pos_feat_sim = 1 - cosine_similarity(
                            pseudo_embeddings[ry == c].flatten(start_dim=1), pos_embeddings
                        ).min()
                        # Diversity constraint
                        neg_feat_sim = cosine_similarity(
                            pseudo_embeddings[ry == c].flatten(start_dim=1), neg_embeddings
                        ).max()
                        pos_match_loss += pos_feat_sim * args.lambda_pos / num_ry
                        neg_match_loss += neg_feat_sim * args.lambda_neg / num_ry
                        

                    # Update the auxiliary memories
                    real_memory[c].extend(x[ry == c].detach().split(1))
                    pseudo_memory[c].extend(pseudo_embeddings[ry == c].detach().split(1))
                    while len(real_memory[c]) > args.memory_size:
                        real_memory[c].pop(0)
                    while len(pseudo_memory[c]) > args.memory_size:
                        pseudo_memory[c].pop(0)
                    

                all_loss = loss + pos_match_loss + neg_match_loss
            else:
                all_loss = loss


            opt.zero_grad()
            all_loss.backward()
            torch.nn.utils.clip_grad_norm_(params_to_optimize, max_norm=1.0)
            opt.step()
            # scheduler.step()
            update_ema(ema, model.module)

            # Log loss values:
            running_loss += loss.item()
            if pos_match_loss or neg_match_loss:
                running_loss_pos += pos_match_loss.item()
                running_loss_neg += neg_match_loss.item()

            log_steps += 1
            train_steps += 1
            if train_steps % args.log_every == 0:
                # Measure training speed:
                torch.cuda.synchronize()
                end_time = time()
                steps_per_sec = log_steps / (end_time - start_time)
                # Reduce loss history over all processes:
                avg_loss = torch.tensor(running_loss / log_steps, device=device)
                avg_loss_pos = torch.tensor(running_loss_pos / log_steps, device=device)
                avg_loss_neg = torch.tensor(running_loss_neg / log_steps, device=device)
                dist.all_reduce(avg_loss, op=dist.ReduceOp.SUM)
                dist.all_reduce(avg_loss_pos, op=dist.ReduceOp.SUM)
                dist.all_reduce(avg_loss_neg, op=dist.ReduceOp.SUM)
                avg_loss = avg_loss.item() / dist.get_world_size()
                avg_loss_pos = avg_loss_pos.item() / dist.get_world_size()
                avg_loss_neg = avg_loss_neg.item() / dist.get_world_size()
                logger.info(f"(step={train_steps:07d}) Train Loss: {avg_loss:.4f} {avg_loss_pos:.4f} {avg_loss_neg:.4f}, Train Steps/Sec: {steps_per_sec:.2f}")
                # Reset monitoring variables:
                running_loss = 0
                running_loss_pos = 0
                running_loss_neg = 0
                log_steps = 0
                start_time = time()

            # Save DiT checkpoint:
            if train_steps % args.ckpt_every == 0 and train_steps > 0:
                if rank == 0:
                    checkpoint = {
                        "model": model.module.state_dict(),
                        "ema": ema.state_dict(),
                        "opt": opt.state_dict(),
                        "args": args
                    }
                    checkpoint_path = f"{checkpoint_dir}/{train_steps:07d}.pt"
                    torch.save(checkpoint, checkpoint_path)
                    logger.info(f"Saved checkpoint to {checkpoint_path}")
                dist.barrier()
    if rank == 0:
        checkpoint = {
            "model": model.module.state_dict(),
            "ema": ema.state_dict(),
            "opt": opt.state_dict(),
            "args": args
        }
        checkpoint_path = f"{checkpoint_dir}/last.pt"
        torch.save(checkpoint, checkpoint_path)
        logger.info(f"Saved checkpoint to {checkpoint_path}")  

    model.eval()  # important! This disables randomized embedding dropout
    # do any sampling/FID calculation/etc. with ema (or model) in eval mode ...
    
    logger.info("Done!")
    cleanup()


if __name__ == "__main__":
    # Default args here will train DiT-XL/2 with the hyperparameters we used in our paper (except training iters).
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-path", type=str, required=True)
    parser.add_argument("--results-dir", type=str, default="results")
    parser.add_argument("--tag", type=str, default="")
    parser.add_argument("--model", type=str, choices=list(DiT_models.keys()), default="DiT-XL/2")
    parser.add_argument("--image-size", type=int, choices=[256, 512], default=256)
    parser.add_argument("--num-classes", type=int, default=1000, help='the class number for the total dataset')
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--global-batch-size", type=int, default=256)
    parser.add_argument("--global-seed", type=int, default=0)
    parser.add_argument("--vae", type=str, choices=["ema", "mse"], default="ema")  # Choice doesn't affect training
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--log-every", type=int, default=100)
    parser.add_argument("--ckpt-every", type=int, default=500)
    parser.add_argument("--nclass", type=int, default=10, help='the class number for distillation training')
    parser.add_argument("--finetune-ipc", type=int, default=1000, help='the number of samples participating in the fine-tuning')
    parser.add_argument("--ckpt", type=str, default=None,
                        help="Optional path to a DiT checkpoint (default: auto-download a pre-trained DiT-XL/2 model).")
    parser.add_argument("--condense", action="store_true", default=False, help='whether conduct distillation')
    parser.add_argument("--spec", type=str, default='none', help='specific subset for distillation')
    parser.add_argument('--lambda-pos', default=0.002, type=float, help='weight for representativeness constraint')
    parser.add_argument('--lambda-neg', default=0.008, type=float, help='weight for diversity constraint')
    parser.add_argument("--memory-size", type=int, default=64, help='the memory size')
    parser.add_argument("--phase", type=int, default=0, help='the phase number for generating large datasets')
    args = parser.parse_args()
    main(args)
