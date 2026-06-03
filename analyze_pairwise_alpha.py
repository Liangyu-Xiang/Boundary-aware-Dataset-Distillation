"""
Analyze pair-wise target/prediction scale for choosing pairwise_alpha.
"""
import sys
import torch
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torchvision import transforms
import numpy as np
from PIL import Image
from time import time
import argparse
import logging

from data import ImageFolder
from models import DiT_models
from download import find_model
from diffusion import create_diffusion
from diffusers.models import AutoencoderKL
from misc.utils import load_model


def cleanup():
    dist.destroy_process_group()


def create_logger():
    if dist.get_rank() == 0:
        logger = logging.getLogger(__name__)
        logger.propagate = False
        logger.setLevel(logging.INFO)
        if not logger.handlers:
            ch = logging.StreamHandler(stream=sys.stdout)
            ch.setLevel(logging.INFO)
            formatter = logging.Formatter('[%(asctime)s] %(message)s')
            ch.setFormatter(formatter)
            logger.addHandler(ch)
    else:
        logger = logging.getLogger(__name__)
        logger.addHandler(logging.NullHandler())
    return logger


def center_crop_arr(pil_image, image_size):
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


@torch.no_grad()
def main(args):
    assert torch.cuda.is_available(), "This script requires at least one GPU."

    dist.init_process_group("nccl")
    assert args.global_batch_size % dist.get_world_size() == 0, "Batch size must be divisible by world size."
    rank = dist.get_rank()
    device = rank % torch.cuda.device_count()
    seed = args.global_seed * dist.get_world_size() + rank
    torch.manual_seed(seed)
    torch.cuda.set_device(device)

    logger = create_logger()
    logger.info(f"Starting rank={rank}, seed={seed}, world_size={dist.get_world_size()}.")

    latent_size = args.image_size // 8
    model = DiT_models[args.model](input_size=latent_size, num_classes=args.num_classes)
    state_dict = find_model(args.ckpt)
    model.load_state_dict(state_dict, strict=False)
    model = DDP(model.to(device), device_ids=[rank])
    model.eval()

    diffusion = create_diffusion(timestep_respacing="")
    vae_path = f"./pretrained_models/stabilityai/sd-vae-ft-{args.vae}"
    vae = AutoencoderKL.from_pretrained(vae_path).to(device)
    vae.eval()

    expert_model = load_model(
        model_name='resnet18',
        dataset=args.spec,
        pretrained=True,
        classes=range(args.nclass)
    )
    expert_normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
    )
    expert_transform = transforms.Compose(
        [
            transforms.Resize(224 // 7 * 8, antialias=True),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            expert_normalize,
        ]
    )
    expert_model.eval()
    expert_model.to(device)
    for p in expert_model.parameters():
        p.requires_grad = False

    transform = transforms.Compose([
        transforms.Lambda(lambda pil_image: center_crop_arr(pil_image, args.image_size)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], inplace=True)
    ])
    dataset = ImageFolder(
        args.data_path,
        transform=transform,
        expert_transform=expert_transform,
        nclass=args.nclass,
        ipc=args.finetune_ipc,
        spec=args.spec,
        phase=args.phase,
        seed=0,
        return_origin=True,
    )
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
    idx_to_y = torch.tensor(dataset.original_labels, device=device, dtype=torch.long)
    logger.info(f"Dataset contains {len(dataset):,} images ({args.data_path})")

    total_pred_mean_sum = torch.zeros(1, device=device)
    total_pred_abs_sum = torch.zeros(1, device=device)
    total_pred_min = torch.full((1,), float("inf"), device=device)
    total_pred_max = torch.full((1,), float("-inf"), device=device)
    total_target_mean_sum = torch.zeros(1, device=device)
    total_target_abs_sum = torch.zeros(1, device=device)
    total_target_min = torch.full((1,), float("inf"), device=device)
    total_target_max = torch.full((1,), float("-inf"), device=device)
    total_gap_abs_sum = torch.zeros(1, device=device)
    total_gap_sq_sum = torch.zeros(1, device=device)
    total_valid_pairs = torch.zeros(1, device=device)
    total_weight = torch.zeros(1, device=device)
    weighted_base_target_sum = torch.zeros(1, device=device)
    weighted_base_sq_sum = torch.zeros(1, device=device)

    start_time = time()
    max_steps = args.max_steps if args.max_steps > 0 else None
    logger.info("Running pair-wise scale analysis.")

    for step, (x_dit, x_expert, _ry, _y) in enumerate(loader, start=1):
        x_dit = x_dit.to(device)
        x_expert = x_expert.to(device)

        expert_logits = expert_model(x_expert)
        expert_probs = torch.softmax(expert_logits, dim=1)
        topk_vals, topk_idx = expert_probs.topk(args.ratio_topk, dim=1)
        y_topk = idx_to_y[topk_idx]
        q_topk = topk_vals / (topk_vals.sum(dim=1, keepdim=True) + 1e-8)

        x = vae.encode(x_dit).latent_dist.sample().mul_(0.18215)
        t = torch.randint(0, diffusion.num_timesteps, (x.shape[0],), device=device)

        energy_list = []
        for k in range(y_topk.shape[1]):
            loss_dict_k = diffusion.training_losses(model, x, t, dict(y=y_topk[:, k]))
            energy_list.append(loss_dict_k["loss"])
        energies = torch.stack(energy_list, dim=1)

        log_q_topk = torch.log(q_topk + 1e-8)
        base_pairwise = -(energies.unsqueeze(1) - energies.unsqueeze(2))
        target_pairwise = log_q_topk.unsqueeze(1) - log_q_topk.unsqueeze(2)

        pairwise_mask = 1.0 - torch.eye(y_topk.shape[1], device=device, dtype=energies.dtype)
        valid_pair_mask = pairwise_mask.unsqueeze(0).bool().expand(x.shape[0], -1, -1)
        pairwise_weight = q_topk.unsqueeze(1) * q_topk.unsqueeze(2) * pairwise_mask.unsqueeze(0)

        base_valid = base_pairwise[valid_pair_mask]
        target_valid = target_pairwise[valid_pair_mask]
        gap_valid = base_valid - target_valid
        weight_valid = pairwise_weight[valid_pair_mask]

        total_pred_mean_sum += base_valid.sum()
        total_pred_abs_sum += base_valid.abs().sum()
        total_pred_min = torch.minimum(total_pred_min, base_valid.min().unsqueeze(0))
        total_pred_max = torch.maximum(total_pred_max, base_valid.max().unsqueeze(0))
        total_target_mean_sum += target_valid.sum()
        total_target_abs_sum += target_valid.abs().sum()
        total_target_min = torch.minimum(total_target_min, target_valid.min().unsqueeze(0))
        total_target_max = torch.maximum(total_target_max, target_valid.max().unsqueeze(0))
        total_gap_abs_sum += gap_valid.abs().sum()
        total_gap_sq_sum += gap_valid.pow(2).sum()
        total_valid_pairs += torch.tensor([base_valid.numel()], device=device, dtype=base_valid.dtype)

        total_weight += weight_valid.sum()
        weighted_base_target_sum += (weight_valid * base_valid * target_valid).sum()
        weighted_base_sq_sum += (weight_valid * base_valid.pow(2)).sum()

        if step % args.log_every == 0 and rank == 0:
            elapsed = time() - start_time
            logger.info(f"Processed {step} steps in {elapsed:.2f}s")

        if max_steps is not None and step >= max_steps:
            break

    stats = [
        total_pred_mean_sum, total_pred_abs_sum, total_pred_min, total_pred_max,
        total_target_mean_sum, total_target_abs_sum, total_target_min, total_target_max,
        total_gap_abs_sum, total_gap_sq_sum, total_valid_pairs,
        total_weight, weighted_base_target_sum, weighted_base_sq_sum,
    ]
    for stat in stats:
        if stat is total_pred_min:
            dist.all_reduce(stat, op=dist.ReduceOp.MIN)
        elif stat is total_pred_max:
            dist.all_reduce(stat, op=dist.ReduceOp.MAX)
        elif stat is total_target_min:
            dist.all_reduce(stat, op=dist.ReduceOp.MIN)
        elif stat is total_target_max:
            dist.all_reduce(stat, op=dist.ReduceOp.MAX)
        else:
            dist.all_reduce(stat, op=dist.ReduceOp.SUM)

    pred_mean = (total_pred_mean_sum / total_valid_pairs.clamp_min(1.0)).item()
    pred_abs_mean = (total_pred_abs_sum / total_valid_pairs.clamp_min(1.0)).item()
    target_mean = (total_target_mean_sum / total_valid_pairs.clamp_min(1.0)).item()
    target_abs_mean = (total_target_abs_sum / total_valid_pairs.clamp_min(1.0)).item()
    gap_abs_mean = (total_gap_abs_sum / total_valid_pairs.clamp_min(1.0)).item()
    gap_rmse = torch.sqrt(total_gap_sq_sum / total_valid_pairs.clamp_min(1.0)).item()
    alpha_abs = (total_target_abs_sum / total_pred_abs_sum.clamp_min(1e-8)).item()
    alpha_l2 = (weighted_base_target_sum / weighted_base_sq_sum.clamp_min(1e-8)).item()

    if rank == 0:
        logger.info("Pair-wise scale analysis finished.")
        logger.info(
            f"base pred mean/abs/min/max: {pred_mean:.4f}/{pred_abs_mean:.4f}/"
            f"{total_pred_min.item():.4f}/{total_pred_max.item():.4f}"
        )
        logger.info(
            f"target mean/abs/min/max: {target_mean:.4f}/{target_abs_mean:.4f}/"
            f"{total_target_min.item():.4f}/{total_target_max.item():.4f}"
        )
        logger.info(f"base-target gap abs/rmse: {gap_abs_mean:.4f}/{gap_rmse:.4f}")
        logger.info(f"Suggested alpha from abs-scale ratio: {alpha_abs:.4f}")
        logger.info(f"Suggested alpha from weighted least squares: {alpha_l2:.4f}")

    cleanup()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-path", type=str, required=True)
    parser.add_argument("--model", type=str, choices=list(DiT_models.keys()), default="DiT-XL/2")
    parser.add_argument("--image-size", type=int, choices=[256, 512], default=256)
    parser.add_argument("--num-classes", type=int, default=1000)
    parser.add_argument("--global-batch-size", type=int, default=256)
    parser.add_argument("--global-seed", type=int, default=0)
    parser.add_argument("--vae", type=str, choices=["ema", "mse"], default="ema")
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--log-every", type=int, default=100)
    parser.add_argument("--nclass", type=int, default=10)
    parser.add_argument("--finetune-ipc", type=int, default=1000)
    parser.add_argument("--ckpt", type=str, default=None)
    parser.add_argument("--spec", type=str, default='none')
    parser.add_argument("--phase", type=int, default=0)
    parser.add_argument("--ratio-topk", type=int, default=2)
    parser.add_argument("--max-steps", type=int, default=-1, help="Stop early after this many steps; -1 means full pass.")
    args = parser.parse_args()
    main(args)
