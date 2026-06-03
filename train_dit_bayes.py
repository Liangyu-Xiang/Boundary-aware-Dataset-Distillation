"""
Fine-tune class-conditional DiT with Bayes posterior alignment.
"""
import argparse
import os
from copy import deepcopy
from glob import glob
from time import time

import torch
import torch.distributed as dist
from diffusers.models import AutoencoderKL
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torchvision import transforms

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

COLOR_CYAN = "\033[96m"
COLOR_YELLOW = "\033[93m"
COLOR_GREEN = "\033[92m"
COLOR_MAGENTA = "\033[95m"
COLOR_RESET = "\033[0m"

from data import ImageFolder
from diffusion import create_diffusion
from download import find_model
from misc.utils import load_model
from models import DiT_models
from train_dit import center_crop_arr, create_logger, mark_difffit_trainable, requires_grad, update_ema

def training_losses_without_label_dropout(model, diffusion, x_start, t, y, noise):
    """
    Bayes alignment compares explicit labels, so classifier-free label dropout is
    disabled only for this auxiliary forward.
    """
    module = model.module if isinstance(model, DDP) else model
    old_dropout_prob = module.y_embedder.dropout_prob
    module.y_embedder.dropout_prob = 0.0
    try:
        return diffusion.training_losses(model, x_start, t, dict(y=single_label_condition(y)), noise=noise)
    finally:
        module.y_embedder.dropout_prob = old_dropout_prob


def single_label_condition(y):
    """
    Avoid DiT's multi-label branch, which keys off y.shape[0] == 3 or 4.
    A [1, B] tensor is flattened by the normal single-label branch in models.py.
    """
    return y.view(1, -1)


def compute_bayes_alignment_loss(model, diffusion, x_start, t, noise, expert_probs, idx_to_y, args):
    """
    Align the posterior induced by top-K conditional denoising losses with the
    expert classifier posterior.
    """
    if expert_probs.shape[1] != idx_to_y.numel():
        raise ValueError(
            f"expert_probs has {expert_probs.shape[1]} classes, but dataset label map has {idx_to_y.numel()} classes."
        )
    topk = min(args.topk, expert_probs.shape[1])
    expert_probs = expert_probs.detach()
    expert_probs_topk, topk_idx = torch.topk(expert_probs, k=topk, dim=1)
    q_topk = expert_probs_topk / expert_probs_topk.sum(dim=1, keepdim=True).clamp_min(1e-8)
    q_topk = q_topk.detach()

    topk_labels = idx_to_y[topk_idx]
    batch_size = x_start.shape[0]
    x_start_rep = x_start[:, None].expand(batch_size, topk, *x_start.shape[1:]).reshape(
        batch_size * topk, *x_start.shape[1:]
    )
    noise_rep = noise[:, None].expand(batch_size, topk, *noise.shape[1:]).reshape(
        batch_size * topk, *noise.shape[1:]
    )
    t_rep = t[:, None].expand(batch_size, topk).reshape(batch_size * topk)
    y_rep = topk_labels.reshape(batch_size * topk)
    loss_dict_rep = training_losses_without_label_dropout(
        model=model,
        diffusion=diffusion,
        x_start=x_start_rep,
        t=t_rep,
        y=y_rep,
        noise=noise_rep,
    )
    cond_losses = loss_dict_rep["loss"].view(batch_size, topk)

    logits_gen = -args.bayes_alpha * cond_losses
    log_q_hat = torch.log_softmax(logits_gen, dim=1)
    q_hat = log_q_hat.exp()
    q_topk = q_topk.clamp_min(1e-8)
    log_q = torch.log(q_topk)
    loss_bayes = (q_topk * (log_q - log_q_hat)).sum(dim=1).mean()
    debug_info = {
        "topk_labels": topk_labels.detach(),
        "expert_probs_topk": expert_probs_topk.detach(),
        "expert_probs_topk_norm": q_topk.detach(),
        "diffusion_probs_topk": q_hat.detach(),
    }
    return loss_bayes, cond_losses, debug_info


def log_debug_distributions(logger, epoch, train_steps, y, debug_info, cond_losses):
    """
    Temporary debug print for comparing expert posterior and diffusion-induced posterior.
    """
    y_cpu = y.detach().cpu().tolist()
    topk_labels = debug_info["topk_labels"].cpu().tolist()
    expert_probs_topk = debug_info["expert_probs_topk"].cpu().tolist()
    expert_probs_topk_norm = debug_info["expert_probs_topk_norm"].cpu().tolist()
    diffusion_probs_topk = debug_info["diffusion_probs_topk"].cpu().tolist()
    cond_losses = cond_losses.detach().cpu().tolist()

    logger.info(f"[debug_probs] epoch={epoch} step={train_steps}")
    for sample_idx in range(len(y_cpu)):
        logger.info(
            "[debug_probs] "
            f"sample={sample_idx} "
            f"label={y_cpu[sample_idx]} "
            f"topk_labels={topk_labels[sample_idx]} "
            f"expert_probs_topk={expert_probs_topk[sample_idx]} "
            f"expert_probs_topk_norm={expert_probs_topk_norm[sample_idx]} "
            f"diffusion_probs_topk={diffusion_probs_topk[sample_idx]} "
            f"cond_losses={cond_losses[sample_idx]}"
        )


def main(args):
    """
    Fine-tune a DiT model.
    """
    assert torch.cuda.is_available(), "Training currently requires at least one GPU."
    assert args.topk > 0, "topk must be positive."

    dist.init_process_group("nccl")
    assert args.global_batch_size % dist.get_world_size() == 0, "Batch size must be divisible by world size."
    rank = dist.get_rank()
    device = rank % torch.cuda.device_count()
    seed = args.global_seed * dist.get_world_size() + rank
    torch.manual_seed(seed)
    torch.cuda.set_device(device)
    print(f"Starting rank={rank}, seed={seed}, world_size={dist.get_world_size()}.")

    if rank == 0:
        os.makedirs(args.results_dir, exist_ok=True)
        experiment_index = len(glob(f"{args.results_dir}/*"))
        model_string_name = args.model.replace("/", "-")
        experiment_dir = f"{args.results_dir}/{experiment_index:03d}-{model_string_name}"
        if args.tag:
            experiment_dir += f"-{args.tag}"
        checkpoint_dir = f"{experiment_dir}/checkpoints"
        os.makedirs(checkpoint_dir, exist_ok=True)
        logger = create_logger(experiment_dir)
        logger.info(f"Experiment directory created at {experiment_dir}")
    else:
        checkpoint_dir = None
        logger = create_logger(None)

    assert args.image_size % 8 == 0, "Image size must be divisible by 8 (for the VAE encoder)."
    latent_size = args.image_size // 8
    model = DiT_models[args.model](
        input_size=latent_size,
        num_classes=args.num_classes,
    )
    state_dict = find_model(args.ckpt)
    model.load_state_dict(state_dict, strict=False)
    ema = deepcopy(model).to(device)
    requires_grad(ema, False)
    diffusion = create_diffusion(timestep_respacing="")
    vae_path = f"./pretrained_models/stabilityai/sd-vae-ft-{args.vae}"
    vae = AutoencoderKL.from_pretrained(vae_path).to(device)
    vae.eval()
    logger.info(f"DiT Parameters: {sum(p.numel() for p in model.parameters()):,}")

    model = mark_difffit_trainable(model)
    model = DDP(model.to(device), device_ids=[rank])
    params_to_optimize = [p for p in model.parameters() if p.requires_grad]
    total_params = sum(p.numel() for p in params_to_optimize)
    print(f"Number of Trainable Parameters: {total_params * 1.e-6:.2f} M")
    opt = torch.optim.AdamW(params_to_optimize, lr=args.lr, weight_decay=0)

    expert_model = load_model(
        model_name=args.expert_model,
        dataset=args.spec,
        pretrained=True,
        classes=range(args.nclass),
    )
    expert_normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
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

    transform = transforms.Compose(
        [
            transforms.Lambda(lambda pil_image: center_crop_arr(pil_image, args.image_size)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], inplace=True),
        ]
    )
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
        seed=args.global_seed,
    )
    loader = DataLoader(
        dataset,
        batch_size=int(args.global_batch_size // dist.get_world_size()),
        shuffle=False,
        sampler=sampler,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True,
    )
    idx_to_y = torch.tensor(dataset.original_labels, device=device, dtype=torch.long)
    logger.info(f"Dataset contains {len(dataset):,} images ({args.data_path})")

    update_ema(ema, model.module, decay=0)
    model.train()
    ema.eval()

    train_steps = 0
    log_steps = 0
    running_loss_denoise = 0.0
    running_loss_bayes = 0.0
    running_loss_total = 0.0
    running_cond_loss = 0.0
    start_time = time()

    logger.info(f"Training for {args.epochs} epochs...")
    for epoch in range(args.epochs):
        epoch_start_time = time()
        sampler.set_epoch(epoch)
        logger.info(f"Beginning epoch {epoch}...")
        for batch_idx, (x_dit, x_expert, _, y) in enumerate(loader):
            x_dit = x_dit.to(device)
            x_expert = x_expert.to(device)
            y = y.to(device)

            with torch.no_grad():
                expert_logits = expert_model(x_expert)
                expert_probs = torch.softmax(expert_logits, dim=1).detach()

                x_start = vae.encode(x_dit).latent_dist.sample().mul_(0.18215)
                noise = torch.randn_like(x_start)
                t = torch.randint(0, diffusion.num_timesteps, (x_start.shape[0],), device=device)

            loss_dict = diffusion.training_losses(model, x_start, t, dict(y=single_label_condition(y)), noise=noise)
            loss_denoise = loss_dict["loss"].mean()

            loss_bayes, cond_losses, debug_info = compute_bayes_alignment_loss(
                model=model,
                diffusion=diffusion,
                x_start=x_start,
                t=t,
                noise=noise,
                expert_probs=expert_probs,
                idx_to_y=idx_to_y,
                args=args,
            )
            loss_total = loss_denoise + args.lambda_bayes * loss_bayes

            if args.debug_print_probs and rank == 0 and batch_idx == 0:
                log_debug_distributions(logger, epoch, train_steps, y, debug_info, cond_losses)

            opt.zero_grad()
            loss_total.backward()
            opt.step()
            update_ema(ema, model.module)

            running_loss_denoise += loss_denoise.item()
            running_loss_bayes += loss_bayes.item()
            running_loss_total += loss_total.item()
            running_cond_loss += cond_losses.mean().item()
            log_steps += 1
            train_steps += 1

            if train_steps % args.log_every == 0:
                torch.cuda.synchronize()
                end_time = time()
                steps_per_sec = log_steps / (end_time - start_time)

                avg_denoise = torch.tensor(running_loss_denoise / log_steps, device=device)
                avg_bayes = torch.tensor(running_loss_bayes / log_steps, device=device)
                avg_total = torch.tensor(running_loss_total / log_steps, device=device)
                avg_cond = torch.tensor(running_cond_loss / log_steps, device=device)
                for value in [avg_denoise, avg_bayes, avg_total, avg_cond]:
                    dist.all_reduce(value, op=dist.ReduceOp.SUM)
                avg_denoise = avg_denoise.item() / dist.get_world_size()
                avg_bayes = avg_bayes.item() / dist.get_world_size()
                avg_total = avg_total.item() / dist.get_world_size()
                avg_cond = avg_cond.item() / dist.get_world_size()

                logger.info(
                    f"{COLOR_MAGENTA}(step={train_steps:07d}){COLOR_RESET} "
                    f"{COLOR_CYAN}loss_denoise={avg_denoise:.4f}{COLOR_RESET} "
                    f"{COLOR_YELLOW}loss_bayes={avg_bayes:.4f}{COLOR_RESET} "
                    f"{COLOR_GREEN}loss_total={avg_total:.4f}{COLOR_RESET} "
                    f"{COLOR_CYAN}cond_loss={avg_cond:.4f}{COLOR_RESET} "
                    f"Train Steps/Sec={steps_per_sec:.2f}"
                )
                running_loss_denoise = 0.0
                running_loss_bayes = 0.0
                running_loss_total = 0.0
                running_cond_loss = 0.0
                log_steps = 0
                start_time = time()

        epoch_time = time() - epoch_start_time
        logger.info(f"Epoch {epoch} finished in {epoch_time:.2f} seconds ({epoch_time / 60:.2f} minutes)")
        if rank == 0:
            checkpoint = {
                "model": model.module.state_dict(),
                "ema": ema.state_dict(),
                "opt": opt.state_dict(),
                "args": args,
                "epoch": epoch,
                "train_steps": train_steps,
            }
            checkpoint_path = f"{checkpoint_dir}/epoch_{epoch:03d}.pt"
            torch.save(checkpoint, checkpoint_path)
            logger.info(f"[Epoch {epoch}] Saved checkpoint to {checkpoint_path}")
        dist.barrier()

    model.eval()
    logger.info("Done!")
    dist.destroy_process_group()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-path", type=str, required=True)
    parser.add_argument("--results-dir", type=str, default="results")
    parser.add_argument("--tag", type=str, default="")
    parser.add_argument("--model", type=str, choices=list(DiT_models.keys()), default="DiT-XL/2")
    parser.add_argument("--image-size", type=int, choices=[256, 512], default=256)
    parser.add_argument("--num-classes", type=int, default=1000, help="the class number for the total dataset")
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--global-batch-size", type=int, default=256)
    parser.add_argument("--global-seed", type=int, default=0)
    parser.add_argument("--vae", type=str, choices=["ema", "mse"], default="ema")
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--log-every", type=int, default=100)
    parser.add_argument("--nclass", type=int, default=10, help="the class number for distillation training")
    parser.add_argument("--finetune-ipc", type=int, default=1000)
    parser.add_argument(
        "--ckpt",
        type=str,
        default=None,
        help="Optional path to a DiT checkpoint (default: auto-download a pre-trained DiT-XL/2 model).",
    )
    parser.add_argument("--spec", type=str, default="none", help="specific subset for distillation")
    parser.add_argument("--phase", type=int, default=0, help="the phase number for generating large datasets")
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--expert-model", type=str, default="resnet18")
    parser.add_argument("--topk", type=int, default=2)
    parser.add_argument("--bayes-alpha", type=float, default=1.0)
    parser.add_argument("--lambda-bayes", type=float, default=0.1)
    parser.add_argument("--debug-print-probs", action="store_true", default=False)
    args = parser.parse_args()
    main(args)
