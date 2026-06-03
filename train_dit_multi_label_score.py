"""
Fine-tuning DiT with minimax criteria.
"""
import sys
import math
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
from misc.utils import load_model


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

    # Setup data:
    transform = transforms.Compose([
        transforms.Lambda(lambda pil_image: center_crop_arr(pil_image, args.image_size)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], inplace=True)
    ])
    dataset = ImageFolder(args.data_path, transform=transform, expert_transform=expert_transform, nclass=args.nclass,
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
    logger.info(f"Dataset contains {len(dataset):,} images ({args.data_path})")

    # Prepare models for training:
    update_ema(ema, model.module, decay=0)  # Ensure EMA is initialized with synced weights
    model.train()  # important! This enables embedding dropout for classifier-free guidance
    ema.eval()  # EMA model should always be in eval mode

    # Variables for monitoring/logging purposes:
    train_steps = 0
    log_steps = 0
    running_loss, running_loss_pos, running_loss_neg, running_loss_add = 0, 0, 0, 0
    start_time = time()
    real_memory = defaultdict(list)
    pseudo_memory = defaultdict(list)
    # real_belief_memory = defaultdict(list)
    idx_to_y = torch.tensor(
        dataset.original_labels,
        device=device,
        dtype=torch.long
    )
    

    logger.info(f"Training for {args.epochs} epochs...")
    for epoch in range(args.epochs):
        sampler.set_epoch(epoch)
        logger.info(f"Beginning epoch {epoch}...")
        for x_dit, x_expert, ry, y in loader: 

            ry_t = ry.to(device).long()
            ry = ry.numpy()
            x_dit = x_dit.to(device)
            x_expert = x_expert.to(device)
            y = y.to(device)

            with torch.no_grad():
                expert_logits = expert_model(x_expert)          # [B, C]
                expert_probs = torch.softmax(expert_logits, dim=1)         # [B, C]
                expert_pred = expert_probs.argmax(dim=1)          # [B]
                expert_wrong = (expert_pred != ry_t)

                # one-hot labels (use true label y, not ry)
                onehot = torch.zeros_like(expert_probs)
                onehot.scatter_(1, ry_t.view(-1, 1), 1.0)

                temperatures = [3.0]
                grad_proxy = 0.0
                for T in temperatures:
                    probs_T = torch.softmax(expert_logits / T, dim=1)
                    grad_proxy = grad_proxy + torch.norm(
                        probs_T - onehot, dim=1
                    )
                grad_proxy = grad_proxy / len(temperatures)   # 均匀平均
                # weight = grad_proxy
                
                # ---------- top-2 predicted classes ----------
                top2_vals, top2_idx = expert_probs.topk(2, dim=1)  # [B, 2]
                top1_cls = top2_idx[:, 0]                          # [B]
                top2_cls = top2_idx[:, 1]                          # [B]

                # ---------- build secondary label ----------
                # default: no boundary class
                y_secondary = idx_to_y[top2_cls]          # [B]

                # case 1: expert prediction is WRONG → use top-1
                y_secondary[expert_wrong] = idx_to_y[top1_cls[expert_wrong]]



            with torch.no_grad():
                # Map input images to latent space + normalize latents:
                x = vae.encode(x_dit).latent_dist.sample().mul_(0.18215)
            t = torch.randint(0, diffusion.num_timesteps, (x.shape[0],), device=device)
            model_kwargs1 = dict(y=y)
            loss_dict1 = diffusion.training_losses(model, x, t, model_kwargs1)
            pseudo_embeddings = loss_dict1["output"]

            model_kwargs2 = dict(y=y_secondary)
            loss_dict2 = diffusion.training_losses(model, x, t, model_kwargs2)


            # loss = weight * loss_dict["loss"]
            # loss = loss.mean()
            weights = top2_vals / top2_vals.sum(dim=1, keepdim=True)
            weighted_loss = weights[:, 0] * loss_dict1["loss"] + weights[:, 1] * loss_dict2["loss"]
            loss = weighted_loss.mean()
            # Calculate minimax criteria
            pos_match_loss = torch.tensor(0.).to(device)
            neg_match_loss = torch.tensor(0.).to(device)
            # classifier_loss = torch.tensor(0.).to(device)
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
                    # belief_memory_update = torch.cat((belief, u), dim=1)
                    # real_belief_memory[c].extend(belief_memory_update.detach().split(1))
                    while len(real_memory[c]) > args.memory_size:
                        real_memory[c].pop(0)
                    while len(pseudo_memory[c]) > args.memory_size:
                        pseudo_memory[c].pop(0)
                    # while len(real_belief_memory[c]) > args.memory_size * 2:
                    #     real_belief_memory[c].pop(0)
                    

                all_loss = loss + pos_match_loss + neg_match_loss
            else:
                all_loss = loss


            opt.zero_grad()
            all_loss.backward()
            opt.step()
            update_ema(ema, model.module)

            # Log loss values:
            running_loss += loss.item()
            if pos_match_loss or neg_match_loss:
                running_loss_pos += pos_match_loss.item()
                running_loss_neg += neg_match_loss.item()
            # running_loss_add += classifier_loss.item()

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
                # avg_loss_alea = torch.tensor(running_loss_add / log_steps, device=device)
                dist.all_reduce(avg_loss, op=dist.ReduceOp.SUM)
                dist.all_reduce(avg_loss_pos, op=dist.ReduceOp.SUM)
                dist.all_reduce(avg_loss_neg, op=dist.ReduceOp.SUM)
                # dist.all_reduce(avg_loss_alea, op=dist.ReduceOp.SUM)
                avg_loss = avg_loss.item() / dist.get_world_size()
                avg_loss_pos = avg_loss_pos.item() / dist.get_world_size()
                avg_loss_neg = avg_loss_neg.item() / dist.get_world_size()
                # avg_loss_alea = avg_loss_alea.item() / dist.get_world_size()
                logger.info(f"(step={train_steps:07d}) Train Loss: {avg_loss:.4f} {avg_loss_pos:.4f} {avg_loss_neg:.4f} Train Steps/Sec: {steps_per_sec:.2f}")
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
