"""
Fine-tuning Stable Diffusion with multi-label gated expert guidance.
"""
import os
import random
import argparse
from typing import List

import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader
from torchvision import transforms, datasets
from PIL import Image

from accelerate import Accelerator
from accelerate.utils import set_seed as accel_set_seed
from diffusers import AutoencoderKL, UNet2DConditionModel, DDPMScheduler
from diffusers.optimization import get_scheduler
from diffusers.training_utils import EMAModel
from transformers import CLIPTokenizer, CLIPTextModel

from misc.utils import load_model


# ============================================================
# Utilities
# ============================================================
def set_seed(seed: int):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def freeze(model: nn.Module):
    for p in model.parameters():
        p.requires_grad = False


def unfreeze(model: nn.Module):
    for p in model.parameters():
        p.requires_grad = True


def _prune_checkpoints(output_dir: str, checkpoints_total_limit: int):
    if not checkpoints_total_limit or checkpoints_total_limit <= 0:
        return
    entries = []
    for name in os.listdir(output_dir):
        if not name.startswith("checkpoint-"):
            continue
        step = name.split("checkpoint-")[-1]
        if not step.isdigit():
            continue
        entries.append((int(step), name))
    entries.sort()
    while len(entries) > checkpoints_total_limit:
        _, to_remove = entries.pop(0)
        path = os.path.join(output_dir, to_remove)
        for root, dirs, files in os.walk(path, topdown=False):
            for fname in files:
                os.remove(os.path.join(root, fname))
            for dname in dirs:
                os.rmdir(os.path.join(root, dname))
        os.rmdir(path)


# ============================================================
# Prompt builder
# ============================================================
def default_prompt_template(class_name: str) -> str:
    return f"a photo of a {class_name}"


# ============================================================
# Expert gating
# ============================================================
@torch.no_grad()
def get_expert_gated_labels(
    expert_model: nn.Module,
    expert_images: torch.Tensor,
    max_secondary_weight: float = 0.5,
    temperature: float = 1.0,
    multi_threshold: float = 0.1,
):
    """
    Returns:
        top1_idx, top2_idx: [B]
        w1, w2: [B]
        use_multi: [B] bool
    """
    logits = expert_model(expert_images)
    probs = torch.softmax(logits / temperature, dim=1)

    top2_vals, top2_idx = probs.topk(k=2, dim=1)
    top1_idx = top2_idx[:, 0]
    top2_idx = top2_idx[:, 1]

    p1 = top2_vals[:, 0]
    p2 = top2_vals[:, 1]

    use_multi = p2 > multi_threshold

    denom = (p1 + p2).clamp(min=1e-8)
    w2 = p2 / denom
    w2 = torch.clamp(w2, max=max_secondary_weight)
    w1 = 1.0 - w2

    # If not boundary, fall back to single label.
    w2 = torch.where(use_multi, w2, torch.zeros_like(w2))
    w1 = torch.where(use_multi, w1, torch.ones_like(w1))

    return top1_idx, top2_idx, w1, w2, use_multi


# ============================================================
# Text encoding
# ============================================================
def encode_prompts(
    tokenizer: CLIPTokenizer,
    text_encoder: CLIPTextModel,
    prompts: List[str],
    device: torch.device,
    max_length: int = None,
):
    if max_length is None:
        max_length = tokenizer.model_max_length

    tokens = tokenizer(
        prompts,
        padding="max_length",
        truncation=True,
        max_length=max_length,
        return_tensors="pt",
    )
    input_ids = tokens.input_ids.to(device)
    attention_mask = tokens.attention_mask.to(device)

    outputs = text_encoder(
        input_ids=input_ids,
        attention_mask=attention_mask,
    )
    return outputs.last_hidden_state


# ============================================================
# One training step
# ============================================================
def compute_gated_sd_loss(
    vae: AutoencoderKL,
    unet: UNet2DConditionModel,
    noise_scheduler: DDPMScheduler,
    text_encoder: CLIPTextModel,
    tokenizer: CLIPTokenizer,
    images: torch.Tensor,
    prompt1: List[str],
    prompt2: List[str],
    w1: torch.Tensor,
    w2: torch.Tensor,
    use_multi: torch.Tensor,
    device: torch.device,
):
    # Encode images to latents.
    with torch.no_grad():
        latents = vae.encode(images).latent_dist.sample()
        latents = latents * vae.config.scaling_factor

    noise = torch.randn_like(latents)
    timesteps = torch.randint(
        0,
        noise_scheduler.config.num_train_timesteps,
        (latents.shape[0],),
        device=device,
        dtype=torch.long,
    )
    noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)

    with torch.no_grad():
        emb1 = encode_prompts(tokenizer, text_encoder, prompt1, device)
        emb2 = encode_prompts(tokenizer, text_encoder, prompt2, device)

    pred1 = unet(noisy_latents, timesteps, encoder_hidden_states=emb1).sample
    pred2 = unet(noisy_latents, timesteps, encoder_hidden_states=emb2).sample

    loss1 = F.mse_loss(pred1.float(), noise.float(), reduction="none").mean(dim=(1, 2, 3))
    loss2 = F.mse_loss(pred2.float(), noise.float(), reduction="none").mean(dim=(1, 2, 3))

    per_sample = loss1
    if use_multi.any():
        per_sample = torch.where(
            use_multi,
            w1 * loss1 + w2 * loss2,
            loss1,
        )

    total_loss = per_sample.mean()

    stats = {
        "loss": total_loss.item(),
        "loss1": loss1.mean().item(),
        "loss2": loss2.mean().item(),
        "w1": w1.mean().item(),
        "w2": w2.mean().item(),
        "use_multi": use_multi.float().mean().item(),
    }
    return total_loss, stats


# ============================================================
# Main
# ============================================================
def main(args):
    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision=args.mixed_precision,
    )
    accel_set_seed(args.seed)
    device = accelerator.device

    tokenizer = CLIPTokenizer.from_pretrained(args.sd_path, subfolder="tokenizer")
    text_encoder = CLIPTextModel.from_pretrained(args.sd_path, subfolder="text_encoder")
    vae = AutoencoderKL.from_pretrained(args.sd_path, subfolder="vae")
    unet = UNet2DConditionModel.from_pretrained(args.sd_path, subfolder="unet")
    noise_scheduler = DDPMScheduler.from_pretrained(args.sd_path, subfolder="scheduler")

    text_encoder.to(device)
    vae.to(device)
    unet.to(device)

    if args.gradient_checkpointing:
        unet.enable_gradient_checkpointing()

    freeze(vae)
    freeze(text_encoder)
    unfreeze(unet)

    expert_model = load_model(
        model_name=args.expert_model,
        dataset=args.expert_dataset,
        pretrained=True,
        classes=range(args.num_classes),
    )
    expert_model.to(device)
    expert_model.eval()
    freeze(expert_model)

    with open(args.class_names_path, "r", encoding="utf-8") as f:
        class_names = [line.strip() for line in f if line.strip()]
    if len(class_names) != args.num_classes:
        raise ValueError(
            f"class_names count {len(class_names)} != num_classes {args.num_classes}"
        )

    image_transforms = [
        transforms.Resize(args.resolution, interpolation=transforms.InterpolationMode.BILINEAR),
    ]
    if args.center_crop:
        image_transforms.append(transforms.CenterCrop(args.resolution))
    if args.random_flip:
        image_transforms.append(transforms.RandomHorizontalFlip())
    image_transforms.extend([
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
    ])
    image_transform = transforms.Compose(image_transforms)
    expert_transform = transforms.Compose([
        transforms.Resize(256, interpolation=transforms.InterpolationMode.BILINEAR),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])

    raw_dataset = datasets.ImageFolder(args.data_path)

    class DualTransformDataset(torch.utils.data.Dataset):
        def __init__(self, samples, image_transform, expert_transform):
            self.samples = samples
            self.image_transform = image_transform
            self.expert_transform = expert_transform

        def __len__(self):
            return len(self.samples)

        def __getitem__(self, idx):
            path, label = self.samples[idx]
            image = Image.open(path).convert("RGB")
            img_sd = self.image_transform(image)
            img_expert = self.expert_transform(image)
            return img_sd, img_expert, label, path

    dataset = DualTransformDataset(
        raw_dataset.samples,
        image_transform=image_transform,
        expert_transform=expert_transform,
    )

    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True,
    )

    optimizer = torch.optim.AdamW(
        unet.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )
    num_update_steps_per_epoch = max(1, len(loader) // args.gradient_accumulation_steps)
    max_train_steps = args.epochs * num_update_steps_per_epoch
    lr_scheduler = get_scheduler(
        name=args.lr_scheduler,
        optimizer=optimizer,
        num_warmup_steps=args.lr_warmup_steps,
        num_training_steps=max_train_steps,
    )
    ema_unet = None
    if args.use_ema:
        ema_unet = EMAModel(
            unet.parameters(),
            model_cls=UNet2DConditionModel,
            model_config=unet.config,
        )

    os.makedirs(args.output_dir, exist_ok=True)
    unet, optimizer, loader, lr_scheduler = accelerator.prepare(
        unet, optimizer, loader, lr_scheduler
    )
    unet.train()

    global_step = 0
    optimizer.zero_grad(set_to_none=True)
    for epoch in range(args.epochs):
        for step, (images, expert_images, _, _) in enumerate(loader):
            images = images.to(device, non_blocking=True)
            expert_images = expert_images.to(device, non_blocking=True)

            with torch.no_grad():
                top1_idx, top2_idx, w1, w2, use_multi = get_expert_gated_labels(
                    expert_model,
                    expert_images,
                    max_secondary_weight=args.max_secondary_weight,
                    temperature=args.expert_temperature,
                    multi_threshold=args.multi_threshold,
                )

            prompt1 = [default_prompt_template(class_names[i]) for i in top1_idx.tolist()]
            prompt2 = [default_prompt_template(class_names[i]) for i in top2_idx.tolist()]

            w1 = w1.to(device)
            w2 = w2.to(device)
            use_multi = use_multi.to(device)

            with accelerator.accumulate(unet):
                with accelerator.autocast():
                    loss, stats = compute_gated_sd_loss(
                        vae=vae,
                        unet=unet,
                        noise_scheduler=noise_scheduler,
                        text_encoder=text_encoder,
                        tokenizer=tokenizer,
                        images=images,
                        prompt1=prompt1,
                        prompt2=prompt2,
                        w1=w1,
                        w2=w2,
                        use_multi=use_multi,
                        device=device,
                    )

                accelerator.backward(loss)
                if accelerator.sync_gradients:
                    if args.max_grad_norm is not None and args.max_grad_norm > 0:
                        accelerator.clip_grad_norm_(unet.parameters(), args.max_grad_norm)
                    optimizer.step()
                    optimizer.zero_grad(set_to_none=True)
                    lr_scheduler.step()
                    if ema_unet is not None:
                        ema_unet.step(accelerator.unwrap_model(unet).parameters())

                    global_step += 1

                    if accelerator.is_main_process and global_step % args.log_every == 0:
                        print(
                            f"[Epoch {epoch+1}/{args.epochs}] "
                            f"step={global_step} "
                            f"loss={stats['loss']:.4f} "
                            f"loss1={stats['loss1']:.4f} "
                            f"loss2={stats['loss2']:.4f} "
                            f"w1={stats['w1']:.4f} "
                            f"w2={stats['w2']:.4f} "
                            f"use_multi={stats['use_multi']:.3f}"
                        )

                    checkpoint_every = args.checkpointing_steps or args.save_every
                    if (
                        accelerator.is_main_process
                        and checkpoint_every > 0
                        and global_step % checkpoint_every == 0
                    ):
                        save_dir = os.path.join(args.output_dir, f"checkpoint-{global_step}")
                        os.makedirs(save_dir, exist_ok=True)
                        unet_to_save = accelerator.unwrap_model(unet)
                        unet_to_save.save_pretrained(os.path.join(save_dir, "unet"))
                        if ema_unet is not None:
                            ema_unet.store(unet_to_save.parameters())
                            ema_unet.copy_to(unet_to_save.parameters())
                            unet_to_save.save_pretrained(os.path.join(save_dir, "unet_ema"))
                            ema_unet.restore(unet_to_save.parameters())
                        _prune_checkpoints(args.output_dir, args.checkpoints_total_limit)
                        print(f"Saved checkpoint to {save_dir}")

    final_dir = os.path.join(args.output_dir, "final")
    if accelerator.is_main_process:
        os.makedirs(final_dir, exist_ok=True)
        unet_to_save = accelerator.unwrap_model(unet)
        unet_to_save.save_pretrained(os.path.join(final_dir, "unet"))
        if ema_unet is not None:
            ema_unet.store(unet_to_save.parameters())
            ema_unet.copy_to(unet_to_save.parameters())
            unet_to_save.save_pretrained(os.path.join(final_dir, "unet_ema"))
            ema_unet.restore(unet_to_save.parameters())
        print(f"Training done. Final model saved to {final_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--data_path", type=str, required=True)
    parser.add_argument("--sd_path", type=str, required=True)
    parser.add_argument("--class_names_path", type=str, required=True)
    parser.add_argument("--output_dir", type=str, default="./sd_expert_finetune")

    parser.add_argument("--expert_model", type=str, default="resnet18")
    parser.add_argument("--expert_dataset", type=str, default="imagenette")
    parser.add_argument("--num_classes", type=int, required=True)
    parser.add_argument("--expert_temperature", type=float, default=1.0)
    parser.add_argument("--max_secondary_weight", type=float, default=0.5)
    parser.add_argument("--multi_threshold", type=float, default=0.1)

    parser.add_argument("--resolution", type=int, default=512)
    parser.add_argument("--center_crop", action="store_true", default=True)
    parser.add_argument("--no_center_crop", action="store_false", dest="center_crop")
    parser.add_argument("--random_flip", action="store_true", default=True)
    parser.add_argument("--no_random_flip", action="store_false", dest="random_flip")
    parser.add_argument("--train_batch_size", "--batch_size", type=int, default=4, dest="batch_size")
    parser.add_argument("--num_train_epochs", "--epochs", type=int, default=10, dest="epochs")
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--weight_decay", type=float, default=1e-2)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument(
        "--mixed_precision",
        type=str,
        default="no",
        choices=["no", "fp16", "bf16"],
    )

    parser.add_argument("--log_every", type=int, default=50)
    parser.add_argument("--save_every", type=int, default=1000)
    parser.add_argument("--checkpointing_steps", type=int, default=0)
    parser.add_argument("--checkpoints_total_limit", type=int, default=0)
    parser.add_argument("--validation_epochs", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1)
    parser.add_argument("--gradient_checkpointing", action="store_true", default=False)
    parser.add_argument("--max_grad_norm", type=float, default=1.0)
    parser.add_argument("--lr_scheduler", type=str, default="constant")
    parser.add_argument("--lr_warmup_steps", type=int, default=0)
    parser.add_argument("--use_ema", action="store_true", default=False)

    args = parser.parse_args()
    main(args)
