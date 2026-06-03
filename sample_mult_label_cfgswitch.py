"""
Sample images with a configurable ratio of multi-condition samples.
"""
import os
import argparse

import torch
from tqdm import tqdm

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

from torchvision.utils import save_image
from diffusion import create_diffusion
from diffusers.models import AutoencoderKL
from download import find_model
from models import DiT_models


SINGLE_COND_CFG = 1.0
MULTI_COND_CFG = 1.0


def get_stage_scheduled_weight(base_weight, noise_level, args):
    if noise_level >= args.high_noise_until:
        return 1.0 - args.high_boundary_factor * (1.0 - base_weight)
    if noise_level >= args.low_noise_from:
        return base_weight

    low_stage_progress = (args.low_noise_from - noise_level) / max(args.low_noise_from, 1e-8)
    low_stage_target = 1.0 - args.low_boundary_factor * (1.0 - base_weight)
    return base_weight + low_stage_progress * (low_stage_target - base_weight)


def get_stage_cfg_scale(base_cfg_scale, noise_level, args):
    if noise_level >= args.high_noise_until:
        return base_cfg_scale * args.high_cfg_scale_mult
    if noise_level >= args.low_noise_from:
        return base_cfg_scale * args.mid_cfg_scale_mult
    return base_cfg_scale * args.low_cfg_scale_mult


class MultiCondStageCFG:
    def __init__(self, model, num_timesteps, args):
        self.model = model
        self.num_timesteps = num_timesteps
        self.args = args

    def __call__(self, x, t, y, cfg_scale):
        scheduled_cfg_scale = cfg_scale
        scheduled_y = y
        if self.args.enable_stage_cfg and y.shape[0] == 4:
            noise_level = t[0].item() / max(self.num_timesteps - 1, 1)
            scheduled_cfg_scale = get_stage_cfg_scale(cfg_scale, noise_level, self.args)
            scheduled_y = y.clone()
            base_weight = scheduled_y[-1]
            scheduled_y[-1] = get_stage_scheduled_weight(base_weight, noise_level, self.args)
        return self.model.forward_with_cfg(x, t, scheduled_y, scheduled_cfg_scale)


def load_selected_classes(spec, nclass, phase):
    with open("./misc/class_indices.txt", "r") as fp:
        all_classes = [line.strip() for line in fp]

    if spec == "woof":
        file_list = "./misc/class_woof.txt"
    elif spec == "nette":
        file_list = "./misc/class_nette.txt"
    elif spec == "100":
        file_list = "./misc/class100.txt"
    elif spec == "1k":
        file_list = "./misc/class_indices.txt"
    else:
        raise ValueError(f"Unsupported dataset spec '{spec}'.")

    with open(file_list, "r") as fp:
        sel_classes = [line.strip() for line in fp]

    phase = max(0, phase)
    cls_from = nclass * phase
    cls_to = nclass * (phase + 1)
    sel_classes = sel_classes[cls_from:cls_to]
    class_labels = [all_classes.index(sel_class) for sel_class in sel_classes]
    return sel_classes, class_labels


def build_conditioning(class_label, spec_indices, batch_size, use_multi_condition, label_weight, device):
    primary_label = torch.full((batch_size,), class_label, device=device, dtype=torch.long)

    if not use_multi_condition:
        y_null = torch.full((batch_size,), 1000, device=device, dtype=torch.long)
        y = torch.cat([primary_label, y_null], dim=0)
        return y, SINGLE_COND_CFG, [[class_label]]

    mask = spec_indices != class_label
    candidates = spec_indices[mask]
    if len(candidates) == 0:
        raise ValueError("No secondary labels are available for multi-condition sampling.")

    random_indices = torch.randperm(len(candidates), device=device)[:1]
    random_label = candidates[random_indices].repeat(batch_size)
    conditioned_labels = torch.stack([primary_label, random_label], dim=0)
    y_null = torch.full((batch_size,), 1000, device=device, dtype=torch.long)
    mix_weight = torch.full((batch_size,), label_weight, device=device)
    y = torch.cat([conditioned_labels, y_null.unsqueeze(0)], dim=0)
    y = torch.cat([y, mix_weight.unsqueeze(0)], dim=0)
    label_indices = conditioned_labels.detach().cpu().tolist()
    return y, MULTI_COND_CFG, label_indices


def should_use_multi_condition(shift, num_multi_condition):
    return shift < num_multi_condition


def main(args):
    torch.manual_seed(args.seed)
    torch.set_grad_enabled(False)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    sel_classes, class_labels = load_selected_classes(args.spec, args.nclass, args.phase)

    if args.ckpt is None:
        assert args.model == "DiT-XL/2", "Only DiT-XL/2 models are available for auto-download."
        assert args.image_size in [256, 512]
        assert args.num_classes == 1000

    latent_size = args.image_size // 8
    model = DiT_models[args.model](
        input_size=latent_size,
        num_classes=args.num_classes,
    ).to(device)

    ckpt_path = args.ckpt or f"DiT-XL-2-{args.image_size}x{args.image_size}.pt"
    state_dict = find_model(ckpt_path)
    model.load_state_dict(state_dict, strict=False)
    model.eval()

    diffusion = create_diffusion(str(args.num_sampling_steps))
    scheduled_model = MultiCondStageCFG(model, diffusion.num_timesteps, args)
    vae_path = f"./pretrained_models/stabilityai/sd-vae-ft-{args.vae}"
    vae = AutoencoderKL.from_pretrained(vae_path).to(device)

    batch_size = 1
    spec_indices = torch.tensor(class_labels, device=device, dtype=torch.long)

    for class_label, sel_class in zip(class_labels, sel_classes):
        print(class_label)
        os.makedirs(os.path.join(args.save_dir, sel_class), exist_ok=True)
        num_multi_condition = int(args.num_samples * args.multi_cond_ratio)

        for shift in tqdm(range(args.num_samples // batch_size)):
            z = torch.randn(batch_size, 4, latent_size, latent_size, device=device)
            use_multi_condition = should_use_multi_condition(shift, num_multi_condition)
            y, cfg_scale, label_indices = build_conditioning(
                class_label=class_label,
                spec_indices=spec_indices,
                batch_size=batch_size,
                use_multi_condition=use_multi_condition,
                label_weight=args.label_weight,
                device=device,
            )

            z = torch.cat([z, z], dim=0)
            model_kwargs = dict(y=y, cfg_scale=cfg_scale)

            samples = diffusion.p_sample_loop(
                scheduled_model,
                z.shape,
                z,
                clip_denoised=False,
                model_kwargs=model_kwargs,
                progress=False,
                device=device,
            )
            samples, _ = samples.chunk(2, dim=0)
            samples = vae.decode(samples / 0.18215).sample

            for image_index, image in enumerate(samples):
                label_suffix = "_".join(str(idx) for row in label_indices for idx in row)
                file_index = image_index + shift * batch_size + args.total_shift
                mode_tag = "multi" if use_multi_condition else "single"
                if use_multi_condition:
                    file_name = f"{file_index}_{label_suffix}_{mode_tag}_w{args.label_weight:.2f}.png"
                else:
                    file_name = f"{file_index}_{label_suffix}_{mode_tag}.png"
                image_path = os.path.join(args.save_dir, sel_class, file_name)
                save_image(image, image_path, normalize=True, value_range=(-1, 1))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, choices=list(DiT_models.keys()), default="DiT-XL/2")
    parser.add_argument("--vae", type=str, choices=["ema", "mse"], default="mse")
    parser.add_argument("--image-size", type=int, choices=[256, 512], default=256)
    parser.add_argument("--num-classes", type=int, default=1000)
    parser.add_argument("--num-sampling-steps", type=int, default=50)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--ckpt",
        type=str,
        default=None,
        help="Optional path to a DiT checkpoint (default: auto-download a pre-trained DiT-XL/2 model).",
    )
    parser.add_argument("--spec", type=str, default="none", help="specific subset for generation")
    parser.add_argument("--save-dir", type=str, default="../logs/test", help="directory to save generated images")
    parser.add_argument("--num-samples", type=int, default=100, help="desired IPC for generation")
    parser.add_argument("--total-shift", type=int, default=0, help="index offset for output file names")
    parser.add_argument("--nclass", type=int, default=10, help="number of classes for generation")
    parser.add_argument("--phase", type=int, default=0, help="phase number for generating large datasets")
    parser.add_argument("--multi-cond-ratio", type=float, default=0.5,
                        help="Fraction of samples per class generated with multi-condition conditioning.")
    parser.add_argument("--high-noise-until", type=float, default=0.7,
                        help="Noise-level threshold above which multi-condition sampling uses the high-stage CFG multiplier.")
    parser.add_argument("--low-noise-from", type=float, default=0.3,
                        help="Noise-level threshold below which multi-condition sampling uses the low-stage CFG multiplier.")
    parser.add_argument("--high-cfg-scale-mult", type=float, default=0.9,
                        help="Multiplier on multi-condition cfg_scale during the high-noise stage.")
    parser.add_argument("--mid-cfg-scale-mult", type=float, default=1.0,
                        help="Multiplier on multi-condition cfg_scale during the mid-noise stage.")
    parser.add_argument("--low-cfg-scale-mult", type=float, default=1.15,
                        help="Multiplier on multi-condition cfg_scale during the low-noise stage.")
    parser.add_argument("--high-boundary-factor", type=float, default=0.1,
                        help="Fraction of boundary mixing retained in the high-noise stage for multi-condition sampling.")
    parser.add_argument("--low-boundary-factor", type=float, default=0.35,
                        help="Fraction of boundary mixing retained in the low-noise stage for multi-condition sampling.")
    parser.add_argument("--disable-stage-cfg", action="store_false", dest="enable_stage_cfg",
                        help="Disable stage-wise CFG scheduling and boundary-weight scheduling for the multi-condition half of sampling.")
    parser.add_argument(
        "--label-weight",
        type=float,
        default=0.5,
        help="Primary-label mixing weight used during the multi-condition half of sampling.",
    )
    args = parser.parse_args()
    if not (1.0 >= args.high_noise_until >= args.low_noise_from >= 0.0):
        raise ValueError("Expected 1.0 >= high_noise_until >= low_noise_from >= 0.0")
    if not (0.0 <= args.multi_cond_ratio <= 1.0):
        raise ValueError("Expected 0.0 <= multi_cond_ratio <= 1.0")
    main(args)
