"""
Sample new images from a pre-trained DiT.
"""
import os
import torch
from tqdm import tqdm
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
from torchvision.utils import save_image
from diffusion import create_diffusion
from diffusers.models import AutoencoderKL
from download import find_model
from models import DiT_models
import argparse
from resnet import resnet18
import numpy as np
from misc.utils import load_model
from torchvision import transforms
import warnings
warnings.filterwarnings("ignore")


def get_stage_scheduled_weight(base_weight, noise_level, args):
    """
    Schedule the primary-vs-boundary mixing weight across timesteps.
    noise_level: 1.0 means highest noise, 0.0 means final denoising step.
    """
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


class StageAwareCFG:
    def __init__(self, model, num_timesteps, args):
        self.model = model
        self.num_timesteps = num_timesteps
        self.args = args

    def __call__(self, x, t, y, cfg_scale):
        if not self.args.enable_stage_cfg:
            return self.model.forward_with_cfg(x, t, y, cfg_scale)

        noise_level = t[0].item() / max(self.num_timesteps - 1, 1)
        scheduled_cfg_scale = get_stage_cfg_scale(cfg_scale, noise_level, self.args)

        if y.shape[0] == 4:
            scheduled_y = y.clone()
            base_weight = scheduled_y[-1]
            scheduled_y[-1] = get_stage_scheduled_weight(base_weight, noise_level, self.args)
        else:
            scheduled_y = y

        return self.model.forward_with_cfg(x, t, scheduled_y, scheduled_cfg_scale)

def main(args):
    # Setup PyTorch:
    torch.manual_seed(args.seed)
    torch.set_grad_enabled(False)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Labels to condition the model
    with open('./misc/class_indices.txt', 'r') as fp:
        all_classes = fp.readlines()
    all_classes = [class_index.strip() for class_index in all_classes]
    if args.spec == 'woof':
        file_list = './misc/class_woof.txt'
    elif args.spec == 'nette':
        file_list = './misc/class_nette.txt'
    elif args.spec == '100':
        file_list = './misc/class100.txt'
    elif args.spec == '1k':
        file_list = './misc/class_indices.txt'
    else:
        raise ValueError(f"Unsupported dataset spec '{args.spec}'.")
    with open(file_list, 'r') as fp:
        sel_classes = fp.readlines()

    phase = max(0, args.phase)
    cls_from = args.nclass * phase
    cls_to = args.nclass * (phase + 1)
    sel_classes = sel_classes[cls_from:cls_to]
    sel_classes = [sel_class.strip() for sel_class in sel_classes]
    class_labels = []
    
    for sel_class in sel_classes:
        class_labels.append(all_classes.index(sel_class))

    if args.ckpt is None:
        assert args.model == "DiT-XL/2", "Only DiT-XL/2 models are available for auto-download."
        assert args.image_size in [256, 512]
        assert args.num_classes == 1000

    # Load model:
    latent_size = args.image_size // 8
    model = DiT_models[args.model](
        input_size=latent_size,
        num_classes=args.num_classes
    ).to(device)
    # Auto-download a pre-trained model or load a custom DiT checkpoint from train.py:
    ckpt_path = args.ckpt or f"DiT-XL-2-{args.image_size}x{args.image_size}.pt"
    state_dict = find_model(ckpt_path)
    model.load_state_dict(state_dict, strict=False)
    model.eval()  # important!
    diffusion = create_diffusion(str(args.num_sampling_steps))
    scheduled_model = StageAwareCFG(model, diffusion.num_timesteps, args)
    print(args.vae)
    vae_path = f"./pretrained_models/stabilityai/sd-vae-ft-{args.vae}"
    vae = AutoencoderKL.from_pretrained(vae_path).to(device)

    expert_model = None
    expert_transform = None
    if args.enable_selection:
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
                        # transforms.ToTensor(),
                        expert_normalize,
                    ]
                )
        expert_model.eval()
        expert_model.to(device)
        for p in expert_model.parameters():
            p.requires_grad = False


    batch_size = 1
    spec_indices = torch.tensor(class_labels, device=device)
    # confusion_matrix = np.load("/data/mmc_lyxiang/DD/MinimaxDiffusion/results/test/confusion_matrix_epoch_0.npy")
    # confusion_matrix = torch.tensor(confusion_matrix, device=device)
    # boundary_dist = [5, 3, 2, 3, 2]
    # boundary_dist = [1] * 5
    # boundary_dist = torch.tensor(boundary_dist, device=device)


    imagenet_to_subset = {
        imagenet_id: subset_id
        for subset_id, imagenet_id in enumerate(class_labels)
    }

    for class_label, sel_class in zip(class_labels, sel_classes):
        print(class_label)
        os.makedirs(os.path.join(args.save_dir, sel_class), exist_ok=True)
        for shift in tqdm(range(args.num_samples // batch_size)):
            # Create sampling noise:
            candidate_batch_size = batch_size * args.num_candidates if args.enable_selection else batch_size
            z = torch.randn(candidate_batch_size, 4, latent_size, latent_size, device=device)
            mask = spec_indices != torch.tensor(class_label, device=device)
            candidates = spec_indices[mask]
            num_random = max(0, 1)
            num_random = min(num_random, len(candidates))
            if num_random > 0:
                random_indices = torch.randperm(len(candidates), device=device)[:num_random]
                random_label = candidates[random_indices]
                random_labels = random_label.repeat(candidate_batch_size)
            else:
                random_labels = torch.empty(0, dtype=torch.long, device=device)
            primary_label = torch.tensor([class_label] * candidate_batch_size, device=device)
            # class_indices = (spec_indices == class_label).nonzero(as_tuple=True)[0].item()
            # confusion_row = confusion_matrix[class_indices].clone()
            # confusion_row[class_indices] = 0  # 去掉自己
            # probs = confusion_row / (confusion_row.sum() + 1e-8)
            # topk = torch.topk(probs, k=num_random)  # 取前 num_random 个最混类
            # random_indices = topk.indices
            # random_labels = spec_indices[random_indices]
            conditioned_labels = torch.stack([primary_label, random_labels], 0)
            # confusion_rate = confusion_matrix[random_indices, class_indices] / (confusion_matrix[class_indices, random_indices] + confusion_matrix[random_indices, class_indices])
            # idx = int(shift / 10)
            # label_weight = 0.5 * torch.exp( - 0.7 * (torch.tensor(5) - 0.25) )
            # label_weight = 0.9
            # label_weight = 0.5 * (shift + 1) / (args.num_samples / batch_size)

            # weight_id = 0
            weight_id = min(1, shift * 2 // (args.num_samples // batch_size))
            # if weight_id < 7:
            #     label_weight = 0.5
            # else:
            #     label_weight = 1.0
            label_weight = 0.5 * weight_id + 0.5
            # label_weight = 0.5


            # weight_id_candidates = [0, 5]
            # weight_id = weight_id_candidates[shift % len(weight_id_candidates)]
            # label_weight = 0.1 * weight_id + 0.5

            confusion_rate = torch.tensor([label_weight] * candidate_batch_size, device= device)
            label_indices = conditioned_labels.detach().cpu().tolist()
              
            # Setup classifier-free guidance:
            z = torch.cat([z, z], 0)
            y_null = torch.tensor([1000] * candidate_batch_size, device=device)
            y = torch.cat([conditioned_labels, y_null.unsqueeze(0)], 0)
            y = torch.cat([y, confusion_rate.unsqueeze(0)], 0)              # 最后一个元素是混淆率作为权重
            model_kwargs = dict(y=y, cfg_scale=args.cfg_scale)

            # Sample images:
            samples = diffusion.p_sample_loop(
                scheduled_model, z.shape, z,
                clip_denoised=False, model_kwargs=model_kwargs,
                progress=False, device=device
            )
            samples, _ = samples.chunk(2, dim=0)
            samples = vae.decode(samples / 0.18215).sample

            if args.enable_selection:
                with torch.no_grad():
                    images = (samples + 1) / 2
                    logits = expert_model(expert_transform(images))
                    probs = torch.softmax(logits, dim=1)

                primary_idx = imagenet_to_subset[class_label]
                boundary_idx = imagenet_to_subset[random_label.item()]

                p_primary = probs[:, primary_idx]
                p_boundary = probs[:, boundary_idx]

                lambda_hat = p_primary / (p_primary + p_boundary + 1e-8)

                lambda_target = label_weight

                delta = (lambda_hat - lambda_target).abs()

                best_idx = delta.argmin()  # [B]
                final_samples = samples[best_idx].unsqueeze(0)
            else:
                final_samples = samples[:batch_size]


            # Save and display images:
            for image_index, image in enumerate(final_samples):
                label_tokens = [str(idx) for idx in label_indices]
                label_suffix = "_".join(label_tokens) if label_tokens else str(class_label)
                file_index = image_index + shift * batch_size + args.total_shift
                file_name = f"{label_suffix}_w{weight_id}_{file_index}.png"
                image_path = os.path.join(args.save_dir, sel_class, file_name)
                save_image(image, image_path, normalize=True, value_range=(-1, 1))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, choices=list(DiT_models.keys()), default="DiT-XL/2")
    parser.add_argument("--vae", type=str, choices=["ema", "mse"], default="mse")
    parser.add_argument("--image-size", type=int, choices=[256, 512], default=256)
    parser.add_argument("--num-classes", type=int, default=1000)
    parser.add_argument("--cfg-scale", type=float, default=4.0)
    parser.add_argument("--num-sampling-steps", type=int, default=50)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--ckpt", type=str, default=None,
                        help="Optional path to a DiT checkpoint (default: auto-download a pre-trained DiT-XL/2 model).")
    parser.add_argument("--spec", type=str, default='none', help='specific subset for generation')
    parser.add_argument("--save-dir", type=str, default='../logs/test', help='the directory to put the generated images')
    parser.add_argument("--num-samples", type=int, default=100, help='the desired IPC for generation')
    parser.add_argument("--total-shift", type=int, default=0, help='index offset for the file name')
    parser.add_argument("--nclass", type=int, default=10, help='the class number for generation')
    parser.add_argument("--phase", type=int, default=0, help='the phase number for generating large datasets')
    parser.add_argument("--high-noise-until", type=float, default=0.7,
                        help="Noise-level threshold above which sampling stays close to single-condition guidance.")
    parser.add_argument("--low-noise-from", type=float, default=0.3,
                        help="Noise-level threshold below which target-class guidance is strengthened again.")
    parser.add_argument("--high-boundary-factor", type=float, default=0.1,
                        help="Fraction of boundary mixing retained in the high-noise stage; 0 means nearly single-condition.")
    parser.add_argument("--low-boundary-factor", type=float, default=0.35,
                        help="Fraction of boundary mixing retained in the low-noise stage when restoring target fidelity.")
    parser.add_argument("--high-cfg-scale-mult", type=float, default=0.9,
                        help="Multiplier on cfg_scale during the high-noise stage.")
    parser.add_argument("--mid-cfg-scale-mult", type=float, default=1.0,
                        help="Multiplier on cfg_scale during the mid-noise stage.")
    parser.add_argument("--low-cfg-scale-mult", type=float, default=1.15,
                        help="Multiplier on cfg_scale during the low-noise stage.")
    parser.add_argument("--disable-stage-cfg", action="store_false", dest="enable_stage_cfg",
                        help="Disable stage-wise CFG scheduling and use a fixed cfg_scale for all timesteps.")
    parser.add_argument("--disable-selection", action="store_false", dest="enable_selection",
                        help="Disable expert-based candidate selection and save the sampled result directly.")
    parser.add_argument("--num-candidates", type=int, default=4,
                        help="Number of candidates to generate per sample when expert-based selection is enabled.")
    # parser.add_argument("--num-random-labels", type=int, default=1, help='number of additional random labels to include for conditioning')
    args = parser.parse_args()
    if args.num_candidates < 1:
        raise ValueError("Expected num_candidates >= 1")
    if not (1.0 >= args.high_noise_until >= args.low_noise_from >= 0.0):
        raise ValueError("Expected 1.0 >= high_noise_until >= low_noise_from >= 0.0")
    main(args)
