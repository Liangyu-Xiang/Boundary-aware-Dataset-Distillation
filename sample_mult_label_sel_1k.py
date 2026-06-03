"""
Sample new images from a pre-trained DiT.
"""
import os
import math
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

    # =========================
    # Class range selection
    # =========================
    num_all_classes = len(all_classes)

    if args.class_from is not None or args.class_to is not None:
        # —— 显式指定 from / to（优先生效）——
        cls_from = 0 if args.class_from is None else args.class_from
        cls_to   = num_all_classes if args.class_to is None else args.class_to

        assert 0 <= cls_from < cls_to <= num_all_classes, \
            f"Invalid class range: [{cls_from}, {cls_to})"

        print(f"[INFO] Using explicit class range: [{cls_from}, {cls_to})")

    else:
        # —— 保持原有 phase / nclass 行为 —— 
        phase = max(0, args.phase)
        cls_from = args.nclass * phase
        cls_to = min(num_all_classes, args.nclass * (phase + 1))

        print(f"[INFO] Using phase-based class range: [{cls_from}, {cls_to})")
        
    sel_classes = sel_classes[cls_from:cls_to]
    sel_classes = [sel_class.strip() for sel_class in sel_classes]
    class_labels = []
    all_class_labels = []
    
    for sel_class in sel_classes:
        class_labels.append(all_classes.index(sel_class))
    for a_class in all_classes:
        all_class_labels.append(all_classes.index(a_class))

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
    print(args.vae)
    vae_path = f"./pretrained_models/stabilityai/sd-vae-ft-{args.vae}"
    vae = AutoencoderKL.from_pretrained(vae_path).to(device)

    
    expert_model = load_model(
        model_name='resnet18',
        dataset=args.spec,
        pretrained=True,
        classes=range(1000)
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


    batch_size = 8
    spec_indices = torch.tensor(all_class_labels, device=device)
    # confusion_matrix = np.load("/data/mmc_lyxiang/DD/MinimaxDiffusion/results/test/confusion_matrix_epoch_0.npy")
    # confusion_matrix = torch.tensor(confusion_matrix, device=device)
    # boundary_dist = [5, 3, 2, 3, 2]
    # boundary_dist = [1] * 5
    # boundary_dist = torch.tensor(boundary_dist, device=device)


    imagenet_to_subset = {
        imagenet_id: subset_id
        for subset_id, imagenet_id in enumerate(all_class_labels)
    }

    for class_label, sel_class in zip(class_labels, sel_classes):
        print(class_label)
        os.makedirs(os.path.join(args.save_dir, sel_class), exist_ok=True)
        for shift in tqdm(range(math.ceil(args.num_samples / batch_size))):
            # Create sampling noise:
            candidate_batch_size = batch_size * 4
            z = torch.randn(candidate_batch_size, 4, latent_size, latent_size, device=device)
            mask = spec_indices != torch.tensor(class_label, device=device)
            candidates = spec_indices[mask]
            candidates = spec_indices[mask]
            num_candidates = len(candidates)
            assert num_candidates > 0, "No valid random labels available."
            rand_idx = torch.randint(
                low=0,
                high=num_candidates,
                size=(candidate_batch_size,),
                device=device
            )
            random_labels = candidates[rand_idx]
            # num_random = max(0, 1)
            # num_random = min(num_random, len(candidates))
            # if num_random > 0:
            #     random_indices = torch.randperm(len(candidates), device=device)[:num_random]
            #     random_label = candidates[random_indices]
            #     random_labels = random_label.repeat(candidate_batch_size)
            # else:
            #     random_labels = torch.empty(0, dtype=torch.long, device=device)
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
                model.forward_with_cfg, z.shape, z,
                clip_denoised=False, model_kwargs=model_kwargs,
                progress=False, device=device
            )
            samples, _ = samples.chunk(2, dim=0)
            samples = vae.decode(samples / 0.18215).sample

            with torch.no_grad():
                images = (samples + 1) / 2
                logits = expert_model(expert_transform(images))
                probs = torch.softmax(logits, dim=1)
            
            primary_idx = imagenet_to_subset[class_label]
            boundary_indices = torch.tensor(
                [imagenet_to_subset[l.item()] for l in random_labels],
                device=device
            )

            p_primary = probs[:, primary_idx]
            p_boundary = probs.gather(
                1, boundary_indices.unsqueeze(1)
            ).squeeze(1)

            lambda_hat = p_primary / (p_primary + p_boundary + 1e-8)

            lambda_target = label_weight

            delta = (lambda_hat - lambda_target).abs()

            _, best_idx = torch.topk(
                delta,
                k=batch_size,
                largest=False,   # 取最小的
                sorted=True
            )

            final_samples = samples[best_idx]
            final_random_labels = random_labels[best_idx]

            remain = args.num_samples - shift * batch_size
            if remain < batch_size:
                final_samples = final_samples[:remain]
                final_random_labels = final_random_labels[:remain]


            # Save and display images:
            for image_index, image in enumerate(final_samples):
                pri = class_label
                rnd = final_random_labels[image_index].item()

                label_suffix = f"{pri}_{rnd}"
                file_index = (
                    shift * batch_size
                    + image_index
                    + args.total_shift
                )

                file_name = f"{label_suffix}_w{weight_id}_{file_index}.png"
                image_path = os.path.join(args.save_dir, sel_class, file_name)

                save_image(
                    image,
                    image_path,
                    normalize=True,
                    value_range=(-1, 1)
                )


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
    parser.add_argument("--class-from", type=int, default=None,
                    help="start class index (inclusive)")
    parser.add_argument("--class-to", type=int, default=None,
                        help="end class index (exclusive)")
    # parser.add_argument("--num-random-labels", type=int, default=1, help='number of additional random labels to include for conditioning')
    args = parser.parse_args()
    main(args)
