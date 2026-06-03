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
    print(args.vae)
    vae_path = f"./pretrained_models/stabilityai/sd-vae-ft-{args.vae}"
    vae = AutoencoderKL.from_pretrained(vae_path).to(device)




    
    # expert_model = resnet18(pretrained=False).to(device)
    # expert_path = "/data/mmc_lyxiang/KD/EKD/output/Evidential_Teacher/ResNet18_ImageNet/student_best"
    # state_dict = torch.load(expert_path)["model"]
    # expert_model.load_state_dict(state_dict)
    # expert_model.eval()


    batch_size = 1
    woof_indices = torch.tensor([155, 159, 162, 167, 182, 193, 207, 229, 258, 273], device=device)
    spec_indices = torch.tensor(class_labels, device=device)
    # confusion_matrix = np.load("/data/mmc_lyxiang/DD/MinimaxDiffusion/results/test/confusion_matrix_epoch_0.npy")
    # confusion_matrix = torch.tensor(confusion_matrix, device=device)
    # boundary_dist = [5, 3, 2, 3, 2]
    # boundary_dist = [1] * 5
    # boundary_dist = torch.tensor(boundary_dist, device=device)

    # Class Embedding From DiT
    class_embedding = model.y_embedder(spec_indices, train=False)



    for class_label, sel_class in zip(class_labels, sel_classes):
        print(class_label)
        os.makedirs(os.path.join(args.save_dir, sel_class), exist_ok=True)
        for shift in tqdm(range(args.num_samples // batch_size)):
            # Create sampling noise:
            z = torch.randn(batch_size, 4, latent_size, latent_size, device=device)
            mask = spec_indices != torch.tensor(class_label, device=device)
            candidates = spec_indices[mask]
            num_random = max(0, 1)
            num_random = min(num_random, len(candidates))
            if num_random > 0:
                random_indices = torch.randperm(len(candidates), device=device)[:num_random]
                random_labels = candidates[random_indices]
            else:
                random_labels = torch.empty(0, dtype=torch.long, device=device)
            primary_label = torch.tensor([class_label], device=device)
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


            # weight_id_candidates = [0, 5]
            # weight_id = weight_id_candidates[shift % len(weight_id_candidates)]
            # label_weight = 0.1 * weight_id + 0.5

            confusion_rate = torch.tensor([label_weight], device= device)
            label_indices = conditioned_labels.detach().cpu().tolist()
              
            # Setup classifier-free guidance:
            z = torch.cat([z, z], 0)
            y_null = torch.tensor([1000] * batch_size, device=device)
            y = torch.cat([conditioned_labels, y_null.unsqueeze(0)], 0)
            y = torch.cat([y, confusion_rate.unsqueeze(0)], 0)              # 最后一个元素是混淆率作为权重
            model_kwargs = dict(y=y, cfg_scale=args.cfg_scale)

            # Sample images:
            samples = diffusion.p_sample_loop(
                model.forward_with_cfg, z.shape, z, clip_denoised=False, model_kwargs=model_kwargs, progress=False, device=device
            )
            samples, _ = samples.chunk(2, dim=0)  # Remove null class samples
            samples = vae.decode(samples / 0.18215).sample

            # Save and display images:
            for image_index, image in enumerate(samples):
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
    # parser.add_argument("--num-random-labels", type=int, default=1, help='number of additional random labels to include for conditioning')
    args = parser.parse_args()
    main(args)
