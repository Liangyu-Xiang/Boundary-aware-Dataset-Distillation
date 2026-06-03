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
from data import transform_imagenet
import train_models.resnet as RN
import argparse
from resnet import resnet18

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
    elif args.spec == 'imagenet1k':
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

    expert_model = RN.ResNet('imagenet',
                          18,
                          10,
                          norm_type='instance',
                          size=224,
                          nch=3).to(device)
    expert_path = "/data/mmc_lyxiang/DD/MinimaxDiffusion/results/imagenet10/resnet18in_EDL_NLL_noKL_cut/model_best.pth.tar"
    state_dict = torch.load(expert_path)["state_dict"]
    expert_model.load_state_dict(state_dict)
    expert_model.eval()
    expert_transform, _ =  transform_imagenet(augment=False,
                                size=224,
                                from_tensor=True)


    batch_size = 1
    class_to_idx = {'n02096294': 0, 'n02093754': 1, 'n02111889': 2, 'n02088364': 3, 'n02086240': 4, 'n02089973': 5, 'n02087394': 6, 'n02115641': 7, 'n02099601': 8, 'n02105641': 9}
    for class_label, sel_class in zip(class_labels, sel_classes):
        os.makedirs(os.path.join(args.save_dir, sel_class), exist_ok=True)
        for shift in tqdm(range(args.num_samples // batch_size)):
            # Create sampling noise:
            z = torch.randn(batch_size, 4, latent_size, latent_size, device=device)
            y = torch.tensor([class_label], device=device)

            # Setup classifier-free guidance:
            z = torch.cat([z, z], 0)
            y_null = torch.tensor([1000] * batch_size, device=device)
            y = torch.cat([y, y_null], 0)
            model_kwargs = dict(y=y, cfg_scale=args.cfg_scale)

            # Sample images:
            samples = diffusion.p_sample_loop(
                model.forward_with_cfg, z.shape, z, clip_denoised=False, model_kwargs=model_kwargs, progress=False, device=device
            )
            samples, _ = samples.chunk(2, dim=0)  # Remove null class samples
            samples = vae.decode(samples / 0.18215).sample

            # Save and display images:
            for image_index, image in enumerate(samples):
                class_id = class_to_idx[sel_class]
                logits = expert_model(expert_transform(image).unsqueeze(0))

                # EDL evidence
                evidence = torch.nn.functional.softplus(logits)
                alpha = evidence + 1
                alpha0 = alpha.sum(dim=1, keepdim=True)
                term = torch.digamma(alpha + 1.0) - torch.digamma(alpha0 + 1.0)                  # posterior mean (EDL probability)
                data_u = - torch.sum((alpha / alpha0) * term, dim=1) # E_p[ H(P(y|π)) ]

                # ----- High-confidence filtering -----
                conf = data_u
                data_u_norm = (conf - 1.0) / (1.5 - 1.0 + 1e-6)
                data_u_norm = torch.clamp(data_u_norm, 0.0, 1.0) 
                pred_class = evidence.argmax().item()

                # 设置阈值（你可以挪到 argparse）
                UNCERT_THRESHOLD = 0.15
                print(data_u_norm.item())

                # 只保存置信度高的样本，且预测类别正确
                if data_u_norm.item() <= UNCERT_THRESHOLD and pred_class == class_id:
                    save_path = os.path.join(
                        args.save_dir,
                        sel_class,
                        f"{image_index + shift * batch_size + args.total_shift}.png"
                    )
                    save_image(image, save_path, normalize=True, value_range=(-1, 1))


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
    args = parser.parse_args()
    main(args)
