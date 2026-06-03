"""
Sample new images from a pre-trained DiT.
"""
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
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
from data import transform_imagenet
import train_models.resnet as RN
from torchvision.transforms import v2



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
    else:
        file_list = './misc/class100.txt'
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
    # expert_model = resnet18(pretrained=False).to(device)
    expert_path = "/data/mmc_lyxiang/DD/MinimaxDiffusion/results/imagenet10/resnet18in_evidential_resnet18imagewoof_cut/model_best.pth.tar"
    state_dict = torch.load(expert_path)["state_dict"]
    expert_model.load_state_dict(state_dict)
    expert_model.eval()
    transform, _ =  transform_imagenet(augment=False,
                                size=224,
                                from_tensor=True)
    # transform = v2.Compose([
    #                         v2.Resize(256),
    #                         v2.CenterCrop(224),
    #                         v2.Normalize(mean=[0.485, 0.456, 0.406],
    #                                     std=[0.229, 0.224, 0.225])
    #                     ])

    
    batch_size = 1
    confident_flag = 0 # 该flag表示当前生成的图像和哪个类别相比最不unconfident
    max_per_subclass = args.num_samples // 10
    samples_counts = [0] * 10
    shift = 0
    class_to_idx = {'n02096294': 0, 'n02093754': 1, 'n02111889': 2, 'n02088364': 3, 'n02086240': 4, 'n02089973': 5, 'n02087394': 6, 'n02115641': 7, 'n02099601': 8, 'n02105641': 9}
    for class_label, sel_class in zip(class_labels, sel_classes):
        os.makedirs(os.path.join(args.save_dir, sel_class), exist_ok=True)
        samples_counts = [0] * 10
        shift = 0
        for _ in tqdm(range(args.num_samples * 10)):
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
                # woof_indices = [155, 159, 162, 167, 182, 193, 207, 229, 258, 273]
                # logits = expert_model(transform(image).unsqueeze(0))[0][:, woof_indices]
                # alpha = torch.exp(logits) + torch.exp(torch.tensor(-1.43))
                # S = torch.sum(alpha, dim=-1, keepdim=True)
                # probs = alpha / S
                # probs = probs.squeeze(0)

                logits = expert_model(transform(image).unsqueeze(0))
                alpha = torch.exp(logits) + 1
                S = torch.sum(alpha, dim=-1, keepdim=True)
                probs = alpha / S
                probs = probs.squeeze(0)

                top3_probs, top3_indices = torch.topk(probs, 3)
                for rank, (p, idx) in enumerate(zip(top3_probs.tolist(), top3_indices.tolist()), start=1):
                    print(f"  Top{rank} -> Class {idx}, Prob = {p:.4f}")

                max_p, pred_class = torch.max(probs, dim=0)
                print("Current Sampling Class:", class_to_idx[sel_class], "Predicted Class:", pred_class)
                top2 = torch.topk(probs, 2)
                second_p, second_class = top2.values[1].item(), top2.indices[1].item()
                if (max_p - second_p) > 0.8 and pred_class == class_to_idx[sel_class]:
                    c = pred_class.item()
                    if samples_counts[c] < max_per_subclass:
                        save_image(
                            image,
                            os.path.join(
                                args.save_dir,
                                sel_class,
                                f"{image_index + shift * batch_size + args.total_shift}.png"
                            ),
                            normalize=True, value_range=(-1, 1)
                        )
                        samples_counts[c] += 1
                        shift += 1

                # ---- 判断并保存 unconfident 样本 ----
                elif (max_p - second_p) < 0.2 and pred_class == class_to_idx[sel_class]:
                    uc = second_class
                    if samples_counts[uc] < max_per_subclass:
                        save_image(
                            image,
                            os.path.join(
                                args.save_dir,
                                sel_class,
                                f"{image_index + shift * batch_size + args.total_shift}.png"
                            ),
                            normalize=True, value_range=(-1, 1)
                        )
                        samples_counts[uc] += 1
                        shift += 1
                # elif class_label != woof_indices[pred_class]:
                #     uc = pred_class
                #     if samples_counts[uc] < max_per_subclass:
                #         save_image(
                #             image,
                #             os.path.join(
                #                 args.save_dir,
                #                 sel_class,
                #                 f"{image_index + shift * batch_size + args.total_shift}.png"
                #             ),
                #             normalize=True, value_range=(-1, 1)
                #         )
                #         samples_counts[uc] += 1
                #         shift += 1
        if min(samples_counts) >= max_per_subclass:
            print(f"✅ 所有类别已采样 {max_per_subclass} 张，提前结束采样。")
            break


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
