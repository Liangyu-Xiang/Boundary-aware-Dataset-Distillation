"""
Sample images from a fine-tuned Stable Diffusion v1.5 UNet.
"""
import os
import argparse
from typing import List

import torch
from diffusers import StableDiffusionPipeline, UNet2DConditionModel
from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion_multi_label import StableDiffusionMultiLabelPipeline


def load_class_names(path: str) -> List[str]:
    with open(path, "r", encoding="utf-8") as f:
        return [line.strip() for line in f if line.strip()]


def encode_prompts(pipe: StableDiffusionPipeline, prompts: List[str], device: str):
    tokenizer = pipe.tokenizer
    text_encoder = pipe.text_encoder
    tokens = tokenizer(
        prompts,
        padding="max_length",
        truncation=True,
        max_length=tokenizer.model_max_length,
        return_tensors="pt",
    )
    input_ids = tokens.input_ids.to(device)
    attention_mask = tokens.attention_mask.to(device)
    with torch.no_grad():
        outputs = text_encoder(input_ids=input_ids, attention_mask=attention_mask)
    return outputs.last_hidden_state


def pick_second_class(class_names: List[str], idx: int, mode: str, generator: torch.Generator) -> str:
    if mode == "same":
        return class_names[idx]
    if mode == "next":
        return class_names[(idx + 1) % len(class_names)]
    # random
    if len(class_names) == 1:
        return class_names[idx]
    choices = list(range(len(class_names)))
    choices.remove(idx)
    rand_idx = torch.randint(0, len(choices), (1,), generator=generator, device="cpu").item()
    return class_names[choices[rand_idx]]


def main(args):
    torch.manual_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    pipe = StableDiffusionMultiLabelPipeline.from_pretrained(args.sd_path, torch_dtype=torch.float16)
    pipe = pipe.to("cuda")

    class_names = load_class_names(args.class_names_path)
    os.makedirs(args.output_dir, exist_ok=True)

    for class_idx, class_name in enumerate(class_names):
        class_dir = os.path.join(args.output_dir, class_name)
        os.makedirs(class_dir, exist_ok=True)

        half_point = max(1, args.num_samples_per_class // 2)
        for start in range(0, args.num_samples_per_class, args.batch_size):
            bs = min(args.batch_size, args.num_samples_per_class - start)
            gen_cpu = torch.Generator(device="cpu").manual_seed(args.seed + start)
            gen_cuda = torch.Generator(device=device).manual_seed(args.seed + start)

            prompt1 = args.prompt_template.format(class_name=class_name)
            class_name_2 = pick_second_class(
                class_names, class_idx, args.second_prompt_mode, gen_cpu
            )
            prompt2 = args.prompt_template.format(class_name=class_name_2)

            prompt1_list = [prompt1] * bs
            prompt2_list = [prompt2] * bs

            global_idx = start
            weight_id = min(1, global_idx * 2 // args.num_samples_per_class)
            boundary_weights = 0.5 - weight_id * 0.5


            images = pipe(
                prompt=prompt1_list,
                boundary_prompt=prompt2_list,
                boundary_weight = boundary_weights,
                height=args.resolution,
                width=args.resolution,
            ).images
            for i, image in enumerate(images):
                image.save(os.path.join(class_dir, f"{start + i:05d}.png"))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--sd_path", type=str, required=True)
    # parser.add_argument("--unet_path", type=str, required=True)
    parser.add_argument("--class_names_path", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--num_samples_per_class", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--num_inference_steps", type=int, default=30)
    parser.add_argument("--guidance_scale", type=float, default=7.5)
    parser.add_argument("--resolution", type=int, default=512)
    parser.add_argument("--precision", type=str, default="fp16", choices=["fp16", "bf16", "fp32"])
    parser.add_argument("--prompt_template", type=str, default="a photo of a {class_name}")
    parser.add_argument("--negative_prompt", type=str, default="")
    parser.add_argument("--second_prompt_mode", type=str, default="random", choices=["random", "next", "same"])
    parser.add_argument("--prompt_weight", type=float, default=0.5)
    parser.add_argument("--first_half_weight", type=float, default=0.5)
    parser.add_argument("--second_half_weight", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--attention_slicing", action="store_true", default=False)
    parser.add_argument("--show_progress", action="store_true", default=False)
    args = parser.parse_args()
    main(args)
