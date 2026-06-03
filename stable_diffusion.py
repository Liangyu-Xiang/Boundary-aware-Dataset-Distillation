from diffusers import StableDiffusionPipeline
import torch

model_path = "/data/mmc_lyxiang/DD/MinimaxDiffusion/pretrained_models/stable-diffusion-v1-5/"
pipe = StableDiffusionPipeline.from_pretrained(model_path, torch_dtype=torch.float16)
pipe = pipe.to("cuda")

prompt = "a photo of Golden retriever and Samoyed"
image = pipe(prompt).images[0]  
    
image.save("SD_Generated.png")