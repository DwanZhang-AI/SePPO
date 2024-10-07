import os
import argparse
import numpy as np
import torch
from diffusers import StableDiffusionPipeline, UNet2DConditionModel
from PIL import Image

torch.set_grad_enabled(False)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate images and calculate scores.")
    parser.add_argument('--unet_checkpoint', type=str, required=True, help="Path to the UNet model checkpoint")
    parser.add_argument('--prompt', type=str, required=True, help="Prompt")

    args = parser.parse_args()

    unet = UNet2DConditionModel.from_pretrained(args.unet_checkpoint, torch_dtype=torch.float16).to('cuda')

    pipe = StableDiffusionPipeline.from_pretrained("pt-sk/stable-diffusion-1.5", torch_dtype=torch.float16)

    pipe = pipe.to('cuda')
    pipe.safety_checker = None
    pipe.unet = unet
    generator = torch.Generator(device='cuda').manual_seed(0)
    gs = 7.5

    ims = pipe(prompt=args.prompt, generator=generator, guidance_scale=gs).images[0]
    img_path = os.path.join('SePPO', "0.png")
    
    if isinstance(ims, np.ndarray):
        ims = Image.fromarray(ims)
    ims.save(img_path, format='PNG')
