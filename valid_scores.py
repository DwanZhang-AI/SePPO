import os
import random
import numpy as np
import argparse
import torch
from datasets import load_dataset
from huggingface_hub import login
from diffusers import StableDiffusionPipeline, UNet2DConditionModel, StableDiffusionXLPipeline
from PIL import Image
from statistics import mean
import ImageReward as RM
import json
from utils.pickscore_utils import Selector as pick_selector
from utils.aes_utils import Selector as aes_Selector
from utils.hps_utils import Selector as hps_Selector

# Disable gradients for inference
torch.set_grad_enabled(False)

# Define helper functions
def read_hpd(json_path):
    with open(json_path, 'r') as file:
        data = json.load(file)
        return data

def median(lst):
    sorted_lst = sorted(lst)
    n = len(sorted_lst)

    if n == 0:
        raise ValueError("The list is empty")

    mid = n // 2

    if n % 2 == 0:
        return (sorted_lst[mid - 1] + sorted_lst[mid]) / 2.0
    else:
        return sorted_lst[mid]

def gen(model, prompt, generator, gs):
    im = model(prompt=prompt, generator=generator, guidance_scale=gs).images[0]
    return im

# Add argparse functionality
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate images and calculate scores.")
    
    # Arguments for Huggingface login and dataset paths
    parser.add_argument('--token', type=str, required=True, help="Huggingface token for login")
    parser.add_argument('--json_path', type=str, help="Path to the hps JSON file")
    parser.add_argument('--pretrained_model_name', type=str, default="pt-sk/stable-diffusion-1.5", help="Pretrained model to use")
    parser.add_argument('--unet_checkpoint', type=str, required=True, help="Path to the UNet model checkpoint")
    parser.add_argument('--dataset_name', type=str, required=True, help="Dataset name to load (e.g. pickapic, parti)")

    args = parser.parse_args()

    # Login to Huggingface
    login(token=args.token)

    # Load models and selectors
    ps_selector = pick_selector('cuda')
    aes_selector = aes_Selector('cuda')
    hps_selector = hps_Selector('cuda')
    rm_model = RM.load("ImageReward-v1.0")

    # Load the dataset
    if args.dataset_name == "pickapic_valid":
        val_dataset = load_dataset("yuvalkirstain/pickapic_v2_no_images", split="validation_unique")
    elif args.dataset_name == "parti_prompt":
        val_dataset = load_dataset('nateraw/parti-prompts', split="train")
    else:
        val_dataset = read_hpd(args.json_path)

    # Load the UNet model from the checkpoint
    unet = UNet2DConditionModel.from_pretrained(
        args.unet_checkpoint, torch_dtype=torch.float16
    ).to('cuda')

    # Guidance scale setting
    gs = 7.5 if 'stable-diffusion-1' in args.pretrained_model_name else 5

    # Load the appropriate Stable Diffusion pipeline
    if 'stable-diffusion-1' in args.pretrained_model_name:
        pipe = StableDiffusionPipeline.from_pretrained(
            args.pretrained_model_name, torch_dtype=torch.float16
        )
    else:
        pipe = StableDiffusionXLPipeline.from_pretrained(
            args.pretrained_model_name, torch_dtype=torch.float16,
            variant="fp16", use_safetensors=True
        )

    pipe = pipe.to('cuda')
    pipe.safety_checker = None
    pipe.unet = unet

    # Set a fixed generator for deterministic output
    generator = torch.Generator(device='cuda').manual_seed(0)

    # Initialize score lists
    score_list = []
    aes_list = []
    hps_list = []
    rm_list = []

    # Iterate over the dataset and generate images
    for i, prompt in enumerate(val_dataset):
        if args.dataset_name == "parti_prompt":
            cap = prompt['Prompt']
        elif args.dataset_name == "pickapic_valid":
            cap = prompt["caption"]
        else:
            cap = prompt['prompt']
        
        ims = gen(pipe, cap, generator, gs)
        ps_scores = ps_selector.score(ims, cap)
        aes_scores = aes_selector.score(ims, cap)
        hps_scores = hps_selector.score(ims, cap)
        rm_scores = rm_model.score(cap, ims)

        score_list.extend(ps_scores)
        aes_list.extend(aes_scores)
        hps_list.extend(hps_scores)
        rm_list.append(rm_scores)

    # Print final average scores
    print(f"Final Score: PickScore: {mean(score_list)}, AES: {mean(aes_list)}, HPSv2_score: {mean(hps_list)}, ImageReward: {mean(rm_list)}")
