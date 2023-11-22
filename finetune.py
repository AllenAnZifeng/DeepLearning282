import os
from typing import Dict, Optional
import random

import wandb
import torch
import numpy as np

from diffusers import (AutoencoderKL, DDPMScheduler,
                       StableDiffusionInstructPix2PixPipeline,
                       UNet2DConditionModel)
from transformers import CLIPTextModel, CLIPTokenizer

from arguments import parse_args


# Setup WandB logger to log to team "eecs182"
def setup_logger(
    log_name: str,
    params: Dict,
    project_name: str = "colorization_finetune",
) -> wandb.Run:
    with open("wandb_api_key", "r") as f:
        api_key = f.readline()
    os.environ["WANDB_API_KEY"] = api_key
    os.environ["WANDB_PROJECT"] = project_name
    wandb.login()
    
    logger = wandb.init(
        project = project_name,
        entity = "eecs182",
        config = params,
        name = log_name,
    )
    return logger


# Load pretrained modules
def load_models(model_name: str):
    # Load submodules
    noise_scheduler = DDPMScheduler.from_pretrained(
        model_name,
        subfolder="scheduler"
    )
    tokenizer = CLIPTokenizer.from_pretrained(
        model_name,
        subfolder="tokenizer"
    )
    text_encoder = CLIPTextModel.from_pretrained(
        model_name,
        subfolder="text_encoder"
    )
    vae = AutoencoderKL.from_pretrained(
        model_name,
        subfolder="vae"
    )
    unet = UNet2DConditionModel.from_pretrained(
        model_name,
        subfolder="unet"
    )
    
    # Freeze some parameters
    vae.requires_grad_(False)
    text_encoder.requires_grad_(False)
    
    return noise_scheduler, tokenizer, text_encoder, vae, unet
    

def main():
    args = parse_args()
    
    # Setup reproducibility
    seed = args.seed
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    
    if (
        args.cuda
        and torch.cuda.is_available()
    ):
        torch.cuda.manual_seed_all(seed)
    
    logger = setup_logger(args.log_name, args.params) if args.log_name is not None else None

    noise_scheduler, tokenizer, text_encoder, vae, unet = load_models(args.params["model_name"])
    
    

if __name__ == "__main__":
    main()