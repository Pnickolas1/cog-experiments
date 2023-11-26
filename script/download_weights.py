#!/usr/bin/env python

# Run this before you deploy it on replicate
import os
import sys
import torch
from diffusers import DiffusionPipeline, AutoencoderKL

# append project directory to path so predict.py can be imported
sys.path.append('.')
from predict import MODEL_NAME, MODEL_CACHE, COLORING_BOOK_MODEL_NAME, COLORING_BOOK_WEIGHTS_NAME

# Make cache folders
if not os.path.exists(MODEL_CACHE):
    os.makedirs(MODEL_CACHE)

# Download SDXL-VAE-FP16-Fix
better_vae = AutoencoderKL.from_pretrained(
    "madebyollin/sdxl-vae-fp16-fix",
    torch_dtype=torch.float16,
)

# Download RealvisXl-v2.0
pipe = DiffusionPipeline.from_pretrained(
    MODEL_NAME,
    vae=better_vae,
    torch_dtype=torch.float16,
    use_safetensors=True,
)

pipe.load_lora_weights(COLORING_BOOK_MODEL_NAME, weight_name=COLORING_BOOK_WEIGHTS_NAME, adapter_name="coloringbook")
pipe.set_adapters("coloringbook", adapter_weights=1.0, adapter_names=["coloringbook"])
# First, fuse the LoRA parameters.
pipe.fuse_lora()

pipe.save_pretrained(MODEL_CACHE, safe_serialization=True).to("cuda")
