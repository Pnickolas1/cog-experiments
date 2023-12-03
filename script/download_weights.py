

# Run this before you deploy it on replicate, because if you don't
# whenever you run the model, it will download the weights from the
# internet, which will take a long time.

import torch
from diffusers import AutoencoderKL, DiffusionPipeline
from diffusers.pipelines.stable_diffusion.safety_checker import (
    StableDiffusionSafetyChecker,
)
import peft
from predict import MODEL_NAME, MODEL_CACHE, COLORING_BOOK_MODEL_NAME, COLORING_BOOK_WEIGHTS_NAME


# # Download RealvisXl-v2.0
# pipe = DiffusionPipeline.from_pretrained(
#     MODEL_NAME,
#     vae=better_vae,
#     torch_dtype=torch.float16,
#     use_safetensors=True,
# )

# pipe = DiffusionPipeline.from_pretrained(
#     "stabilityai/stable-diffusion-xl-base-1.0",
#     torch_dtype=torch.float16,
#     use_safetensors=True,
#     variant="fp16",
# )

# pipe.save_pretrained("./cache", safe_serialization=True)

better_vae = AutoencoderKL.from_pretrained(
    "madebyollin/sdxl-vae-fp16-fix", torch_dtype=torch.float16
)

pipe = DiffusionPipeline.from_pretrained(
    "stabilityai/stable-diffusion-xl-base-1.0",
    vae=better_vae,
    torch_dtype=torch.float16,
    use_safetensors=True,
    variant="fp16",
)

pipe.save_pretrained("./sdxl-cache", safe_serialization=True)

pipe = DiffusionPipeline.from_pretrained(
    "stabilityai/stable-diffusion-xl-refiner-1.0",
    torch_dtype=torch.float16,
    use_safetensors=True,
    variant="fp16",
)

# TODO - we don't need to save all of this and in fact should save just the unet, tokenizer, and config.
pipe.save_pretrained("./refiner-cache", safe_serialization=True)


safety = StableDiffusionSafetyChecker.from_pretrained(
    "CompVis/stable-diffusion-safety-checker",
    torch_dtype=torch.float16,
)

# pipe.load_lora_weights(COLORING_BOOK_MODEL_NAME, weight_name=COLORING_BOOK_WEIGHTS_NAME, adapter_name="coloringbook")
# pipe.set_adapters("coloringbook")
# #pipe.set_adapters("coloringbook", adapter_weights=1.0, adapter_names=["coloringbook"])  --> did not work
# #pipe.set_adapters(adapter_weights=1.0, adapter_names=["coloringbook"]) 
# # First, fuse the LoRA parameters.
# pipe.fuse_lora()

pipe.save_pretrained(MODEL_CACHE, safe_serialization=True)


safety.save_pretrained("./safety-cache")
