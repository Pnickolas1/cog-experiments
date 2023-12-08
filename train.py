import os
os.environ['HF_HUB_ENABLE_HF_TRANSFER'] = '1'
import shutil
from huggingface_hub import hf_hub_download
from cog import BaseModel, Input, Path

from predict import SDXL_MODEL_CACHE, SDXL_URL, download_weights
OUTPUT_DIR = "training_out"

class TrainingOutput(BaseModel):
    weights_a: Path
    weights_b: Path

def train(
        repo_id_a: str = Input(
            description="First Hugging Face repo to import",
        ),
        weights_a: str = Input(
            description="First safetensor weights to import",
        ),
        repo_id_b: str = Input(
            description="Second Hugging Face repo to import",
        ),
        weights_b: str = Input(
            description="Second safetensor weights to import",
        )
) -> TrainingOutput:

    dest_a = 'lora_a.safetensors'
    dest_b = 'lora_b.safetensors'

    # Download and save first set of weights
    if os.path.exists(dest_a):
        os.remove(dest_a)
    fn_a = hf_hub_download(repo_id=repo_id_a, filename=weights_a)
    shutil.copy(fn_a, dest_a)
    os.remove(fn_a)

    # Download and save second set of weights
    if os.path.exists(dest_b):
        os.remove(dest_b)
    fn_b = hf_hub_download(repo_id=repo_id_b, filename=weights_b)
    shutil.copy(fn_b, dest_b)
    os.remove(fn_b)

    print(f"Weights copied to {dest_a} and {dest_b}")
    return TrainingOutput(weights_a=Path(dest_a), weights_b=Path(dest_b))