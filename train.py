import os
os.environ['HF_HUB_ENABLE_HF_TRANSFER'] = '1'
import shutil
from huggingface_hub import hf_hub_download
from cog import BaseModel, Input, Path

from predict import SDXL_MODEL_CACHE, SDXL_URL, download_weights
OUTPUT_DIR = "training_out"

class TrainingOutput(BaseModel):
    weights: Path

def train(
        repo_id: str = Input(
            description="huggingface repo to import ",
        ),
        weights: str = Input(
            description="safetensor weights to import ",
        ),
) -> TrainingOutput:

    dest = 'lora.safetensors'
    if os.path.exists(dest):
        os.remove(dest)
    
    fn = hf_hub_download(repo_id=repo_id, filename=weights)

    shutil.copy(fn, dest)
    os.remove(fn)
    print(f"weights copied to {dest}")
    return TrainingOutput(weights=Path(dest))