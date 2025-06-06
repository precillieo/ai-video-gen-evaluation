import os
from dotenv import load_dotenv
import torch

load_dotenv()

HF_TOKEN = os.getenv('HUGGINGFACE_TOKEN')

DEVICE = "cpu"

MODEL_SETTINGS = {
    "torch_dtype": torch.float32,
    "low_cpu_mem_usage": True
} 