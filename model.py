import torch
import gc
import clip
import streamlit as st
from diffusers import StableDiffusionPipeline
from transformers import BlipProcessor, BlipForConditionalGeneration
from huggingface_hub import login
from config import HF_TOKEN, DEVICE

class ModelManager:

    def __init__(self):
        self.device = DEVICE
        self.current_model = None
        self._authenticate()
        torch.set_num_threads(4)  # Use 4 CPU threads

    def _authenticate(self):
        if not HF_TOKEN:
            raise ValueError("HUGGINGFACE_TOKEN environment variable is not set")
        try:
            login(HF_TOKEN)
        except Exception as e:
            raise ValueError(f"Authentication failed: {str(e)}")

    @st.cache_resource
    def load_clip_model(_self):
        with st.spinner("Loading CLIP model..."):
            model, preprocess = clip.load("ViT-B/32", device=_self.device)
            return model, preprocess

    @st.cache_resource
    def load_blip_model(_self):
        with st.spinner("Loading BLIP model..."):
            processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
            model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")
            model = model.to(_self.device)
            return model, processor

    def load_sd_1_5(self):
        with st.spinner("Loading Stable Diffusion 1.5..."):
            self.unload_models()
            pipe = StableDiffusionPipeline.from_pretrained(
                "runwayml/stable-diffusion-v1-5",
                torch_dtype=torch.float32,
                low_cpu_mem_usage=True
            )
            pipe = pipe.to(self.device)
            self.current_model = "SD_1_5"
            return pipe

    def load_sd_2_1(self):
        with st.spinner("Loading Stable Diffusion 2.1..."):
            self.unload_models()
            pipe = StableDiffusionPipeline.from_pretrained(
                "stabilityai/stable-diffusion-2-1",
                torch_dtype=torch.float32,
                low_cpu_mem_usage=True
            )
            pipe = pipe.to(self.device)
            self.current_model = "SD_2_1"
            return pipe

    def unload_models(self):
        gc.collect()
        self.current_model = None

    def get_memory_info(self):
        return "Running on CPU" 