import streamlit as st
import torch
import os
import gc
import numpy as np
import tempfile
import time
from PIL import Image
import matplotlib.pyplot as plt
import io
import clip
from decord import VideoReader, cpu
# import cv2
from diffusers import StableDiffusionPipeline, FluxPipeline
from transformers import BlipProcessor, BlipForConditionalGeneration
import torch.nn.functional as F

# Set page config
st.set_page_config(page_title="AI Video/Image Gen Evaluation", layout="wide")

# Initialize session state
if 'device' not in st.session_state:
    st.session_state.device = "cuda" if torch.cuda.is_available() else "cpu"
    st.session_state.current_model = None

# Cache CLIP and BLIP models (smaller models for evaluation)
@st.cache_resource
def load_clip_model():
    device = st.session_state.device
    model, preprocess = clip.load("ViT-B/32", device=device)
    return model, preprocess

@st.cache_resource
def load_blip_model():
    processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
    model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")
    model = model.to(st.session_state.device)
    return model, processor

# Model loading functions WITHOUT caching (for generation models)
def load_sd_1_5():
    device = st.session_state.device
    model_id = "runwayml/stable-diffusion-v1-5"
    unload_models()  # Unload any previous model
    pipe = StableDiffusionPipeline.from_pretrained(
        model_id, 
        torch_dtype=torch.float16 if device == "cuda" else torch.float32,
        low_cpu_mem_usage=True
    )
    pipe = pipe.to(device)
    st.session_state.current_model = "SD_1_5"
    return pipe

def load_sd_2_1():
    device = st.session_state.device
    model_id = "stabilityai/stable-diffusion-2-1"
    unload_models()  # Unload any previous model
    pipe = StableDiffusionPipeline.from_pretrained(
        model_id, 
        torch_dtype=torch.float16 if device == "cuda" else torch.float32,
        low_cpu_mem_usage=True
    )
    pipe = pipe.to(device)
    st.session_state.current_model = "SD_2_1"
    return pipe

def load_flux():
    device = st.session_state.device
    unload_models()  # Unload any previous model
    pipe = FluxPipeline.from_pretrained(
        "black-forest-labs/FLUX.1-dev", 
        torch_dtype=torch.float16 if device == "cuda" else torch.float32,
        low_cpu_mem_usage=True
    )
    pipe = pipe.to(device)
    st.session_state.current_model = "FLUX"
    return pipe

# Unload models and clear CUDA cache
def unload_models():
    # Clear CUDA cache
    if st.session_state.device == "cuda":
        torch.cuda.empty_cache()
    # Force garbage collection
    gc.collect()
    st.session_state.current_model = None

# Utility functions
def generate_blip_caption(image):
    model, processor = load_blip_model()
    inputs = processor(image, return_tensors="pt").to(st.session_state.device)
    with torch.no_grad():
        out = model.generate(**inputs, max_length=30)
    caption = processor.decode(out[0], skip_special_tokens=True)
    return caption

def extract_frames(video_path, num_frames=8):
    vr = VideoReader(video_path, ctx=cpu(0))
    total_frames = len(vr)
    interval = max(total_frames // num_frames, 1)
    frame_indices = [i * interval for i in range(min(num_frames, total_frames))]
    frames = vr.get_batch(frame_indices).asnumpy()
    images = [Image.fromarray(frame) for frame in frames]
    return images

def compute_clip_score(prompt_text, images):
    model_clip, preprocess_clip = load_clip_model()
    
    with torch.no_grad():
        text_tokens = clip.tokenize([prompt_text]).to(st.session_state.device)
        text_emb = model_clip.encode_text(text_tokens)

        # Process images
        if isinstance(images, list):
            # For video frames
            frame_embs = []
            for frame in images:
                processed = preprocess_clip(frame).unsqueeze(0).to(st.session_state.device)
                frame_embs.append(model_clip.encode_image(processed))
            image_emb = torch.cat(frame_embs, dim=0).mean(dim=0, keepdim=True)
        else:
            # For single image
            processed = preprocess_clip(images).unsqueeze(0).to(st.session_state.device)
            image_emb = model_clip.encode_image(processed)

        # Normalize embeddings
        text_emb = text_emb / text_emb.norm(dim=-1, keepdim=True)
        image_emb = image_emb / image_emb.norm(dim=-1, keepdim=True)

        # Calculate similarity
        similarity = (image_emb @ text_emb.T).item() * 100  # Convert to percentage
    
    return similarity

def text_to_image(model_name, prompt, guidance=7.5, steps=30):
    with st.spinner(f"Loading {model_name} model..."):
        if model_name == "Stable Diffusion 1.5":
            pipe = load_sd_1_5()
        elif model_name == "Stable Diffusion 2.1":
            pipe = load_sd_2_1()
        elif model_name == "FLUX AI":
            pipe = load_flux()
    
    with st.spinner(f"Generating image with {model_name}..."):
        # Set seed for reproducibility
        generator = torch.Generator(device=st.session_state.device).manual_seed(42)
        
        if model_name == "FLUX AI":
            image = pipe(
                prompt,
                height=512,
                width=512,
                guidance_scale=guidance,
                num_inference_steps=steps,
                generator=generator
            ).images[0]
        else:
            image = pipe(
                prompt, 
                guidance_scale=guidance, 
                num_inference_steps=steps,
                generator=generator
            ).images[0]
        
        # Move model to CPU and clear GPU memory after generation
        if st.session_state.device == "cuda":
            pipe.to("cpu")
            torch.cuda.empty_cache()
    
    return image

# Main app layout
st.title("AI Video/Image Gen Evaluation")

# Sidebar for model selection
st.sidebar.header("Model Settings")
evaluation_type = st.sidebar.selectbox(
    "Select Generation Type",
    ["Text to Image", "Text to Video", "Image to Video"]
)

# Memory management info
memory_info = ""
if st.session_state.device == "cuda":
    allocated = torch.cuda.memory_allocated() / (1024 ** 3)
    reserved = torch.cuda.memory_reserved() / (1024 ** 3)
    memory_info = f"GPU Memory: {allocated:.2f}GB allocated, {reserved:.2f}GB reserved"

st.sidebar.info(memory_info if memory_info else "Running on CPU")

# Add a button to manually clear memory
if st.sidebar.button("Clear Memory"):
    unload_models()
    st.sidebar.success("Memory cleared!")

# Main content area
if evaluation_type == "Text to Image":
    st.header("Text to Image Generation")
    
    # Input prompt
    prompt = st.text_area("Enter your prompt:", "A photo of a cute cat sitting on a wooden chair")
    
    # Model parameters
    col1, col2 = st.columns(2)
    with col1:
        guidance_scale = st.slider("Guidance Scale", 1.0, 20.0, 7.5, 0.5)
    with col2:
        steps = st.slider("Inference Steps", 10, 100, 30, 5)
    
    # Models to evaluate
    t2i_models = ["Stable Diffusion 1.5", "Stable Diffusion 2.1", "FLUX AI"]
    
    # Option to select specific models
    selected_models = st.multiselect(
        "Select models to evaluate (default: all)",
        t2i_models,
        default=t2i_models
    )
    
    if not selected_models:
        st.warning("Please select at least one model to evaluate")
    elif st.button("Generate and Evaluate"):
        # Set up the display area
        results_cols = st.columns(len(selected_models))
        metrics_cols = st.columns(len(selected_models))
        
        results = {}
        
        # Generate images sequentially for each model and calculate metrics
        for i, model_name in enumerate(selected_models):
            with results_cols[i]:
                st.subheader(model_name)
                image = text_to_image(model_name, prompt, guidance_scale, steps)
                st.image(image, use_container_width=True)
                results[model_name] = {"image": image}
                
                # Calculate CLIP Score
                clip_score = compute_clip_score(prompt, image)
                st.metric("CLIP Score", f"{clip_score:.2f}%")
                
                # BLIP Caption
                blip_caption = generate_blip_caption(image)
                st.text_area("BLIP Caption", blip_caption, height=100)
                
                results[model_name]["clip_score"] = clip_score
                results[model_name]["blip_caption"] = blip_caption
            
            # Explicitly unload the model after each use
            unload_models()
        
        # Display comparison chart
        st.subheader("Model Comparison - CLIP Score")
        chart_data = {model: results[model]["clip_score"] for model in selected_models}
        
        fig, ax = plt.subplots(figsize=(10, 5))
        colors = ['#FF9999', '#66B2FF', '#99FF99']
        bars = ax.bar(chart_data.keys(), chart_data.values(), color=colors[:len(chart_data)])
        ax.set_ylim(0, 100)
        ax.set_ylabel('CLIP Score (%)')
        ax.set_title('Text-to-Image Model Comparison')
        
        # Add value labels on top of each bar
        for bar in bars:
            height = bar.get_height()
            ax.annotate(f'{height:.1f}',
                        xy=(bar.get_x() + bar.get_width() / 2, height),
                        xytext=(0, 3),  # 3 points vertical offset
                        textcoords="offset points",
                        ha='center', va='bottom')
        
        # Display the chart
        st.pyplot(fig)
        
        # Final memory cleanup
        unload_models()

elif evaluation_type == "Text to Video":
    st.header("Text to Video Generation")
    st.warning("This feature is under development. For now, you can upload videos and evaluate them.")
    
    # Input prompt
    prompt = st.text_area("Enter your prompt:", "A cat walking in a garden")
    
    # Video upload
    uploaded_videos = st.file_uploader("Upload videos to evaluate", type=["mp4", "avi", "mov"], accept_multiple_files=True)
    
    if uploaded_videos and st.button("Evaluate Videos"):
        if len(uploaded_videos) > 3:
            st.warning("Please upload a maximum of 3 videos for comparison")
            uploaded_videos = uploaded_videos[:3]
        
        # Set up columns for each video
        video_cols = st.columns(len(uploaded_videos))
        metrics_cols = st.columns(len(uploaded_videos))
        
        results = {}
        
        # Process each uploaded video
        for i, video_file in enumerate(uploaded_videos):
            with video_cols[i]:
                # Save the uploaded video to a temporary file
                temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4')
                temp_file.write(video_file.read())
                temp_file.close()
                
                st.subheader(f"Video {i+1}: {video_file.name}")
                st.video(temp_file.name)
                
                # Extract frames
                frames = extract_frames(temp_file.name)
                results[video_file.name] = {"frames": frames, "path": temp_file.name}
        
        # Calculate and display metrics
        for i, video_name in enumerate(results.keys()):
            with metrics_cols[i]:
                frames = results[video_name]["frames"]
                
                # Display first frame
                st.image(frames[0], caption="First frame", use_container_width=True)
                
                # CLIP Score
                clip_score = compute_clip_score(prompt, frames)
                st.metric("CLIP Score", f"{clip_score:.2f}%")
                
                # BLIP Caption for first frame
                blip_caption = generate_blip_caption(frames[0])
                st.text_area("BLIP Caption (first frame)", blip_caption, height=100)
                
                results[video_name]["clip_score"] = clip_score
                results[video_name]["blip_caption"] = blip_caption
                
                # Clear GPU memory after processing each video
                if st.session_state.device == "cuda":
                    torch.cuda.empty_cache()
        
        # Display comparison chart
        st.subheader("Video Comparison - CLIP Score")
        chart_data = {name: results[name]["clip_score"] for name in results.keys()}
        
        fig, ax = plt.subplots(figsize=(10, 5))
        bars = ax.bar(chart_data.keys(), chart_data.values(), color=['#FF9999', '#66B2FF', '#99FF99'][:len(chart_data)])
        ax.set_ylim(0, 100)
        ax.set_ylabel('CLIP Score (%)')
        ax.set_title('Text-to-Video Evaluation')
        
        # Add value labels on top of each bar
        for bar in bars:
            height = bar.get_height()
            ax.annotate(f'{height:.1f}',
                        xy=(bar.get_x() + bar.get_width() / 2, height),
                        xytext=(0, 3),  # 3 points vertical offset
                        textcoords="offset points",
                        ha='center', va='bottom')
        
        # Clean up temp files
        for video_name in results:
            os.unlink(results[video_name]["path"])
            
        # Display the chart
        st.pyplot(fig)
        
        # Final memory cleanup
        unload_models()

elif evaluation_type == "Image to Video":
    st.header("Image to Video Generation")
    st.warning("This feature is under development. For now, you can upload a reference image and evaluate videos against it.")
    
    # Image upload
    reference_image = st.file_uploader("Upload reference image", type=["jpg", "jpeg", "png"])
    
    # Video upload
    uploaded_videos = st.file_uploader("Upload videos to evaluate", type=["mp4", "avi", "mov"], accept_multiple_files=True)
    
    if reference_image and uploaded_videos and st.button("Evaluate Videos"):
        # Display reference image
        ref_img = Image.open(reference_image)
        st.image(ref_img, caption="Reference Image", width=300)
        
        if len(uploaded_videos) > 3:
            st.warning("Please upload a maximum of 3 videos for comparison")
            uploaded_videos = uploaded_videos[:3]
        
        # Set up columns for each video
        video_cols = st.columns(len(uploaded_videos))
        metrics_cols = st.columns(len(uploaded_videos))
        
        results = {}
        
        # Process each uploaded video
        for i, video_file in enumerate(uploaded_videos):
            with video_cols[i]:
                # Save the uploaded video to a temporary file
                temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4')
                temp_file.write(video_file.read())
                temp_file.close()
                
                st.subheader(f"Video {i+1}: {video_file.name}")
                st.video(temp_file.name)
                
                # Extract frames
                frames = extract_frames(temp_file.name)
                results[video_file.name] = {"frames": frames, "path": temp_file.name}
        
        # Calculate and display metrics
        for i, video_name in enumerate(results.keys()):
            with metrics_cols[i]:
                frames = results[video_name]["frames"]
                
                # Display first frame
                st.image(frames[0], caption="First frame", use_container_width=True)
                
                # Calculate similarity using CLIP
                model_clip, preprocess_clip = load_clip_model()
                
                with torch.no_grad():
                    # Process reference image
                    ref_processed = preprocess_clip(ref_img).unsqueeze(0).to(st.session_state.device)
                    ref_emb = model_clip.encode_image(ref_processed)
                    
                    # Process video frames
                    frame_embs = []
                    for frame in frames:
                        processed = preprocess_clip(frame).unsqueeze(0).to(st.session_state.device)
                        frame_embs.append(model_clip.encode_image(processed))
                    
                    frame_emb = torch.cat(frame_embs, dim=0).mean(dim=0, keepdim=True)
                    
                    # Normalize embeddings
                    ref_emb = ref_emb / ref_emb.norm(dim=-1, keepdim=True)
                    frame_emb = frame_emb / frame_emb.norm(dim=-1, keepdim=True)
                    
                    # Calculate similarity
                    similarity = (frame_emb @ ref_emb.T).item() * 100  # Convert to percentage
                
                st.metric("Image-Video Similarity", f"{similarity:.2f}%")
                results[video_name]["similarity"] = similarity
                
                # Clear GPU memory after each video
                if st.session_state.device == "cuda":
                    torch.cuda.empty_cache()
        
        # Display comparison chart
        st.subheader("Video Comparison - Image Similarity")
        chart_data = {name: results[name]["similarity"] for name in results.keys()}
        
        fig, ax = plt.subplots(figsize=(10, 5))
        bars = ax.bar(chart_data.keys(), chart_data.values(), color=['#FF9999', '#66B2FF', '#99FF99'][:len(chart_data)])
        ax.set_ylim(0, 100)
        ax.set_ylabel('Similarity Score (%)')
        ax.set_title('Image-to-Video Evaluation')
        
        # Add value labels on top of each bar
        for bar in bars:
            height = bar.get_height()
            ax.annotate(f'{height:.1f}',
                        xy=(bar.get_x() + bar.get_width() / 2, height),
                        xytext=(0, 3),  # 3 points vertical offset
                        textcoords="offset points",
                        ha='center', va='bottom')
        
        # Clean up temp files
        for video_name in results:
            os.unlink(results[video_name]["path"])
            
        # Display the chart
        st.pyplot(fig)
        
        # Final memory cleanup
        unload_models()

# Add footer
st.markdown("---")
st.markdown("AI Video/Image Gen Evaluation | Created with Streamlit")