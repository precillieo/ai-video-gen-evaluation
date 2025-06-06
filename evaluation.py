import torch
import cv2
from PIL import Image
import numpy as np
import clip
import streamlit as st
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr
import lpips

class Evaluator:
    
    def __init__(self, model_manager):
        self.model_manager = model_manager
        self.device = model_manager.device
        # Pre-load models to cache them
        self._clip_model, self._clip_preprocess = None, None
        self._blip_model, self._blip_processor = None, None
        self._lpips_model = None


    def _ensure_clip_loaded(self):
        if self._clip_model is None or self._clip_preprocess is None:
            self._clip_model, self._clip_preprocess = self.model_manager.load_clip_model()
        return self._clip_model, self._clip_preprocess


    def _ensure_blip_loaded(self):
        if self._blip_model is None or self._blip_processor is None:
            self._blip_model, self._blip_processor = self.model_manager.load_blip_model()
        return self._blip_model, self._blip_processor


    def _ensure_lpips_loaded(self):
        if self._lpips_model is None:
            self._lpips_model = lpips.LPIPS(net='alex').to(self.device)
        return self._lpips_model


    def generate_blip_caption(self, image):
        with st.spinner("Loading BLIP model..."):
            model, processor = self._ensure_blip_loaded()
        with st.spinner("Generating caption..."):
            inputs = processor(image, return_tensors="pt").to(self.device)
            with torch.no_grad():
                out = model.generate(**inputs, max_length=30)
            return processor.decode(out[0], skip_special_tokens=True)


    def extract_frames(self, video_path, num_frames=8):
        """Extract frames from video file.
        
        Args:
            video_path: Path to video file
            num_frames: Number of frames to extract
        
        Returns:
            List of PIL Images
        """
        with st.spinner("Extracting video frames..."):
            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                raise ValueError(f"Could not open video file: {video_path}")
            
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            if total_frames == 0:
                raise ValueError(f"Video file has no frames: {video_path}")
            
            # Calculate frame interval
            interval = max(total_frames // num_frames, 1)
            frames = []
            
            try:
                for i in range(min(num_frames, total_frames)):
                    cap.set(cv2.CAP_PROP_POS_FRAMES, i * interval)
                    ret, frame = cap.read()
                    if ret:
                        # Convert BGR to RGB
                        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                        # Convert to PIL Image
                        frame = Image.fromarray(frame)
                        frames.append(frame)
            finally:
                cap.release()
            
            if not frames:
                raise ValueError(f"No frames were extracted from video: {video_path}")
            
            return frames


    def compute_clip_score(self, prompt_or_image, images):
        with st.spinner("Loading CLIP model..."):
            model, preprocess = self._ensure_clip_loaded()
        
        # Convert prompt to text embedding
        if isinstance(prompt_or_image, str):
            text = clip.tokenize([prompt_or_image]).to(self.device)
            with torch.no_grad():
                text_features = model.encode_text(text)
        else:
            # If prompt_or_image is a reference image
            if isinstance(prompt_or_image, list):
                prompt_or_image = prompt_or_image[0]  # Take first image if list
            image = preprocess(prompt_or_image).unsqueeze(0).to(self.device)
            with torch.no_grad():
                text_features = model.encode_image(image)

        # Process images
        if not isinstance(images, list):
            images = [images]
        
        # Process each image individually to avoid dimension issues
        image_features = []
        for img in images:
            if isinstance(img, Image.Image):
                # Process single image
                processed = preprocess(img).unsqueeze(0).to(self.device)
                with torch.no_grad():
                    features = model.encode_image(processed)
                    image_features.append(features)
        
        # Stack all features
        image_features = torch.cat(image_features, dim=0)
        
        # Normalize features
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        
        # Compute similarity
        similarity = (100.0 * text_features @ image_features.T).mean()
        
        return float(similarity.cpu().numpy())


    def compute_motion_metrics(self, frames):
        """Compute motion-related metrics for video frames."""
        metrics = {}
        
        # Convert frames to numpy arrays
        frame_arrays = [np.array(frame) for frame in frames]
        
        # Compute optical flow between consecutive frames
        flows = []
        for i in range(len(frame_arrays) - 1):
            prev_gray = cv2.cvtColor(frame_arrays[i], cv2.COLOR_RGB2GRAY)
            curr_gray = cv2.cvtColor(frame_arrays[i + 1], cv2.COLOR_RGB2GRAY)
            flow = cv2.calcOpticalFlowFarneback(prev_gray, curr_gray, None, 0.5, 3, 15, 3, 5, 1.2, 0)
            flows.append(flow)
        
        # Motion magnitude
        motion_magnitudes = [np.sqrt(flow[..., 0]**2 + flow[..., 1]**2).mean() for flow in flows]
        metrics['motion_magnitude'] = np.mean(motion_magnitudes)
        
        # Motion smoothness (variance of motion magnitude)
        metrics['motion_smoothness'] = np.std(motion_magnitudes)
        
        # Temporal consistency (correlation between consecutive flows)
        if len(flows) > 1:
            correlations = []
            for i in range(len(flows) - 1):
                corr = np.corrcoef(flows[i].flatten(), flows[i + 1].flatten())[0, 1]
                correlations.append(corr)
            metrics['temporal_consistency'] = np.mean(correlations)
        
        return metrics


    def compute_visual_quality(self, image, reference_image=None):
        """Compute visual quality metrics for an image."""
        metrics = {}
        
        # Ensure images are RGB
        if image.mode != 'RGB':
            image = image.convert('RGB')
        if reference_image is not None and reference_image.mode != 'RGB':
            reference_image = reference_image.convert('RGB')
        
        # If reference image is provided, resize image to match reference
        if reference_image is not None:
            if image.size != reference_image.size:
                image = image.resize(reference_image.size, Image.BICUBIC)
        
        # Convert to numpy array
        img_array = np.array(image)
        
        # Blur detection using Laplacian variance
        gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
        laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
        metrics['blur_score'] = laplacian_var
        
        # Color consistency (variance of color channels)
        metrics['color_consistency'] = np.std(img_array, axis=(0, 1)).mean()
        
        # Compute LPIPS if reference image is provided
        if reference_image is not None:
            lpips_model = self._ensure_lpips_loaded()
            img_tensor = torch.from_numpy(img_array).permute(2, 0, 1).float() / 255.0
            ref_tensor = torch.from_numpy(np.array(reference_image)).permute(2, 0, 1).float() / 255.0
            
            img_tensor = img_tensor.unsqueeze(0).to(self.device)
            ref_tensor = ref_tensor.unsqueeze(0).to(self.device)
            
            with torch.no_grad():
                metrics['lpips_score'] = float(lpips_model(img_tensor, ref_tensor).cpu().numpy())
            
            # Compute SSIM and PSNR robustly
            min_dim = min(img_array.shape[0], img_array.shape[1])
            if min_dim < 7:
                metrics['ssim'] = None
                metrics['psnr'] = None
            else:
                win_size = min(7, min_dim) if min_dim % 2 == 1 else min(7, min_dim - 1)
                metrics['ssim'] = ssim(
                    img_array, np.array(reference_image),
                    channel_axis=-1, win_size=win_size
                )
                metrics['psnr'] = psnr(img_array, np.array(reference_image))
        
        return metrics


    def evaluate_prompt_alignment(self, prompt, image):
        """Evaluate how well the image follows the prompt instructions."""
        metrics = {}
        
        # Get BLIP caption
        blip_caption = self.generate_blip_caption(image)
        metrics['blip_caption'] = blip_caption
        
        # Compute CLIP score
        clip_score = self.compute_clip_score(prompt, image)
        metrics['clip_score'] = clip_score
        
        # TODO: Add more sophisticated prompt alignment metrics
        # - Object detection and counting
        # - Spatial relationship analysis
        # - Style matching
        
        return metrics


    def text_to_image(self, model_name, prompt, guidance=7.5, steps=30):
        try:
            with st.spinner(f"Loading {model_name} model..."):
                pipe = (self.model_manager.load_sd_1_5() 
                       if model_name == "Stable Diffusion 1.5" 
                       else self.model_manager.load_sd_2_1())
            
            with st.spinner(f"Generating image with {model_name}..."):
                generator = torch.Generator(device=self.device).manual_seed(42)
                image = pipe(
                    prompt,
                    guidance_scale=guidance,
                    num_inference_steps=steps,
                    generator=generator
                ).images[0]
                
                if self.device == "cuda":
                    pipe.to("cpu")
                    torch.cuda.empty_cache()
                
                return image
        except Exception as e:
            raise ValueError(f"Error generating image: {str(e)}") 