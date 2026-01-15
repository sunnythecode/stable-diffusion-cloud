import argparse
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import torch
import time
from dataclasses import dataclass
from typing import Optional
from diffusers import (
    StableDiffusionPipeline,
    DDIMScheduler,
    PNDMScheduler,
    LMSDiscreteScheduler,
    EulerDiscreteScheduler,
    EulerAncestralDiscreteScheduler,
    DPMSolverMultistepScheduler,
    UniPCMultistepScheduler,
)
from contextlib import contextmanager
from stable_diff_trt import profile_stable_diffusion, test_n_stable_diffusion
from pprint import pprint
import torch
from PIL import Image
from transformers import CLIPProcessor, CLIPModel

def compute_clip_score(image_path, text, device=None):
    """
    image_path: path to a single image
    text: a single text string
    device: 'cuda' or 'cpu' (optional)
    """
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")

    # Load the CLIP model
    model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32", torch_dtype=torch.float16, use_safetensors=True).to(device)
    processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

    # Load image
    image = Image.open(image_path).convert("RGB")

    # Preprocess
    inputs = processor(text=[text], images=image, return_tensors="pt", padding=True).to(device)

    # Forward pass
    with torch.no_grad():
        image_features = model.get_image_features(pixel_values=inputs["pixel_values"])
        text_features = model.get_text_features(input_ids=inputs["input_ids"], attention_mask=inputs["attention_mask"])

    # Normalize features
    image_features /= image_features.norm(p=2, dim=-1, keepdim=True)
    text_features /= text_features.norm(p=2, dim=-1, keepdim=True)

    # Compute cosine similarity
    clip_score = (image_features @ text_features.T).item()
    return clip_score


# TensorRT imports
try:
    import tensorrt as trt
    TENSORRT_AVAILABLE = True
except ImportError:
    TENSORRT_AVAILABLE = False
    print("Warning: TensorRT not available. Install with: pip install tensorrt")


@dataclass
class PromptConfig:
    """Configuration for Stable Diffusion generation."""
    
    # Core settings
    prompt: str
    negative_prompt: str = ""
    use_negative_prompt: bool = True
    
    # Generation parameters
    num_inference_steps: int = 50
    guidance_scale: float = 7.5
    seed: int = 42
    
    # Image dimensions (must be multiples of 8)
    height: int = 512
    width: int = 512
    
    # Performance settings
    use_fp16: bool = True
    scheduler_name: str = "default"
    
    # TensorRT settings
    use_tensorrt: bool = False
    tensorrt_engine_dir: str = "./tensorrt_engines"
    tensorrt_build_on_first_run: bool = True
    
    # Output
    output_filename: str = "generated_image.png"
    
    def __post_init__(self):
        """Validate configuration."""
        if self.height % 8 != 0 or self.width % 8 != 0:
            raise ValueError("Height and width must be multiples of 8")
        if self.num_inference_steps < 1:
            raise ValueError("num_inference_steps must be at least 1")
        if self.guidance_scale < 0:
            raise ValueError("guidance_scale must be non-negative")
        if self.use_tensorrt and not TENSORRT_AVAILABLE:
            raise RuntimeError("TensorRT requested but not available. Install with: pip install tensorrt")




if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate images with Stable Diffusion")
    
    parser.add_argument("--prompt", type=str, required=True, help="Text prompt for image generation")
    parser.add_argument("--steps", type=int, default=50, help="Number of inference steps (default: 50)")
    parser.add_argument("--n_runs", type=int, default = 1, help="number of runs")
    parser.add_argument("--seed", type=int, default=42, help="Random seed (default: 42)")
    parser.add_argument("--output", type=str, default="generated_image.png", help="Output filename (default: generated_image.png)")
    parser.add_argument("--width", type=int, default=512, help="Image width in pixels (default: 512)")
    parser.add_argument("--height", type=int, default=512, help="Image height in pixels (default: 512)")
    parser.add_argument("--guidance_scale", type=float, default=7.5, help="Guidance scale (default: 7.5)")
    parser.add_argument("--no_negative_prompt", action="store_true", help="Disable negative prompt")
    parser.add_argument("--use_tensorrt", default=True,action="store_true", help="Enable TensorRT acceleration if available")
    
    args = parser.parse_args()
    
    config = PromptConfig(
        prompt=args.prompt,
        # use_negative_prompt=not args.no_negative_prompt,
        num_inference_steps=args.steps,
        seed=args.seed,
        scheduler_name="ddim",
        output_filename=args.output,
        width=args.width,
        height=args.height,
        guidance_scale=args.guidance_scale,
        use_tensorrt=args.use_tensorrt
    )
    
    print(config)
    
    # Run generation
    img, timings = test_n_stable_diffusion(config, args.n_runs)
    pprint(timings)
    print("CLIP Score:", compute_clip_score(args.output, args.prompt))
    print("\nImage generation complete!")
