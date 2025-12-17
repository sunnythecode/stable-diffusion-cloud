import torch
from diffusers import StableDiffusionPipeline
from PIL import Image
import os

# Configuration
GPU_ID = 1  # Change to your desired GPU (0, 1, 2, etc.)
MODEL_ID = "Manojb/stable-diffusion-2-1-base"
OUTPUT_DIR = "tmp"

# Text prompts for batch generation
prompts = [
    "a serene mountain landscape at sunset, photorealistic",
    "a futuristic city with flying cars, cyberpunk style",
]

# Generation parameters
NUM_INFERENCE_STEPS = 50
GUIDANCE_SCALE = 7.5
HEIGHT = 512
WIDTH = 512
SEED = 42  # Set to None for random seeds

def setup_pipeline(gpu_id):
    """Initialize the Stable Diffusion pipeline on specified GPU"""
    device = f"cuda:{gpu_id}" if torch.cuda.is_available() else "cpu"
    
    if device == "cpu":
        print("Warning: CUDA not available, using CPU (will be slow)")
    else:
        print(f"Using GPU: {torch.cuda.get_device_name(gpu_id)}")
    
    # Load pipeline with fp16 for better performance
    pipe = StableDiffusionPipeline.from_pretrained(
        MODEL_ID,
        torch_dtype=torch.float16 if device != "cpu" else torch.float32,
        safety_checker=None,  # Remove if you want safety checking
    )
    pipe = pipe.to(device)
    
    # Enable memory optimizations
    pipe.enable_attention_slicing()
    
    # Optional: Enable xformers for even better performance
    # Uncomment if you have xformers installed
    # pipe.enable_xformers_memory_efficient_attention()
    
    return pipe, device

def generate_images(pipe, prompts, output_dir):
    """Generate images for a batch of prompts"""
    os.makedirs(output_dir, exist_ok=True)
    
    # Set seed for reproducibility
    generator = torch.Generator(device=pipe.device)
    if SEED is not None:
        generator.manual_seed(SEED)
    
    for idx, prompt in enumerate(prompts):
        print(f"\nGenerating image {idx + 1}/{len(prompts)}")
        print(f"Prompt: {prompt}")
        
        # Generate image
        with torch.autocast(pipe.device.type):
            image = pipe(
                prompt=prompt,
                height=HEIGHT,
                width=WIDTH,
                num_inference_steps=NUM_INFERENCE_STEPS,
                guidance_scale=GUIDANCE_SCALE,
                generator=generator
            ).images[0]
        
        # Save image
        output_path = os.path.join(output_dir, f"image_{idx:03d}.png")
        image.save(output_path)
        print(f"Saved: {output_path}")
    
    print(f"\nAll images saved to {output_dir}/")

def main():
    # Setup pipeline
    print("Loading Stable Diffusion 2.1 model...")
    pipe, device = setup_pipeline(GPU_ID)
    
    # Generate images
    print(f"\nGenerating {len(prompts)} images...")
    generate_images(pipe, prompts, OUTPUT_DIR)
    
    print("\nDone!")

if __name__ == "__main__":
    main()