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
    use_tensorrt: bool = True
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
    
    def __str__(self):
        """Pretty print configuration."""
        return f"""PromptConfig:
  Prompt: "{self.prompt[:60]}{'...' if len(self.prompt) > 60 else ''}"
  Resolution: {self.width}x{self.height}
  Steps: {self.num_inference_steps}
  Guidance Scale: {self.guidance_scale}
  Seed: {self.seed}
  Scheduler: {self.scheduler_name}
  Precision: {'fp16' if self.use_fp16 else 'fp32'}
  TensorRT: {'Enabled' if self.use_tensorrt else 'Disabled'}
  Output: {self.output_filename}"""

@contextmanager
def timer(name):
    """Context manager to time code blocks with GPU synchronization"""
    torch.cuda.synchronize()
    start = time.perf_counter()
    yield
    torch.cuda.synchronize()
    end = time.perf_counter()
    print(f"{name}: {end - start:.4f} seconds")
    return end - start

def count_parameters(model):
    """Count the number of parameters in a model"""
    return sum(p.numel() for p in model.parameters())

def load_tensorrt_pipeline(model_id, config):
    """
    Load Stable Diffusion pipeline with TensorRT optimizationf.
    
    Method 1: Using torch.compile with TensorRT backend (PyTorch 2.0+)
    This is the simplest approach.
    """
    print("\n" + "="*60)
    print("LOADING WITH TENSORRT OPTIMIZATION")
    print("="*60)
    
    dtype = torch.float16 if config.use_fp16 else torch.float32
    
    # Load base pipeline
    pipe = StableDiffusionPipeline.from_pretrained(
        model_id,
        torch_dtype=dtype,
    ).to("cuda")
    
    # Option 1: Use torch.compile with TensorRT backend (PyTorch 2.0+)
    if hasattr(torch, 'compile'):
        print("Using torch.compile with TensorRT backend...")
        try:
            # Compile UNet (the most compute-intensive component)
            pipe.unet = torch.compile(
                pipe.unet,
                backend="tensorrt",
                mode="max-autotune",
            )
            print("✓ UNet compiled with TensorRT")
            
            # Optionally compile VAE decoder
            pipe.vae.decoder = torch.compile(
                pipe.vae.decoder,
                backend="tensorrt",
                mode="max-autotune",
            )
            print("✓ VAE decoder compiled with TensorRT")
            
        except Exception as e:
            print(f"Warning: torch.compile with TensorRT failed: {e}")
            print("Falling back to standard pipeline")
    else:
        print("Warning: torch.compile not available (requires PyTorch 2.0+)")
    
    return pipe

def load_tensorrt_pipeline_manual(model_id, config):
    """
    Method 2: Manual TensorRT conversion using ONNX export.
    This gives more control but requires more setup.
    """
    from diffusers.pipelines.stable_diffusion import StableDiffusionPipelineOutput
    import onnx
    
    print("\n" + "="*60)
    print("MANUAL TENSORRT CONVERSION (Advanced)")
    print("="*60)
    
    # This is a placeholder for manual TensorRT conversion
    # Full implementation requires:
    # 1. Export models to ONNX
    # 2. Convert ONNX to TensorRT engines
    # 3. Wrap engines in custom pipeline
    
    print("Manual TensorRT conversion requires additional setup.")
    print("See: https://github.com/NVIDIA/TensorRT/tree/main/demo/Diffusion")
    
    # For now, fall back to torch.compile method
    return load_tensorrt_pipeline(model_id, config)

def profile_stable_diffusion(config):
    """
    Profile Stable Diffusion with configurable parameters and optional TensorRT.
    """
    timings = {}
    
    # Load the model
    print("Loading Stable Diffusion 2.1...")
    model_id = "sd2-community/stable-diffusion-2-1"
    
    dtype = torch.float16 if config.use_fp16 else torch.float32
    
    model_load_start = time.perf_counter()
    
    if config.use_tensorrt:
        pipe = load_tensorrt_pipeline(model_id, config)
    else:
        pipe = StableDiffusionPipeline.from_pretrained(
            model_id,
            torch_dtype=dtype,
        ).to("cuda")
    
    torch.cuda.synchronize()
    timings['model_load'] = time.perf_counter() - model_load_start
    
    # Configure scheduler
    scheduler_map = {
        "default": None,
        "ddim": DDIMScheduler,
        "pndm": PNDMScheduler,
        "lms": LMSDiscreteScheduler,
        "euler": EulerDiscreteScheduler,
        "euler_a": EulerAncestralDiscreteScheduler,
        "dpm": DPMSolverMultistepScheduler,
        "unipc": UniPCMultistepScheduler,
    }
    
    if config.scheduler_name != "default" and config.scheduler_name in scheduler_map:
        scheduler_class = scheduler_map[config.scheduler_name]
        pipe.scheduler = scheduler_class.from_config(pipe.scheduler.config)
        print(f"\nUsing scheduler: {scheduler_class.__name__}")
    else:
        print(f"\nUsing default scheduler: {type(pipe.scheduler).__name__}")
    
    print(f"\nGeneration Settings:")
    print(f"  Resolution: {config.width}x{config.height}")
    print(f"  Steps: {config.num_inference_steps}")
    print(f"  Guidance Scale: {config.guidance_scale}")
    print(f"  Seed: {config.seed}")
    print(f"  Precision: {'fp16' if config.use_fp16 else 'fp32'}")
    print(f"  TensorRT: {'Enabled' if config.use_tensorrt else 'Disabled'}")
    
    # Model statistics
    print("\n" + "="*60)
    print("MODEL STATISTICS")
    print("="*60)
    print(f"Text Encoder parameters: {count_parameters(pipe.text_encoder):,}")
    print(f"UNet parameters: {count_parameters(pipe.unet):,}")
    print(f"VAE parameters: {count_parameters(pipe.vae):,}")
    print()
    
    # Prepare inputs
    generator = torch.Generator("cuda").manual_seed(config.seed)
    
    # 1. TEXT ENCODER
    print("="*60)
    print("PROFILING TEXT ENCODER")
    print("="*60)
    
    torch.cuda.synchronize()
    text_encode_start = time.perf_counter()
    
    text_inputs = pipe.tokenizer(
        config.prompt,
        padding="max_length",
        max_length=pipe.tokenizer.model_max_length,
        truncation=True,
        return_tensors="pt",
    )
    text_input_ids = text_inputs.input_ids.to("cuda")
    
    with torch.no_grad():
        text_embeddings = pipe.text_encoder(text_input_ids)[0]
    
    torch.cuda.synchronize()
    timings['text_encode_prompt'] = time.perf_counter() - text_encode_start
    print(f"Text Encoder (encode prompt): {timings['text_encode_prompt']:.4f} seconds")
    
    timings['text_encoding_total'] = timings['text_encode_prompt']
    
    # 2. UNET (Denoising)
    print("\n" + "="*60)
    print("PROFILING UNET")
    print("="*60)
    
    # Prepare latents
    pipe.scheduler.set_timesteps(config.num_inference_steps)
    latents = torch.randn(
        (1, pipe.unet.config.in_channels, config.height // 8, config.width // 8),
        generator=generator,
        device="cuda",
        dtype=dtype
    )
    latents = latents * pipe.scheduler.init_noise_sigma
    
    # Warmup for TensorRT (compile on first run)
    if config.use_tensorrt:
        print("Warming up TensorRT (compiling on first run)...")
        with torch.no_grad():
            latent_model_input = pipe.scheduler.scale_model_input(latents, pipe.scheduler.timesteps[0])
            _ = pipe.unet(
                latent_model_input,
                pipe.scheduler.timesteps[0],
                encoder_hidden_states=text_embeddings,
            ).sample
        torch.cuda.synchronize()
        print("Warmup complete!")
    
    # Time each UNet step
    unet_times = []
    for i, t in enumerate(pipe.scheduler.timesteps):
        torch.cuda.synchronize()
        step_start = time.perf_counter()
        
        latent_model_input = pipe.scheduler.scale_model_input(latents, t)
        
        with torch.no_grad():
            noise_pred = pipe.unet(
                latent_model_input,
                t,
                encoder_hidden_states=text_embeddings,
            ).sample
        
        latents = pipe.scheduler.step(noise_pred, t, latents).prev_sample
        
        torch.cuda.synchronize()
        step_time = time.perf_counter() - step_start
        unet_times.append(step_time)
        
        if i % 10 == 0:
            print(f"Step {i}/{config.num_inference_steps}: {step_time:.4f} seconds")
    
    timings['unet_total'] = sum(unet_times)
    timings['unet_avg_step'] = timings['unet_total'] / len(unet_times)
    timings['unet_min_step'] = min(unet_times)
    timings['unet_max_step'] = max(unet_times)
    
    print(f"\nTotal UNet time: {timings['unet_total']:.4f} seconds")
    print(f"Average per step: {timings['unet_avg_step']:.4f} seconds")
    
    # 3. VAE DECODER
    print("\n" + "="*60)
    print("PROFILING VAE DECODER")
    print("="*60)
    
    latents = 1 / pipe.vae.config.scaling_factor * latents
    
    torch.cuda.synchronize()
    vae_start = time.perf_counter()
    
    with torch.no_grad():
        image = pipe.vae.decode(latents).sample
    
    torch.cuda.synchronize()
    timings['vae_decode'] = time.perf_counter() - vae_start
    print(f"VAE Decoder: {timings['vae_decode']:.4f} seconds")
    
    # Post-processing
    image = (image / 2 + 0.5).clamp(0, 1)
    image = image.cpu().permute(0, 2, 3, 1).numpy()
    
    # Convert to PIL Image and save
    from PIL import Image
    image_pil = Image.fromarray((image[0] * 255).astype('uint8'))
    image_pil.save(config.output_filename)
    print(f"\nImage saved to: {config.output_filename}")
    
    # Calculate total time
    timings['total_generation'] = (timings['text_encoding_total'] + 
                                   timings['unet_total'] + 
                                   timings['vae_decode'])
    
    # SUMMARY
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    print(f"Scheduler: {type(pipe.scheduler).__name__}")
    print(f"TensorRT: {'Enabled' if config.use_tensorrt else 'Disabled'}")
    print(f"Model Load: {timings['model_load']:.4f} seconds")
    print(f"Text Encoding total: {timings['text_encoding_total']:.4f} seconds")
    print(f"UNet total: {timings['unet_total']:.4f} seconds ({config.num_inference_steps} steps)")
    print(f"UNet per step: {timings['unet_avg_step']:.4f} seconds")
    print(f"VAE Decoder: {timings['vae_decode']:.4f} seconds")
    print(f"\nTotal generation time: {timings['total_generation']:.4f} seconds")
    
    return image_pil, timings


def test_n_stable_diffusion(config, n):
    timing_results = []
    image_pil = None
    for i in range(n):
        print(f"\n{'='*70}")
        print(f"RUN {i+1}/{n}")
        print('='*70)
        img, t = profile_stable_diffusion(config)
        timing_results.append(t)
        if i == 0:
            image_pil = img
    
    return image_pil, timing_results


if __name__ == "__main__":
    print("\n" + "="*70)
    print("TESTING WITH TENSORRT")
    print("="*70)
    
    config_tensorrt = PromptConfig(
        prompt="Neon cyberpunk street market at night",
        num_inference_steps=50,
        height=768,
        width=768,
        guidance_scale=7.5,
        seed=123,
        scheduler_name="default",
        use_fp16=True,
        use_tensorrt=True,
        output_filename="1.png"
    )
    
    img_tensorrt, tr_tensorrt = test_n_stable_diffusion(config_tensorrt, 1)
    
    print("\n" + "="*70)
    print("PERFORMANCE COMPARISON")
    print("="*70)
    
    print("\nImage generation complete!")