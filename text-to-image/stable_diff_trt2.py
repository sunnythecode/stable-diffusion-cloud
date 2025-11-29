import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
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

# Bitsandbytes import
try:
    import bitsandbytes as bnb
    BITSANDBYTES_AVAILABLE = True
except ImportError:
    BITSANDBYTES_AVAILABLE = False
    print("Warning: bitsandbytes not available. Install with: pip install bitsandbytes")

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
    
    # Quantization settings
    use_w8a16_quantization: bool = False
    quantize_unet: bool = True
    quantize_text_encoder: bool = False
    quantize_vae: bool = False
    
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
        if self.use_w8a16_quantization and not BITSANDBYTES_AVAILABLE:
            raise RuntimeError("W8A16 quantization requested but bitsandbytes not available. Install with: pip install bitsandbytes")
    
    def __str__(self):
        """Pretty print configuration."""
        negative_status = "Disabled" if not self.use_negative_prompt else f'"{self.negative_prompt[:50]}{"..." if len(self.negative_prompt) > 50 else ""}"'
        quant_components = []
        if self.use_w8a16_quantization:
            if self.quantize_unet:
                quant_components.append("UNet")
            if self.quantize_text_encoder:
                quant_components.append("TextEncoder")
            if self.quantize_vae:
                quant_components.append("VAE")
            quant_status = f"Enabled ({', '.join(quant_components)})"
        else:
            quant_status = "Disabled"
            
        return f"""PromptConfig:
  Prompt: "{self.prompt[:60]}{'...' if len(self.prompt) > 60 else ''}"
  Negative: {negative_status}
  Resolution: {self.width}x{self.height}
  Steps: {self.num_inference_steps}
  Guidance Scale: {self.guidance_scale}
  Seed: {self.seed}
  Scheduler: {self.scheduler_name}
  Precision: {'fp16' if self.use_fp16 else 'fp32'}
  W8A16 Quantization: {quant_status}
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

def get_model_memory_usage(model):
    """Calculate approximate memory usage of a model in MB"""
    total_bytes = 0
    for param in model.parameters():
        total_bytes += param.nelement() * param.element_size()
    for buffer in model.buffers():
        total_bytes += buffer.nelement() * buffer.element_size()
    return total_bytes / (1024 ** 2)  # Convert to MB

def quantize_model_w8a16(model, model_name="model"):
    """
    Apply W8A16 quantization to a model using bitsandbytes.
    Weights are quantized to 8-bit, activations remain at 16-bit.
    """
    print(f"\nQuantizing {model_name} to W8A16...")
    
    # Get memory before quantization
    mem_before = get_model_memory_usage(model)
    
    # Replace Linear layers with 8-bit Linear layers
    def replace_linear_with_int8(module, name=""):
        for child_name, child in module.named_children():
            full_name = f"{name}.{child_name}" if name else child_name
            
            if isinstance(child, torch.nn.Linear):
                # Create 8-bit linear layer
                int8_linear = bnb.nn.Linear8bitLt(
                    child.in_features,
                    child.out_features,
                    bias=child.bias is not None,
                    has_fp16_weights=False,  # Use INT8 weights
                    threshold=6.0,
                )
                
                # Copy weights and bias
                int8_linear.weight.data = child.weight.data
                if child.bias is not None:
                    int8_linear.bias.data = child.bias.data
                
                # Replace the layer
                setattr(module, child_name, int8_linear)
                print(f"  Quantized: {full_name} ({child.in_features} -> {child.out_features})")
            else:
                # Recursively apply to child modules
                replace_linear_with_int8(child, full_name)
    
    replace_linear_with_int8(model)
    
    # Get memory after quantization
    mem_after = get_model_memory_usage(model)
    compression_ratio = mem_before / mem_after if mem_after > 0 else 0
    
    print(f"  Memory before: {mem_before:.2f} MB")
    print(f"  Memory after: {mem_after:.2f} MB")
    print(f"  Compression ratio: {compression_ratio:.2f}x")
    print(f"  Memory saved: {mem_before - mem_after:.2f} MB ({(1 - mem_after/mem_before)*100:.1f}%)")
    
    return model

def load_tensorrt_pipeline(model_id, config):
    """
    Load Stable Diffusion pipeline with TensorRT optimization.
    
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

def profile_stable_diffusion(config):
    """
    Profile Stable Diffusion with configurable parameters, optional W8A16 quantization, and TensorRT.
    """
    timings = {}
    
    # Load the model
    print("Loading Stable Diffusion 2.1...")
    model_id = "Manojb/stable-diffusion-2-1-base"
    
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
    
    # Apply W8A16 quantization if requested
    if config.use_w8a16_quantization:
        print("\n" + "="*60)
        print("APPLYING W8A16 QUANTIZATION")
        print("="*60)
        
        quant_start = time.perf_counter()
        
        if config.quantize_unet:
            pipe.unet = quantize_model_w8a16(pipe.unet, "UNet")
        
        if config.quantize_text_encoder:
            pipe.text_encoder = quantize_model_w8a16(pipe.text_encoder, "Text Encoder")
        
        if config.quantize_vae:
            pipe.vae = quantize_model_w8a16(pipe.vae, "VAE")
        
        timings['quantization'] = time.perf_counter() - quant_start
        print(f"\nTotal quantization time: {timings['quantization']:.4f} seconds")
    else:
        timings['quantization'] = 0.0
    
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
    print(f"  W8A16 Quantization: {'Enabled' if config.use_w8a16_quantization else 'Disabled'}")
    print(f"  TensorRT: {'Enabled' if config.use_tensorrt else 'Disabled'}")
    print(f"  Negative Prompt: {'Enabled - ' + config.negative_prompt if config.use_negative_prompt and config.negative_prompt else 'Disabled'}")
    
    # Model statistics
    print("\n" + "="*60)
    print("MODEL STATISTICS")
    print("="*60)
    print(f"Text Encoder parameters: {count_parameters(pipe.text_encoder):,}")
    print(f"Text Encoder memory: {get_model_memory_usage(pipe.text_encoder):.2f} MB")
    print(f"UNet parameters: {count_parameters(pipe.unet):,}")
    print(f"UNet memory: {get_model_memory_usage(pipe.unet):.2f} MB")
    print(f"VAE parameters: {count_parameters(pipe.vae):,}")
    print(f"VAE memory: {get_model_memory_usage(pipe.vae):.2f} MB")
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
    
    # Encode unconditional (negative prompt) - only if enabled
    if config.use_negative_prompt:
        torch.cuda.synchronize()
        neg_encode_start = time.perf_counter()
        
        uncond_input = pipe.tokenizer(
            config.negative_prompt if config.negative_prompt else "",
            padding="max_length",
            max_length=pipe.tokenizer.model_max_length,
            return_tensors="pt",
        )
        uncond_input_ids = uncond_input.input_ids.to("cuda")
        
        with torch.no_grad():
            uncond_embeddings = pipe.text_encoder(uncond_input_ids)[0]
        
        torch.cuda.synchronize()
        timings['text_encode_negative'] = time.perf_counter() - neg_encode_start
        print(f"Text Encoder (encode negative prompt): {timings['text_encode_negative']:.4f} seconds")
        
        text_embeddings = torch.cat([uncond_embeddings, text_embeddings])
        timings['text_encoding_total'] = timings['text_encode_prompt'] + timings['text_encode_negative']
    else:
        print("Negative prompt disabled - skipping unconditional encoding")
        timings['text_encode_negative'] = 0.0
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
    
    # Warmup for TensorRT (compile on first run) or quantized model
    if config.use_tensorrt or config.use_w8a16_quantization:
        print("Warming up model (compiling/calibrating on first run)...")
        with torch.no_grad():
            latent_model_input = torch.cat([latents] * 2) if config.use_negative_prompt else latents
            latent_model_input = pipe.scheduler.scale_model_input(latent_model_input, pipe.scheduler.timesteps[0])
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
        
        if config.use_negative_prompt:
            latent_model_input = torch.cat([latents] * 2)
        else:
            latent_model_input = latents
        
        latent_model_input = pipe.scheduler.scale_model_input(latent_model_input, t)
        
        with torch.no_grad():
            noise_pred = pipe.unet(
                latent_model_input,
                t,
                encoder_hidden_states=text_embeddings,
            ).sample
        
        if config.use_negative_prompt:
            noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
            noise_pred = noise_pred_uncond + config.guidance_scale * (noise_pred_text - noise_pred_uncond)
        
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
    print(f"W8A16 Quantization: {'Enabled' if config.use_w8a16_quantization else 'Disabled'}")
    print(f"TensorRT: {'Enabled' if config.use_tensorrt else 'Disabled'}")
    print(f"Negative Prompt: {'Enabled' if config.use_negative_prompt else 'Disabled'}")
    print(f"Model Load: {timings['model_load']:.4f} seconds")
    if config.use_w8a16_quantization:
        print(f"Quantization: {timings['quantization']:.4f} seconds")
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
    # Example: Compare baseline vs W8A16 quantization
    
    print("\n" + "="*70)
    print("TESTING BASELINE (NO QUANTIZATION)")
    print("="*70)
    
    config_baseline = PromptConfig(
        prompt="a serene mountain landscape at sunset, highly detailed, 4k",
        negative_prompt="blurry, low quality, distorted, ugly",
        use_negative_prompt=False,
        num_inference_steps=20,
        height=512,
        width=512,
        guidance_scale=7.5,
        seed=123,
        scheduler_name="default",
        use_fp16=True,
        use_w8a16_quantization=False,
        use_tensorrt=True,
        output_filename="mountain_sunset_baseline.png"
    )
    
    img_baseline, tr_baseline = test_n_stable_diffusion(config_baseline, 3)
    
