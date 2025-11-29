import os
os.environ["CUDA_VISIBLE_DEVICES"] = "7"
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
from diffusers.quantizers import PipelineQuantizationConfig



@dataclass
class PromptConfig:
    """Configuration for Stable Diffusion generation."""
    
    # Core settings
    prompt: str
    negative_prompt: str = ""
    use_negative_prompt: bool = True  # NEW: Toggle to enable/disable negative prompts
    
    # Generation parameters
    num_inference_steps: int = 50
    guidance_scale: float = 7.5
    seed: int = 42
    
    # Image dimensions (must be multiples of 8)
    height: int = 512
    width: int = 512
    
    # Performance settings
    use_fp16: bool = True
    scheduler_name: str = "default"  # default, ddim, euler, dpm, unipc, etc.
    
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
    
    def __str__(self):
        """Pretty print configuration."""
        negative_status = "Disabled" if not self.use_negative_prompt else f'"{self.negative_prompt[:50]}{"..." if len(self.negative_prompt) > 50 else ""}"'
        return f"""PromptConfig:
  Prompt: "{self.prompt[:60]}{'...' if len(self.prompt) > 60 else ''}"
  Negative: {negative_status}
  Resolution: {self.width}x{self.height}
  Steps: {self.num_inference_steps}
  Guidance Scale: {self.guidance_scale}
  Seed: {self.seed}
  Scheduler: {self.scheduler_name}
  Precision: {'fp16' if self.use_fp16 else 'fp32'}
  Output: {self.output_filename}"""

@contextmanager
def timer(name):
    """Context manager to time code blocks with GPU synchronization"""
    torch.cuda.synchronize()  # Ensure all GPU operations complete before timing
    start = time.perf_counter()
    yield
    torch.cuda.synchronize()  # Ensure all GPU operations complete before timing
    end = time.perf_counter()
    print(f"{name}: {end - start:.4f} seconds")
    return end - start

def count_parameters(model):
    """Count the number of parameters in a model"""
    return sum(p.numel() for p in model.parameters())

def profile_stable_diffusion(config):
    """
    Profile Stable Diffusion with configurable parameters.
    
    Args:
        config: PromptConfig object with all generation settings
    
    Returns:
        tuple: (image_pil, timings_dict) where timings_dict contains all timing measurements
    """
    timings = {}
    
    # Load the model
    print("Loading Stable Diffusion 2.1...")
    model_id = "Manojb/stable-diffusion-2-1-base"
    
    dtype = torch.float16 if config.use_fp16 else torch.float32
    pipeline_quant_config = PipelineQuantizationConfig(
    quant_backend="bitsandbytes_8bit",
    quant_kwargs={"load_in_8bit": True}
)
    
    model_load_start = time.perf_counter()
    pipe = StableDiffusionPipeline.from_pretrained(
        model_id,
        torch_dtype=dtype,
        quantization_config = pipeline_quant_config
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
    print(f"  Negative Prompt: {'Enabled - ' + config.negative_prompt if config.use_negative_prompt and config.negative_prompt else 'Disabled'}")
    
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
    
    # Time each UNet step
    unet_times = []
    for i, t in enumerate(pipe.scheduler.timesteps):
        torch.cuda.synchronize()
        step_start = time.perf_counter()
        
        # Expand latents for classifier free guidance (only if using negative prompt)
        if config.use_negative_prompt:
            latent_model_input = torch.cat([latents] * 2)
        else:
            latent_model_input = latents
        
        latent_model_input = pipe.scheduler.scale_model_input(latent_model_input, t)
        
        # Predict noise
        with torch.no_grad():
            noise_pred = pipe.unet(
                latent_model_input,
                t,
                encoder_hidden_states=text_embeddings,
            ).sample
        
        # Perform guidance (only if using negative prompt)
        if config.use_negative_prompt:
            noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
            noise_pred = noise_pred_uncond + config.guidance_scale * (noise_pred_text - noise_pred_uncond)
        
        # Compute previous noisy sample
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
    print(f"Negative Prompt: {'Enabled' if config.use_negative_prompt else 'Disabled'}")
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
        img, t = profile_stable_diffusion(config)
        timing_results.append(t)
        if i == 0:
            image_pil = img
    
    return image_pil, timing_results




if __name__ == "__main__":
    # Example 1: With negative prompt (default behavior)
    config = PromptConfig(
        prompt="Neon cyberpunk street market at night",
        negative_prompt="blurry, low quality, distorted, ugly",
        use_negative_prompt=False,  # Explicitly enable
        num_inference_steps=50,
        height=768,
        width=768,
        guidance_scale=7.5,
        seed=12345,
        scheduler_name="dpm",
        use_fp16=True,
        output_filename="uh.png"
    )


    # image = profile_stable_diffusion(config)
    img, tr = test_n_stable_diffusion(config, 4)
    breakpoint()
    
    # Example 2: Without negative prompt
    # config = PromptConfig(
    #     prompt="a serene mountain landscape at sunset, highly detailed, 4k",
    #     negative_prompt="blurry, low quality, distorted, ugly",  # Will be ignored
    #     use_negative_prompt=False,  # Disable negative prompt
    #     num_inference_steps=50,
    #     height=768,
    #     width=768,
    #     guidance_scale=7.5,
    #     seed=12345,
    #     scheduler_name="dpm",
    #     use_fp16=True,
    #     output_filename="mountain_sunset_no_negative.png"
    # )
    # image = profile_stable_diffusion(config)
    
    # Example 3: Compare with and without negative prompts
    # for use_neg in [True, False]:
    #     config = PromptConfig(
    #         prompt="a cute cat sitting on a windowsill",
    #         negative_prompt="blurry, ugly, deformed",
    #         use_negative_prompt=use_neg,
    #         num_inference_steps=25,
    #         seed=42,
    #         output_filename=f"cat_{'with' if use_neg else 'without'}_negative.png"
    #     )
    #     print(f"\n{'='*70}")
    #     print(f"Testing {'WITH' if use_neg else 'WITHOUT'} negative prompt")
    #     print('='*70)
    #     image = profile_stable_diffusion(config)
    
    print("\nImage generation complete!")