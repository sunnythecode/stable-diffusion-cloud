# trt_sd_pipeline.py
from diffusers import StableDiffusionPipeline
import torch
import numpy as np
from trt_unet import TRTUNet
import time
from PIL import Image

class TRTStableDiffusionPipeline:
    def __init__(self, model_id, trt_engine_path):
        """SD pipeline with TensorRT UNet"""
        print("Loading Stable Diffusion pipeline...")
        self.pipe = StableDiffusionPipeline.from_pretrained(
            model_id,
            torch_dtype=torch.float16
        ).to("cuda")
        
        # Load TensorRT UNet
        print("Loading TensorRT UNet (W8A16)...")
        self.trt_unet = TRTUNet(trt_engine_path)
        
        print("✓ Pipeline ready!")
    
    def __call__(self, prompt, negative_prompt="", num_inference_steps=50, guidance_scale=7.5, seed=None):
        """Generate image"""
        
        if seed is not None:
            torch.manual_seed(seed)
            np.random.seed(seed)
        
        # Encode prompts
        print(f"Generating: '{prompt}'")
        
        # Positive prompt
        text_inputs = self.pipe.tokenizer(
            [prompt],
            padding="max_length",
            max_length=77,
            truncation=True,
            return_tensors="pt"
        )
        
        with torch.no_grad():
            text_embeddings = self.pipe.text_encoder(text_inputs.input_ids.to("cuda"))[0]
        
        # Negative prompt (for classifier-free guidance)
        uncond_input = self.pipe.tokenizer(
            [negative_prompt if negative_prompt else ""],
            padding="max_length",
            max_length=77,
            return_tensors="pt"
        )
        
        with torch.no_grad():
            uncond_embeddings = self.pipe.text_encoder(uncond_input.input_ids.to("cuda"))[0]
        
        # Concatenate for classifier-free guidance [negative, positive]
        text_embeddings = torch.cat([uncond_embeddings, text_embeddings])
        
        # Prepare latents
        latents = torch.randn(
            (1, 4, 64, 64),
            device="cuda",
            dtype=torch.float16
        )
        
        # Initialize scheduler
        self.pipe.scheduler.set_timesteps(num_inference_steps)
        latents = latents * self.pipe.scheduler.init_noise_sigma
        
        # Denoising loop
        print(f"Running {num_inference_steps} denoising steps...")
        start_time = time.time()
        
        for i, t in enumerate(self.pipe.scheduler.timesteps):
            # Expand latents for classifier-free guidance
            latent_model_input = torch.cat([latents] * 2)
            latent_model_input = self.pipe.scheduler.scale_model_input(latent_model_input, t)
            
            # Convert to numpy for TRT
            sample_np = latent_model_input.cpu().numpy().astype(np.float16)
            timestep_np = np.array([t.item()], dtype=np.int64)
            encoder_hidden_states_np = text_embeddings.cpu().numpy().astype(np.float16)
            
            # TRT inference
            noise_pred = self.trt_unet.infer(
                sample_np,
                timestep_np,
                encoder_hidden_states_np
            )
            
            # Perform guidance
            noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
            noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)
            
            # Compute previous noisy sample
            latents = self.pipe.scheduler.step(noise_pred, t, latents).prev_sample
            
            if (i + 1) % 10 == 0:
                elapsed = time.time() - start_time
                print(f"  Step {i+1}/{num_inference_steps} ({elapsed:.1f}s, {elapsed/(i+1):.3f}s/step)")
        
        inference_time = time.time() - start_time
        print(f"✓ Inference completed in {inference_time:.2f}s ({inference_time/num_inference_steps:.3f}s per step)")
        
        # Decode latents
        print("Decoding image...")
        with torch.no_grad():
            image = self.pipe.vae.decode(latents / 0.18215).sample
        
        # Convert to PIL
        image = (image / 2 + 0.5).clamp(0, 1)
        image = image.cpu().permute(0, 2, 3, 1).numpy()
        image = (image * 255).astype(np.uint8)
        
        return Image.fromarray(image[0])

# Usage example
if __name__ == "__main__":
    # Initialize pipeline
    pipeline = TRTStableDiffusionPipeline(
        model_id="Manojb/stable-diffusion-2-1-base",
        trt_engine_path="unet_fp16.trt"
    )
    
    # Generate image
    image = pipeline(
        prompt="a beautiful landscape with mountains and lake, highly detailed, 4k",
        negative_prompt="blurry, low quality, distorted",
        num_inference_steps=50,
        guidance_scale=7.5,
        seed=42
    )
    
    # Save image
    image.save("output_trt_a16.png")
    print("✓ Image saved to output_trt_w8a16.png")