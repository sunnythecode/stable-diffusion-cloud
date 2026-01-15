import torch
from diffusers import StableDiffusionPipeline
from bitsandbytes.nn import Linear8bitLt

def quantize_to_w8a16(model):
    """
    Recursively replaces all Linear layers with bitsandbytes 8-bit Linear layers.
    """
    for name, module in model.named_children():
        if isinstance(module, torch.nn.Linear):
            # Replace Linear with 8-bit Linear
            # has_fp16_weights=False keeps the storage in 8-bit
            new_module = Linear8bitLt(
                module.in_features, 
                module.out_features, 
                bias=module.bias is not None, 
                has_fp16_weights=False, 
                threshold=6.0
            )
            # Copy weights and bias to the new module
            new_module.weight.data = module.weight.data
            if module.bias is not None:
                new_module.bias.data = module.bias.data
            setattr(model, name, new_module)
        else:
            quantize_to_w8a16(module)

def load_model(model_id="Manojb/stable-diffusion-2-1-base"):
    print(f"Loading model: {model_id}")
    # Load in FP16 but on CPU first to perform quantization
    pipe = StableDiffusionPipeline.from_pretrained(
        model_id, 
        torch_dtype=torch.float16
    )

    print("Quantizing UNet, Text Encoder, and VAE to W8A16...")
    quantize_to_w8a16(pipe.unet)
    quantize_to_w8a16(pipe.text_encoder)
    quantize_to_w8a16(pipe.vae)

    # Move the quantized pipeline to GPU
    pipe.to("cuda")
    print("✅ Model loaded and quantized.")
    return pipe

def generate(pipe, prompt, negative_prompt="", guidance_scale=7.5, seed=None, num_steps=25):
    # Set up generator for reproducibility
    generator = None
    if seed is not None:
        generator = torch.Generator(device="cuda").manual_seed(seed)

    print(f"Generating image with seed: {seed if seed is not None else 'Random'}")
    
    with torch.inference_mode():
        result = pipe(
            prompt=prompt,
            negative_prompt=negative_prompt,
            guidance_scale=guidance_scale,
            num_inference_steps=num_steps,
            generator=generator
        )
    
    return result.images[0]

# --- Usage Example ---
if __name__ == "__main__":
    # 1. Load and quantize once
    pipeline = load_model()

    # 2. Generate with specific parameters
    my_prompt = "A high-tech digital art of a futuristic city, 8k"
    my_neg_prompt = "blurry, distorted, low quality, text"
    
    image = generate(
        pipe=pipeline, 
        prompt=my_prompt, 
        negative_prompt=my_neg_prompt, 
        guidance_scale=8.5, 
        seed=42,
        num_steps=25
    )

    # 3. Save
    image.save("structured_w8a16_output.png")
    print("Done!")