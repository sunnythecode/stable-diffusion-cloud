import torch
from diffusers import StableDiffusionPipeline
from PIL import Image
import os
import pandas as pd
from pathlib import Path
import multiprocessing as mp
from functools import partial
import dataclasses
import time
from contextlib import contextmanager
import bitsandbytes as bnb
BITSANDBYTES_AVAILABLE = True

# Configuration
GPU_LIST = [0, 1, 2, 3, 4, 5, 6, 7]  # List of GPU IDs to use
MODEL_ID = "Manojb/stable-diffusion-2-1-base"
METADATA_CSV = "/home/sandeep_b/qualcomm_sd/coco_val_captions.csv"
OUTPUT_DIR = "/home/sandeep_b/qualcomm_sd/coco_gen_images_12345"

# Generation parameters
NUM_INFERENCE_STEPS = 50
GUIDANCE_SCALE = 7.5
HEIGHT = 512
WIDTH = 512

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
    # mem_before = get_model_memory_usage(model)
    
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
                # print(f"  Quantized: {full_name} ({child.in_features} -> {child.out_features})")
            else:
                # Recursively apply to child modules
                replace_linear_with_int8(child, full_name)
    
    replace_linear_with_int8(model)
    
    # Get memory after quantization
    # mem_after = get_model_memory_usage(model)
    # compression_ratio = mem_before / mem_after if mem_after > 0 else 0
    
    # print(f"  Memory before: {mem_before:.2f} MB")
    # print(f"  Memory after: {mem_after:.2f} MB")
    # print(f"  Compression ratio: {compression_ratio:.2f}x")
    # print(f"  Memory saved: {mem_before - mem_after:.2f} MB ({(1 - mem_after/mem_before)*100:.1f}%)")
    
    return model

def setup_pipeline(gpu_id):
    """Initialize the Stable Diffusion pipeline with W8A16 quantization on specified GPU"""
    device = f"cuda:{gpu_id}"
    
    print(f"[GPU {gpu_id}] Loading model on {torch.cuda.get_device_name(gpu_id)}")
    
    # Load pipeline with 8-bit quantization
    pipe = StableDiffusionPipeline.from_pretrained(
        MODEL_ID,
        torch_dtype=torch.float16,
        # load_in_8bit=True,  # Enable 8-bit quantization
        # device_map={"": device},  # Map to specific GPU
        safety_checker=None,
    )
    pipe.unet = quantize_model_w8a16(pipe.unet, "UNet")
    pipe.text_encoder = quantize_model_w8a16(pipe.text_encoder, "Text Encoder")
    pipe.vae = quantize_model_w8a16(pipe.vae, "VAE")
    # pipe.unet = torch.quantization.quantize_dynamic(
    #     pipe.unet, {torch.nn.Linear}, dtype=torch.qint8
    # )
    pipe = pipe.to(device)

    
    # Note: When using load_in_8bit, the model is already on the device
    # so we don't need pipe.to(device)
    
    # Enable memory optimizations
    pipe.enable_attention_slicing()
    
    # Optional: Enable xformers for even better performance
    # try:
    #     pipe.enable_xformers_memory_efficient_attention()
    #     print(f"[GPU {gpu_id}] xformers enabled")
    # except:
    #     print(f"[GPU {gpu_id}] xformers not available")
    
    print(f"[GPU {gpu_id}] Model loaded with 8-bit quantization")
    
    return pipe, device

def worker(gpu_id, prompt_batch, output_dir):
    """Worker function that processes a batch of prompts on a specific GPU"""
    try:
        # Setup pipeline on this GPU
        pipe, device = setup_pipeline(gpu_id)
        
        os.makedirs(output_dir, exist_ok=True)
        
        print(f"[GPU {gpu_id}] Processing {len(prompt_batch)} prompts")
        
        for idx, row in prompt_batch.iterrows():
            prompt = row['caption']  # Adjust column name based on your CSV
            original_idx = row['filename']
            
            try:
                seed = 12345
                generator = torch.Generator(device="cuda").manual_seed(seed)
                # Generate image
                with torch.autocast(device):
                    image = pipe(
                        prompt=prompt,
                        height=HEIGHT,
                        width=WIDTH,
                        num_inference_steps=NUM_INFERENCE_STEPS,
                        guidance_scale=GUIDANCE_SCALE,
                        generator=generator
                    ).images[0]
                
                # Resize to match real images if needed
                image = image.resize((299, 299), Image.LANCZOS)
                
                # Save with same naming convention as real images
                output_path = os.path.join(output_dir, original_idx)
                image.save(output_path)
                
                if (idx + 1) % 10 == 0:
                    print(f"[GPU {gpu_id}] Generated {idx + 1}/{len(prompt_batch)} images")
                    
            except Exception as e:
                print(f"[GPU {gpu_id}] Error generating image {original_idx}: {e}")
                continue
        
        print(f"[GPU {gpu_id}] Completed batch")
        
        # Cleanup
        del pipe
        torch.cuda.empty_cache()
        
    except Exception as e:
        print(f"[GPU {gpu_id}] Worker error: {e}")

def split_dataframe(df, n_splits):
    """Split dataframe into n_splits chunks"""
    chunk_size = len(df) // n_splits
    chunks = []
    
    for i in range(n_splits):
        start_idx = i * chunk_size
        end_idx = start_idx + chunk_size if i < n_splits - 1 else len(df)
        chunks.append(df.iloc[start_idx:end_idx])
    
    return chunks

def main():
    # Load metadata CSV
    print(f"Loading metadata from {METADATA_CSV}")
    df = pd.read_csv(METADATA_CSV)
    
    # Determine caption column name
    caption_columns = ['caption', 'TEXT', 'text', 'prompt']
    caption_col = None
    for col in caption_columns:
        if col in df.columns:
            caption_col = col
            break
    
    if caption_col is None:
        print(f"Available columns: {df.columns.tolist()}")
        raise ValueError("Could not find caption column in CSV")
    
    # Rename to standard 'caption' for consistency
    if caption_col != 'caption':
        df['caption'] = df[caption_col]
    
    print(f"Loaded {len(df)} prompts")
    print(f"Using {len(GPU_LIST)} GPUs: {GPU_LIST}")
    
    # Create output directory
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # Split prompts across GPUs
    prompt_batches = split_dataframe(df, len(GPU_LIST))
    
    print(f"\nBatch sizes: {[len(batch) for batch in prompt_batches]}")
    
    # Create processes for each GPU
    processes = []
    for gpu_id, batch in zip(GPU_LIST, prompt_batches):
        p = mp.Process(target=worker, args=(gpu_id, batch, OUTPUT_DIR))
        p.start()
        processes.append(p)
        print(f"Started worker on GPU {gpu_id} with {len(batch)} prompts")
    
    # Wait for all processes to complete
    for p in processes:
        p.join()
    
    print("\nAll workers completed!")
    print(f"Generated images saved to {OUTPUT_DIR}/")

if __name__ == "__main__":
    # Required for multiprocessing on some systems
    mp.set_start_method('spawn', force=True)
    main()