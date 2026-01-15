import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import base64
import io
import json
import time
from typing import Optional, List
import asyncio
from concurrent.futures import ThreadPoolExecutor
import torch
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field
import numpy as np
from PIL import Image
from diffusers import (
    StableDiffusionPipeline,
    DDIMScheduler,
    EulerDiscreteScheduler,
    DPMSolverMultistepScheduler,
)
from transformers import CLIPModel, CLIPProcessor

# Configuration
MODEL_ID = "Manojb/stable-diffusion-2-1-base"
NUM_GPUS = 1
print(f"Available GPUs: {NUM_GPUS}")

app = FastAPI(title="Stable Diffusion API", version="1.0.0")

# Enable CORS for public access
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify your frontend domain
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def load_tensorrt_pipeline(model_id, device):
    """
    Load Stable Diffusion pipeline with TensorRT optimizationf.
    
    Method 1: Using torch.compile with TensorRT backend (PyTorch 2.0+)
    This is the simplest approach.
    """
    print("\n" + "="*60)
    print("LOADING WITH TENSORRT OPTIMIZATION")
    print("="*60)
    
    dtype = torch.float16
    
    # Load base pipeline
    pipe = StableDiffusionPipeline.from_pretrained(
        model_id,
        torch_dtype=dtype,
    ).to(device)
    
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

# Request/Response Models
class GenerationRequest(BaseModel):
    prompt: str = Field(..., description="Text prompt for image generation")
    num_steps: int = Field(default=50, ge=1, le=100, description="Number of inference steps")
    num_seeds: int = Field(default=1, ge=1, le=10, description="Number of different seeds to generate")
    num_guidance_samples: int = Field(default=1, ge=1, le=10, description="Number of guidance scale samples")
    guidance_scale_min: float = Field(default=7.5, ge=1.0, le=20.0)
    guidance_scale_max: float = Field(default=7.5, ge=1.0, le=20.0)
    compute_clip_score: bool = Field(default=False, description="Compute CLIP score for generated images")
    seeds: Optional[List[int]] = Field(default=None, description="Specific seeds to use")
    height: int = Field(default=512, description="Image height (must be multiple of 8)")
    width: int = Field(default=512, description="Image width (must be multiple of 8)")
    scheduler: str = Field(default="default", description="Scheduler type: default, ddim, euler, dpm")

# Global models - one pipeline per GPU
pipelines = []
clip_model = None
clip_processor = None
executor = None

def initialize_models():
    """Initialize Stable Diffusion pipelines on all available GPUs"""
    global pipelines, clip_model, clip_processor, executor
    
    print("\n" + "="*60)
    print("INITIALIZING MODELS")
    print("="*60)
    
    # Initialize one pipeline per GPU
    for gpu_id in range(NUM_GPUS):
        print(f"\nLoading Stable Diffusion on GPU {gpu_id}...")
        device = f"cuda:{gpu_id}"
        
        # pipe = StableDiffusionPipeline.from_pretrained(
        #     MODEL_ID,
        #     torch_dtype=torch.float16,
        #     safety_checker=None,
        # ).to(device)
        pipe = load_tensorrt_pipeline(MODEL_ID, device)
        
        # Enable memory optimizations
        pipe.enable_attention_slicing()
        
        pipelines.append(pipe)
        print(f"✓ Pipeline loaded on GPU {gpu_id}")
    
    # Load CLIP model for scoring on first GPU
    print("\nLoading CLIP model for scoring...")
    clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to("cuda:0")
    clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
    print("✓ CLIP model loaded")
    
    # Create thread pool for parallel generation
    executor = ThreadPoolExecutor(max_workers=NUM_GPUS)
    
    print("\n" + "="*60)
    print(f"INITIALIZATION COMPLETE - {NUM_GPUS} GPUs ready")
    print("="*60 + "\n")

def calculate_guidance_scale_values(min_scale: float, max_scale: float, num_samples: int) -> List[float]:
    """Calculate guidance scale values rounded to nearest 0.5"""
    if num_samples == 1:
        return [round(min_scale * 2) / 2]
    
    values = []
    step = (max_scale - min_scale) / (num_samples - 1)
    for i in range(num_samples):
        value = min_scale + step * i
        values.append(round(value * 2) / 2)
    return values

def image_to_base64(pil_image: Image.Image) -> str:
    """Convert PIL Image to base64 string"""
    buffered = io.BytesIO()
    pil_image.save(buffered, format="PNG")
    img_str = base64.b64encode(buffered.getvalue()).decode()
    return img_str

def compute_clip_score(pil_image: Image.Image, text: str) -> tuple:
    """Compute CLIP score for a PIL Image and text prompt"""
    start_time = time.time()
    
    inputs = clip_processor(
        text=[text], 
        images=pil_image, 
        return_tensors="pt", 
        padding=True
    ).to("cuda:0")
    
    with torch.no_grad():
        image_features = clip_model.get_image_features(pixel_values=inputs["pixel_values"])
        text_features = clip_model.get_text_features(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"]
        )
    
    # Normalize and compute similarity
    image_features /= image_features.norm(p=2, dim=-1, keepdim=True)
    text_features /= text_features.norm(p=2, dim=-1, keepdim=True)
    clip_score = (image_features @ text_features.T).item()
    
    computation_time = time.time() - start_time
    return clip_score, computation_time

def generate_single_image(
    gpu_id: int,
    prompt: str,
    num_steps: int,
    seed: int,
    guidance_scale: float,
    height: int,
    width: int,
    scheduler: str,
) -> tuple:
    """Generate a single image on a specific GPU"""
    device = f"cuda:{gpu_id}"
    pipe = pipelines[gpu_id]
    
    # Set scheduler if needed
    if scheduler != "default":
        scheduler_map = {
            "ddim": DDIMScheduler,
            "euler": EulerDiscreteScheduler,
            "dpm": DPMSolverMultistepScheduler,
        }
        if scheduler in scheduler_map:
            pipe.scheduler = scheduler_map[scheduler].from_config(pipe.scheduler.config)
    
    # Generate
    generator = torch.Generator(device=device).manual_seed(seed)
    
    start_time = time.time()
    with torch.no_grad():
        output = pipe(
            prompt=prompt,
            num_inference_steps=num_steps,
            guidance_scale=guidance_scale,
            generator=generator,
            height=height,
            width=width,
        )
    
    generation_time = time.time() - start_time
    image = output.images[0]
    
    return image, generation_time

async def generate_images_stream(request: GenerationRequest):
    """Generate images and stream results as they complete"""
    
    # Prepare seeds
    seeds = request.seeds if request.seeds and len(request.seeds) == request.num_seeds else [
        np.random.randint(0, 2147483647) for _ in range(request.num_seeds)
    ]
    
    # Calculate guidance scales
    guidance_scales = calculate_guidance_scale_values(
        request.guidance_scale_min,
        request.guidance_scale_max,
        request.num_guidance_samples
    )
    
    print(f"\n{'='*60}")
    print(f"Generation Request:")
    print(f"  Prompt: '{request.prompt[:50]}...'")
    print(f"  Seeds: {seeds}")
    print(f"  Guidance scales: {guidance_scales}")
    print(f"  Steps: {request.num_steps}")
    print(f"  Resolution: {request.width}x{request.height}")
    print(f"  Using {NUM_GPUS} GPUs")
    print(f"{'='*60}\n")
    
    total_start_time = time.time()
    generation_times = []
    
    # Create all tasks
    tasks = []
    task_metadata = []
    
    for seed_idx, seed in enumerate(seeds):
        for guidance_idx, guidance_scale in enumerate(guidance_scales):
            # Assign to GPU in round-robin fashion
            gpu_id = len(tasks) % NUM_GPUS
            
            tasks.append((
                gpu_id,
                request.prompt,
                request.num_steps,
                seed,
                guidance_scale,
                request.height,
                request.width,
                request.scheduler,
            ))
            
            task_metadata.append({
                'seed_index': seed_idx,
                'guidance_index': guidance_idx,
                'seed': seed,
                'guidance_scale': guidance_scale,
                'gpu_id': gpu_id,
            })
    
    # Execute tasks in parallel using thread pool
    loop = asyncio.get_event_loop()
    futures = [
        loop.run_in_executor(executor, generate_single_image, *task)
        for task in tasks
    ]
    
    # Stream results as they complete
    for idx, future in enumerate(asyncio.as_completed(futures)):
        try:
            pil_image, generation_time = await future
            generation_times.append(generation_time)
            
            metadata = task_metadata[idx]
            print(f"✓ Generated: Seed {metadata['seed']}, Guidance {metadata['guidance_scale']}, "
                  f"GPU {metadata['gpu_id']}, Time: {generation_time:.2f}s")
            
            # Convert to base64
            img_base64 = image_to_base64(pil_image)
            
            # Compute CLIP score if requested
            clip_score = None
            clip_computation_time = None
            if request.compute_clip_score:
                clip_score, clip_computation_time = compute_clip_score(pil_image, request.prompt)
            
            # Prepare response
            response_data = {
                'type': 'image',
                **metadata,
                'imageBase64': img_base64,
                'generationTime': round(generation_time, 2),
                'clipScore': round(clip_score, 4) if clip_score is not None else None,
                'clipComputationTime': round(clip_computation_time, 2) if clip_computation_time is not None else None,
            }
            
            yield f"data: {json.dumps(response_data)}\n\n"
            
        except Exception as e:
            print(f"Error generating image {idx}: {str(e)}")
            error_data = {
                'type': 'error',
                'message': str(e),
                **task_metadata[idx]
            }
            yield f"data: {json.dumps(error_data)}\n\n"
    
    # Send completion message
    total_time = time.time() - total_start_time
    average_time = sum(generation_times) / len(generation_times) if generation_times else 0
    
    completion_data = {
        'type': 'complete',
        'totalLatency': round(total_time, 2),
        'averageLatency': round(average_time, 2),
        'totalImages': len(generation_times),
    }
    
    print(f"\n{'='*60}")
    print(f"Generation Complete:")
    print(f"  Total time: {total_time:.2f}s")
    print(f"  Average per image: {average_time:.2f}s")
    print(f"  Images generated: {len(generation_times)}")
    print(f"{'='*60}\n")
    
    yield f"data: {json.dumps(completion_data)}\n\n"

@app.post("/generate")
async def generate(request: GenerationRequest):
    """Generate images with streaming response"""
    try:
        return StreamingResponse(
            generate_images_stream(request),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "X-Accel-Buffering": "no",
            }
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
async def health():
    """Health check endpoint"""
    return {
        "status": "ok",
        "models_loaded": len(pipelines) > 0,
        "num_gpus": NUM_GPUS,
        "gpu_names": [torch.cuda.get_device_name(i) for i in range(NUM_GPUS)]
    }

@app.get("/")
async def root():
    """Root endpoint with API info"""
    return {
        "message": "Stable Diffusion API",
        "version": "1.0.0",
        "num_gpus": NUM_GPUS,
        "endpoints": {
            "generate": "POST /generate - Generate images",
            "health": "GET /health - Health check",
        }
    }

@app.on_event("startup")
async def startup_event():
    """Initialize models on startup"""
    initialize_models()

@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on shutdown"""
    global executor
    if executor:
        executor.shutdown(wait=True)
    print("Server shutdown complete")

if __name__ == "__main__":
    import uvicorn
    
    # Run server
    # Use host="0.0.0.0" to make it accessible from any device
    # In production, use a reverse proxy (nginx) with SSL
    print("\n" + "="*60)
    print("Starting FastAPI Server")
    print("="*60)
    print(f"Server will be accessible at: http://YOUR_PUBLIC_IP:8000")
    print(f"API docs available at: http://YOUR_PUBLIC_IP:8000/docs")
    print("="*60 + "\n")
    
    uvicorn.run(
        app,
        host="0.0.0.0",  # Listen on all network interfaces
        port=8000,
        log_level="info",
    )