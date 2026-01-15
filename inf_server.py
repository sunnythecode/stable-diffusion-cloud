import torch
import asyncio
import base64
import io
import time
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from typing import List
from diffusers import StableDiffusionPipeline
from bitsandbytes.nn import Linear8bitLt
from transformers import CLIPProcessor, CLIPModel
from PIL import Image

# --- Configuration ---
MODEL_ID = "Manojb/stable-diffusion-2-1-base"
CLIP_MODEL_ID = "openai/clip-vit-base-patch32"

# --- Request Model ---
class GenerateRequest(BaseModel):
    prompt: str
    negative_prompt: str = ""
    num_steps: int = 20
    num_seeds: int = 1
    num_guidance_samples: int = 1
    guidance_scale_min: float = 7.5
    guidance_scale_max: float = 7.5
    compute_clip_score: bool = False
    seeds: List[int]

# --- Quantization Functions ---
def quantize_to_w8a16(model):
    """
    Recursively replaces all Linear layers with bitsandbytes 8-bit Linear layers.
    """
    for name, module in model.named_children():
        if isinstance(module, torch.nn.Linear):
            new_module = Linear8bitLt(
                module.in_features, 
                module.out_features, 
                bias=module.bias is not None, 
                has_fp16_weights=False, 
                threshold=6.0
            )
            new_module.weight.data = module.weight.data
            if module.bias is not None:
                new_module.bias.data = module.bias.data
            setattr(model, name, new_module)
        else:
            quantize_to_w8a16(module)

def load_model(model_id=MODEL_ID):
    """Load and quantize the Stable Diffusion model"""
    print(f"Loading model: {model_id}")
    pipe = StableDiffusionPipeline.from_pretrained(
        model_id, 
        torch_dtype=torch.float16
    )

    print("Quantizing UNet, Text Encoder, and VAE to W8A16...")
    quantize_to_w8a16(pipe.unet)
    quantize_to_w8a16(pipe.text_encoder)
    quantize_to_w8a16(pipe.vae)

    pipe.to("cuda")
    print("✅ Model loaded and quantized.")
    return pipe

# --- Global Models ---
pipeline = None
clip_model = None
clip_processor = None

# --- CLIP Score Computation ---
def compute_clip_score(pil_image, text):
    """
    Compute CLIP score for a PIL Image and text prompt
    Returns: (clip_score, computation_time)
    """
    start_time = time.time()

    # Preprocess
    inputs = clip_processor(
        text=[text], 
        images=pil_image,
        return_tensors="pt", 
        padding=True
    ).to("cuda")

    # Forward pass
    with torch.no_grad():
        image_features = clip_model.get_image_features(pixel_values=inputs["pixel_values"])
        text_features = clip_model.get_text_features(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"]
        )

    # Normalize features
    image_features /= image_features.norm(p=2, dim=-1, keepdim=True)
    text_features /= text_features.norm(p=2, dim=-1, keepdim=True)

    # Compute cosine similarity
    clip_score = (image_features @ text_features.T).item()

    computation_time = time.time() - start_time

    return clip_score, computation_time

# --- Generation Function ---
def generate_single_image(
    prompt: str,
    negative_prompt: str,
    guidance_scale: float,
    seed: int,
    num_steps: int,
    seed_index: int,
    guidance_index: int,
    compute_clip: bool
):
    """Generate a single image with given parameters"""
    start_time = time.time()
    
    generator = torch.Generator(device="cuda").manual_seed(seed)
    
    with torch.inference_mode():
        result = pipeline(
            prompt=prompt,
            negative_prompt=negative_prompt,
            guidance_scale=guidance_scale,
            num_inference_steps=num_steps,
            generator=generator
        )
    
    image = result.images[0]
    generation_time = time.time() - start_time
    
    # Compute CLIP score if requested
    clip_score = None
    clip_computation_time = None
    if compute_clip and clip_model is not None:
        clip_score, clip_computation_time = compute_clip_score(image, prompt)
        print(f"CLIP Score: {clip_score:.4f}, Time: {clip_computation_time:.2f}s")
    
    # Convert image to base64
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    buffer.seek(0)
    image_base64 = base64.b64encode(buffer.getvalue()).decode()
    
    print(f"Generated image in {generation_time:.2f}s")
    
    return {
        "type": "image",
        "imageBase64": image_base64,
        "seed": seed,
        "seed_index": seed_index,
        "guidance_index": guidance_index,
        "guidance_scale": guidance_scale,
        "generationTime": round(generation_time, 2),
        "clipScore": round(clip_score, 4) if clip_score is not None else None,
        "clipComputationTime": round(clip_computation_time, 2) if clip_computation_time is not None else None
    }

# --- FastAPI App ---
app = FastAPI(title="Stable Diffusion API")

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.on_event("startup")
async def startup_event():
    """Load models on startup"""
    global pipeline, clip_model, clip_processor
    
    print("\n🔍 Initializing models...")
    
    # Load Stable Diffusion model
    pipeline = load_model()
    
    # Load CLIP model
    print("\nLoading CLIP model for scoring...")
    clip_model = CLIPModel.from_pretrained(CLIP_MODEL_ID).to("cuda")
    clip_processor = CLIPProcessor.from_pretrained(CLIP_MODEL_ID)
    print("✅ CLIP model loaded")
    
    print("\n🚀 Server ready!\n")

@app.get("/")
async def root():
    return {"status": "ok", "message": "Stable Diffusion API is running"}

@app.post("/generate")
async def generate_images(request: GenerateRequest):
    """
    Generate images based on the request parameters.
    Returns a Server-Sent Events stream with images as they're generated.
    """
    if pipeline is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    # Calculate guidance scale values
    if request.num_guidance_samples == 1:
        guidance_scales = [round(request.guidance_scale_min * 2) / 2]
    else:
        step = (request.guidance_scale_max - request.guidance_scale_min) / (request.num_guidance_samples - 1)
        guidance_scales = [
            round((request.guidance_scale_min + step * i) * 2) / 2
            for i in range(request.num_guidance_samples)
        ]
    
    async def event_generator():
        """Generate SSE events as images are created"""
        import json
        generation_times = []
        total_images = len(request.seeds) * len(guidance_scales)
        
        print(f"\n🎨 Generating {total_images} images sequentially...")
        
        completed_count = 0
        
        # Generate images one by one
        for seed_idx, seed in enumerate(request.seeds):
            for guidance_idx, guidance_scale in enumerate(guidance_scales):
                # Generate in executor to avoid blocking
                loop = asyncio.get_event_loop()
                result = await loop.run_in_executor(
                    None,
                    generate_single_image,
                    request.prompt,
                    request.negative_prompt,
                    guidance_scale,
                    seed,
                    request.num_steps,
                    seed_idx,
                    guidance_idx,
                    request.compute_clip_score
                )
                
                completed_count += 1
                
                # Collect generation time
                if result.get("generationTime") is not None:
                    generation_times.append(result["generationTime"])
                
                print(f"✓ Completed {completed_count}/{total_images}")
                
                # Send image data
                yield f"data: {json.dumps(result)}\n\n"
        
        # Calculate statistics
        total_time = sum(generation_times) if generation_times else 0
        avg_time = total_time / len(generation_times) if generation_times else 0
        
        completion_data = {
            "type": "complete",
            "totalLatency": round(total_time, 2),
            "averageLatency": round(avg_time, 2)
        }
        yield f"data: {json.dumps(completion_data)}\n\n"
        print(f"\n✅ Complete! Total: {total_time:.2f}s, Average: {avg_time:.2f}s\n")
    
    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
        }
    )

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "model_loaded": pipeline is not None,
        "clip_loaded": clip_model is not None,
        "cuda_available": torch.cuda.is_available()
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)