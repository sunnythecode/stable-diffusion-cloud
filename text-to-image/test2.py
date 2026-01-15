import torch
import torch_tensorrt
from diffusers import StableDiffusionPipeline

pipe = StableDiffusionPipeline.from_pretrained(
    "Manojb/stable-diffusion-2-1-base",
    torch_dtype=torch.float16
).to("cuda")

# Compile UNet with weight-only 8-bit quantization
unet_trt = torch_tensorrt.compile(
    pipe.unet,
    inputs=[torch_tensorrt.Input((1, 4, 64, 64), dtype=torch.half)],
    enabled_precisions={torch.half},   # FP16 activation
    weight_precision=torch.int8,       # W8
    workspace_size=1<<28,
)
pipe.unet = unet_trt

# Generate
image = pipe("A serene lake at sunset", num_inference_steps=25).images[0]
image.show()
