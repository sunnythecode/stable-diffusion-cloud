
# import numpy as np
import pycuda.driver as cuda
# import pycuda.autoinit
import tensorrt as trt
import torch
# FORCE PYTORCH TO INITIALIZE CUDA FIRST
dummy = torch.zeros(1, device="cuda")
from diffusers import DDIMScheduler
from transformers import CLIPTokenizer
import numpy as np
from PIL import Image


print("imports")

class TRTModel:
    def __init__(self, engine_path):
        self.logger = trt.Logger(trt.Logger.ERROR)
        with open(engine_path, "rb") as f:
            self.engine = trt.Runtime(self.logger).deserialize_cuda_engine(f.read())

        
        # self.context = self.engine.create_execution_context()
        self.stream = cuda.Stream()
        self.context = self.engine.create_execution_context()
        self.context.set_optimization_profile_async(0, self.stream.handle)

    def __call__(self, feed_dict):
        for name, tensor in feed_dict.items():
            self.context.set_input_shape(name, tensor.shape)
            self.context.set_tensor_address(name, tensor.data_ptr())

        # Ensure shapes are updated before querying output shapes
        if not self.context.all_binding_shapes_specified:
            raise RuntimeError("Missing input shapes!")

        outputs = {}
        for i in range(self.engine.num_io_tensors):
            name = self.engine.get_tensor_name(i)
            if self.engine.get_tensor_mode(name) == trt.TensorIOMode.OUTPUT:
                shape = self.context.get_tensor_shape(name)
                # Defensive check for the Myelin 's53' error
                if any(s <= 0 for s in shape):
                    raise ValueError(f"Engine returned invalid shape {shape} for {name}")
                
                out_tensor = torch.empty(tuple(shape), dtype=torch.float16, device="cuda")
                outputs[name] = out_tensor
                self.context.set_tensor_address(name, out_tensor.data_ptr())

        self.context.execute_async_v3(self.stream.handle)
        self.stream.synchronize()
        return outputs
def save_image(image_array, filename="output.png"):
    """
    image_array: Numpy array in shape (Height, Width, 3) 
                 with values scaled 0-255.
    """
    # Ensure the data is in 8-bit integer format
    image_data = image_array.astype(np.uint8)
    
    # Create a PIL Image object
    img = Image.fromarray(image_data)
    
    # Save to disk
    img.save(filename)
    print(f"✅ Image saved successfully as {filename}")




# --- 1. SETUP ---
prompt = "a cow"
num_inference_steps = 50
guidance_scale = 7.5

MODEL_ID = "Manojb/stable-diffusion-2-1-base"

# Load standard helpers
tokenizer = CLIPTokenizer.from_pretrained(MODEL_ID, subfolder="tokenizer")
scheduler = DDIMScheduler.from_pretrained(MODEL_ID, subfolder="scheduler")
scheduler.config.prediction_type = "epsilon"

print("tokenizers")

# Load your custom TRT Engines
txt_enc = TRTModel("text_encoder.plan")
unet = TRTModel("unet_w8a16.plan")
vae_dec = TRTModel("vae_decoder.plan")

print('.plan loading')

# --- 2. TEXT ENCODING ---
text_input = tokenizer(prompt, padding="max_length", max_length=tokenizer.model_max_length, truncation=True, return_tensors="pt")

# Use 'output' as discovered in your debug print
txt_enc_out = txt_enc({"input_ids": text_input.input_ids.to("cuda").int()})
cond = txt_enc_out["output"] 

uncond_input = tokenizer([""], padding="max_length", max_length=tokenizer.model_max_length, return_tensors="pt")
uncond = txt_enc({"input_ids": uncond_input.input_ids.to("cuda").int()})["output"]

embeddings = torch.cat([uncond, cond])

# --- 3. LATENT INITIALIZATION ---
latents = torch.randn((1, 4, 64, 64), device="cuda", dtype=torch.float16)
scheduler.set_timesteps(num_inference_steps)
latents = latents * scheduler.init_noise_sigma

print("UNet timestep TRT dtype:", unet.engine.get_tensor_dtype("timestep"))


# --- 4. DENOISING LOOP ---
for t in scheduler.timesteps:
    latent_model_input = torch.cat([latents] * 2)
    latent_model_input = scheduler.scale_model_input(latent_model_input, t)

    timestep_tensor = torch.tensor([t, t], device="cuda", dtype=torch.float16)
    
    unet_output = unet({
        "sample": latent_model_input, 
        "timestep": timestep_tensor, 
        "encoder_hidden_states": embeddings
    })
    
    # Robust check for UNet output name
    unet_key = "out_sample" if "out_sample" in unet_output else list(unet_output.keys())[0]
    noise_pred = unet_output[unet_key]

    noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
    noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)

    latents = scheduler.step(noise_pred, t, latents).prev_sample

    if int(t) == int(scheduler.timesteps[0]) or int(t) == int(scheduler.timesteps[1]):
        print("t:", int(t),"latent std:", latents.float().std().item(), "noise_pred std:", noise_pred.float().std().item())

# --- 5. VAE DECODING ---
# Scale back the latents
# latents = 1 / 0.18215 * latents


print("latents pre-vae:  min/max/std",
      latents.min().item(), latents.max().item(), latents.float().std().item())

vae_output = vae_dec({"latent_sample": latents})
image = vae_output[list(vae_output.keys())[0]]

print("vae raw:         min/max/std",
      image.min().item(), image.max().item(), image.float().std().item())


breakpoint()

# --- 6. IMPROVED POST-PROCESSING ---
# SD 2.1 outputs are often in the range [-1, 1]
# We need to map them to [0, 1] correctly
image = (image / 2 + 0.5).clamp(0, 1)

# Convert from (Batch, C, H, W) to (H, W, C)
image = image.cpu().permute(0, 2, 3, 1).float().numpy()

# Use the first image in the batch
image_np = (image[0] * 255).round().astype(np.uint8)
save_image(image_np, "generated_art.png")