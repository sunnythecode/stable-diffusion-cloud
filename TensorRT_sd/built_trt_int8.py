# build_trt_engine_with_calibration.py
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit
import numpy as np
from diffusers import StableDiffusionPipeline
import torch

class UNetCalibrator(trt.IInt8EntropyCalibrator2):
    def __init__(self, calibration_data, cache_file="unet_calibration.cache"):
        trt.IInt8EntropyCalibrator2.__init__(self)
        
        self.calibration_data = calibration_data
        self.cache_file = cache_file
        self.current_index = 0
        
        # Allocate device memory for calibration
        self.device_input = {}
        for name, data in calibration_data[0].items():
            self.device_input[name] = cuda.mem_alloc(data.nbytes)
    
    def get_batch_size(self):
        return 1
    
    def get_batch(self, names):
        if self.current_index < len(self.calibration_data):
            batch = self.calibration_data[self.current_index]
            self.current_index += 1
            
            # Copy data to device
            for name in names:
                cuda.memcpy_htod(self.device_input[name], batch[name])
            
            return [int(self.device_input[name]) for name in names]
        else:
            return None
    
    def read_calibration_cache(self):
        try:
            with open(self.cache_file, "rb") as f:
                return f.read()
        except:
            return None
    
    def write_calibration_cache(self, cache):
        with open(self.cache_file, "wb") as f:
            f.write(cache)

def generate_calibration_data(num_samples=100):
    """Generate calibration data by running forward passes"""
    print("Generating calibration data...")
    
    model_id = "Manojb/stable-diffusion-2-1-base"
    pipe = StableDiffusionPipeline.from_pretrained(
        model_id,
        torch_dtype=torch.float16
    ).to("cuda")
    
    calibration_data = []
    
    # Generate diverse samples
    prompts = [
        "a photo of a cat",
        "a landscape painting",
        "portrait of a person",
        "abstract art",
        "still life with fruits",
        "cityscape at night",
        "underwater scene",
        "mountain vista"
    ]
    
    for i in range(num_samples):
        prompt_idx = i % len(prompts)
        
        # Encode prompt - BATCH SIZE 2 for classifier-free guidance
        text_inputs = pipe.tokenizer(
            [prompts[prompt_idx]] * 2,  # Duplicate for batch size 2
            padding="max_length",
            max_length=77,
            truncation=True,
            return_tensors="pt"
        )
        
        with torch.no_grad():
            text_embeddings = pipe.text_encoder(text_inputs.input_ids.to("cuda"))[0]
        
        # Random latents and timestep - MATCHING BATCH SIZE
        sample = torch.randn(2, 4, 64, 64, dtype=torch.float16).cuda()
        timestep = torch.randint(0, 1000, (1,), dtype=torch.long).cuda()
        
        # Run forward pass to get realistic activations
        with torch.no_grad():
            _ = pipe.unet(sample, timestep, text_embeddings)
        
        # Store as numpy arrays
        calibration_data.append({
            "sample": np.ascontiguousarray(sample.cpu().numpy().astype(np.float16)),
            "timestep": np.ascontiguousarray(timestep.cpu().numpy().astype(np.int64)),
            "encoder_hidden_states": np.ascontiguousarray(text_embeddings.cpu().numpy().astype(np.float16))
        })
        
        if (i + 1) % 20 == 0:
            print(f"  Generated {i + 1}/{num_samples} samples")
    
    print(f"✓ Generated {len(calibration_data)} calibration samples")
    return calibration_data

def build_engine_with_calibration(onnx_file, engine_file, calibration_data):
    """Build TensorRT engine with INT8 calibration"""
    
    TRT_LOGGER = trt.Logger(trt.Logger.INFO)
    builder = trt.Builder(TRT_LOGGER)
    network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
    parser = trt.OnnxParser(network, TRT_LOGGER)
    
    # Parse ONNX
    print(f"Loading ONNX file: {onnx_file}")
    with open(onnx_file, 'rb') as model:
        if not parser.parse(model.read()):
            for error in range(parser.num_errors):
                print(parser.get_error(error))
            return None
    
    print("✓ ONNX file parsed")
    
    # Configure builder
    config = builder.create_builder_config()
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 6 * (1 << 30))
    
    # Enable FP16 and INT8
    config.set_flag(trt.BuilderFlag.FP16)
    config.set_flag(trt.BuilderFlag.INT8)
    
    # Set calibrator
    calibrator = UNetCalibrator(calibration_data)
    config.int8_calibrator = calibrator
    
    print("✓ Calibrator configured")
    
    # Set optimization profile
    profile = builder.create_optimization_profile()
    profile.set_shape("sample", (1, 4, 64, 64), (2, 4, 64, 64), (4, 4, 64, 64))
    profile.set_shape("timestep", (1,), (1,), (1,))
    profile.set_shape("encoder_hidden_states", (1, 77, 1024), (2, 77, 1024), (4, 77, 1024))
    config.add_optimization_profile(profile)
    
    # Build engine
    print("Building engine with INT8 calibration...")
    print("This will take 10-20 minutes. Progress:")
    config.builder_optimization_level = 5
    
    serialized_engine = builder.build_serialized_network(network, config)
    
    if serialized_engine is None:
        print("ERROR: Failed to build engine")
        return None
    
    # Save engine
    print(f"Saving engine to: {engine_file}")
    with open(engine_file, 'wb') as f:
        f.write(serialized_engine)
    
    import os
    size = os.path.getsize(engine_file) / (1024**2)
    print(f"✓ Engine built successfully: {size:.2f} MB")
    
    return serialized_engine

if __name__ == "__main__":
    # Generate calibration data
    calibration_data = generate_calibration_data(num_samples=100)
    
    # Build engine with calibration
    build_engine_with_calibration(
        "unet_sd21.onnx",
        "unet_w8a16_calibrated.trt",
        calibration_data
    )