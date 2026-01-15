# trt_unet.py
import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit
import numpy as np
import torch

class TRTUNet:
    def __init__(self, engine_path):
        """Initialize TensorRT UNet"""
        self.logger = trt.Logger(trt.Logger.WARNING)
        
        # Load engine
        print(f"Loading TensorRT engine: {engine_path}")
        with open(engine_path, 'rb') as f:
            runtime = trt.Runtime(self.logger)
            self.engine = runtime.deserialize_cuda_engine(f.read())
        
        if self.engine is None:
            raise RuntimeError("Failed to load TensorRT engine")
        
        self.context = self.engine.create_execution_context()
        self.stream = cuda.Stream()
        
        print(f"✓ Engine loaded successfully")
        
        # Print engine info
        for i in range(self.engine.num_io_tensors):
            name = self.engine.get_tensor_name(i)
            mode = self.engine.get_tensor_mode(name)
            shape = self.engine.get_tensor_shape(name)
            dtype = self.engine.get_tensor_dtype(name)
            print(f"  {mode.name}: {name} {shape} {dtype}")
    
    def infer(self, sample, timestep, encoder_hidden_states):
        """Run inference with TensorRT engine"""
        batch_size = sample.shape[0]
        
        # Set input shapes dynamically
        self.context.set_input_shape("sample", sample.shape)
        self.context.set_input_shape("timestep", timestep.shape)
        self.context.set_input_shape("encoder_hidden_states", encoder_hidden_states.shape)
        
        # Get output shape after setting input shapes
        output_shape = self.context.get_tensor_shape("out_sample")
        
        # Allocate device memory
        d_sample = cuda.mem_alloc(sample.nbytes)
        d_timestep = cuda.mem_alloc(timestep.nbytes)
        d_encoder_hidden_states = cuda.mem_alloc(encoder_hidden_states.nbytes)
        
        output = np.empty(output_shape, dtype=np.float16)
        d_output = cuda.mem_alloc(output.nbytes)
        
        # Copy inputs to device
        cuda.memcpy_htod_async(d_sample, sample, self.stream)
        cuda.memcpy_htod_async(d_timestep, timestep, self.stream)
        cuda.memcpy_htod_async(d_encoder_hidden_states, encoder_hidden_states, self.stream)
        
        # Set tensor addresses
        self.context.set_tensor_address("sample", int(d_sample))
        self.context.set_tensor_address("timestep", int(d_timestep))
        self.context.set_tensor_address("encoder_hidden_states", int(d_encoder_hidden_states))
        self.context.set_tensor_address("out_sample", int(d_output))
        
        # Execute inference
        self.context.execute_async_v3(stream_handle=self.stream.handle)
        
        # Copy output back to host
        cuda.memcpy_dtoh_async(output, d_output, self.stream)
        self.stream.synchronize()
        
        return torch.from_numpy(output).cuda()

# Test the TRT engine
if __name__ == "__main__":
    print("Testing TensorRT UNet...")
    trt_unet = TRTUNet("unet_w8a16_calibrated.trt")
    
    # Create dummy inputs
    sample = np.random.randn(2, 4, 64, 64).astype(np.float16)
    timestep = np.array([1], dtype=np.int64)
    encoder_hidden_states = np.random.randn(2, 77, 1024).astype(np.float16)
    
    print("\nRunning test inference...")
    output = trt_unet.infer(sample, timestep, encoder_hidden_states)
    print(f"✓ Inference successful! Output shape: {output.shape}")