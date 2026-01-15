# build_trt_fp16.py
import os
# os.environ["CUDA_VISIBLE_DEVICES"] = "1"
import tensorrt as trt


def build_engine_fp16(onnx_file, engine_file):
    """Build TensorRT engine with FP16 only"""
    
    TRT_LOGGER = trt.Logger(trt.Logger.INFO)
    builder = trt.Builder(TRT_LOGGER)
    network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
    parser = trt.OnnxParser(network, TRT_LOGGER)
    
    print(f"Loading ONNX file: {onnx_file}")
    with open(onnx_file, 'rb') as model:
        if not parser.parse(model.read()):
            for error in range(parser.num_errors):
                print(parser.get_error(error))
            return None
    
    print("✓ ONNX parsed")
    
    config = builder.create_builder_config()
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 4 * (1 << 30))
    
    # Only FP16 - no INT8
    config.set_flag(trt.BuilderFlag.FP16)
    
    # Dynamic shapes
    profile = builder.create_optimization_profile()
    profile.set_shape("sample", (1, 4, 64, 64), (2, 4, 64, 64), (4, 4, 64, 64))
    profile.set_shape("timestep", (1,), (1,), (1,))
    profile.set_shape("encoder_hidden_states", (1, 77, 1024), (2, 77, 1024), (4, 77, 1024))
    config.add_optimization_profile(profile)
    
    print("Building FP16 engine...")
    config.builder_optimization_level = 5
    
    serialized_engine = builder.build_serialized_network(network, config)
    
    if serialized_engine:
        with open(engine_file, 'wb') as f:
            f.write(serialized_engine)
        
        size = os.path.getsize(engine_file) / (1024**2)
        print(f"✓ FP16 Engine built: {size:.2f} MB")
        return serialized_engine
    else:
        print("ERROR: Failed to build engine")
        return None

if __name__ == "__main__":
    build_engine_fp16("unet_sd21.onnx", "unet_fp16.trt")