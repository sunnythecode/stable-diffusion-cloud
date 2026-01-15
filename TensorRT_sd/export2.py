from diffusers import StableDiffusionPipeline
import torch
import warnings

# Suppress the tracer warning
warnings.filterwarnings('ignore', category=torch.jit.TracerWarning)

model_id = "Manojb/stable-diffusion-2-1-base"
pipe = StableDiffusionPipeline.from_pretrained(
    model_id, 
    torch_dtype=torch.float16,
    use_safetensors=True
)

def export_unet():
    unet = pipe.unet.to("cuda")
    unet.eval()
    
    # Create dummy inputs
    batch_size = 2
    sample = torch.randn(batch_size, 4, 64, 64, dtype=torch.float16).cuda()
    timestep = torch.tensor([1], dtype=torch.long).cuda()
    encoder_hidden_states = torch.randn(batch_size, 77, 1024, dtype=torch.float16).cuda()
    
    with torch.no_grad():
        print("Testing forward pass...")
        output = unet(sample, timestep, encoder_hidden_states)
        print(f"✓ Forward pass successful, output shape: {output.sample.shape}")
        
        print("Exporting to ONNX...")
        torch.onnx.export(
            unet,
            (sample, timestep, encoder_hidden_states),
            "unet_sd21.onnx",
            export_params=True,
            opset_version=17,
            do_constant_folding=True,
            input_names=["sample", "timestep", "encoder_hidden_states"],
            output_names=["out_sample"],
            dynamic_axes={
                "sample": {0: "batch"},
                "encoder_hidden_states": {0: "batch"},
            },
            dynamo=False,
            verbose=False
        )
    
    print("✓ UNet exported to unet_sd21.onnx")
    
    # Verify the ONNX file
    import onnx
    onnx_model = onnx.load("unet_sd21.onnx")
    onnx.checker.check_model(onnx_model)
    print("✓ ONNX model is valid")

if __name__ == "__main__":
    export_unet()