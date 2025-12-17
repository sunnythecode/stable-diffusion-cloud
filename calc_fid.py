import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1,2,3,4,5,6,7"
from cleanfid import fid


def calculate_fid_cleanfid(real_images_path, generated_images_path, 
                           mode="clean", device="cuda", batch_size=32):
    """
    Calculate FID using clean-fid library (more robust for varied image sizes).
    
    Args:
        real_images_path: Path to real images directory
        generated_images_path: Path to generated images directory
        mode: "clean" (recommended) or "legacy" 
        device: "cuda" or "cpu"
        batch_size: Batch size for processing
    
    Returns:
        FID score (float)
    """
    
    # Verify directories exist
    if not os.path.exists(real_images_path):
        raise ValueError(f"Real images directory not found: {real_images_path}")
    if not os.path.exists(generated_images_path):
        raise ValueError(f"Generated images directory not found: {generated_images_path}")
    
    print(f"Calculating FID score using clean-fid...")
    print(f"Mode: {mode}")
    print(f"Device: {device}")
    
    # Calculate FID
    score = fid.compute_fid(
        real_images_path,
        generated_images_path,
        mode=mode,
        device=device,
        batch_size=batch_size,
        num_workers=0  # Avoid multiprocessing issues
    )
    
    return score


if __name__ == "__main__":
    REAL_IMAGES = "/home/sandeep_b/qualcomm_sd/coco_val_images"
    GENERATED_IMAGES = "/home/sandeep_b/qualcomm_sd/coco_gen_images_12345"
    
    try:
        fid_score = calculate_fid_cleanfid(
            REAL_IMAGES,
            GENERATED_IMAGES,
            mode="clean",  # "clean" is more accurate, "legacy" matches old pytorch-fid
            device="cuda",  # or "cpu"
            batch_size=32
        )
        
        print(f"\n{'='*60}")
        print(f"FID Score: {fid_score:.4f}")
        print(f"{'='*60}")
        
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()

        # run 1: FID Score: 39.3448