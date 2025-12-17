from PIL import Image
from pathlib import Path
import os

def check_images(directory):
    """
    Check all images in a directory for corruption or issues.
    
    Args:
        directory: Path to image directory
    """
    extensions = ['*.jpg', '*.jpeg', '*.png', '*.JPG', '*.JPEG', '*.PNG']
    image_files = []
    for ext in extensions:
        image_files.extend(Path(directory).glob(ext))
    
    print(f"Checking {len(image_files)} images in {directory}...")
    
    corrupted = []
    sizes = {}
    
    for i, img_path in enumerate(image_files):
        try:
            with Image.open(img_path) as img:
                img.verify()  # Check if image is corrupted
                
            # Reopen to get size (verify() closes the file)
            with Image.open(img_path) as img:
                size = img.size
                if size not in sizes:
                    sizes[size] = []
                sizes[size].append(str(img_path))
                
        except Exception as e:
            corrupted.append((str(img_path), str(e)))
            print(f"✗ Corrupted: {img_path.name} - {e}")
        
        if (i + 1) % 100 == 0:
            print(f"  Checked {i + 1}/{len(image_files)} images...")
    
    print(f"\n{'='*60}")
    print(f"Results for {directory}:")
    print(f"{'='*60}")
    print(f"Total images: {len(image_files)}")
    print(f"Corrupted images: {len(corrupted)}")
    print(f"Unique image sizes: {len(sizes)}")
    
    if sizes:
        print(f"\nImage size distribution:")
        for size, files in sorted(sizes.items(), key=lambda x: len(x[1]), reverse=True)[:10]:
            print(f"  {size}: {len(files)} images")
    
    if corrupted:
        print(f"\nCorrupted images:")
        for path, error in corrupted[:10]:  # Show first 10
            print(f"  {path}: {error}")
        if len(corrupted) > 10:
            print(f"  ... and {len(corrupted) - 10} more")
    
    return corrupted, sizes


def remove_corrupted_images(directory, dry_run=True):
    """
    Remove corrupted images from directory.
    
    Args:
        directory: Path to image directory
        dry_run: If True, only print what would be removed
    """
    corrupted, _ = check_images(directory)
    
    if not corrupted:
        print("\n✓ No corrupted images found!")
        return
    
    print(f"\n{'='*60}")
    if dry_run:
        print(f"DRY RUN: Would remove {len(corrupted)} corrupted images")
        print("Set dry_run=False to actually delete them")
    else:
        print(f"Removing {len(corrupted)} corrupted images...")
        for img_path, _ in corrupted:
            try:
                os.remove(img_path)
                print(f"  Removed: {img_path}")
            except Exception as e:
                print(f"  Failed to remove {img_path}: {e}")
        print("✓ Done!")


if __name__ == "__main__":
    # Check both directories
    REAL_IMAGES = "/home/sandeep_b/qualcomm_sd/fid/dataset"
    GENERATED_IMAGES = "/home/sandeep_b/qualcomm_sd/fid/real"
    
    print("Checking LAION images...")
    check_images(REAL_IMAGES)
    
    print("\n" + "="*60 + "\n")
    
    print("Checking SD-generated images...")
    check_images(GENERATED_IMAGES)
    
    # To remove corrupted images (use with caution!):
    # remove_corrupted_images(REAL_IMAGES, dry_run=True)
    # remove_corrupted_images(GENERATED_IMAGES, dry_run=True)