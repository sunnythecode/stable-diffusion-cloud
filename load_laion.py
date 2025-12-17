from datasets import load_dataset
import requests
from PIL import Image
from io import BytesIO
import pandas as pd
from pathlib import Path

# Setup
output_dir = Path("/home/sandeep_b/qualcomm_sd/fid/gen")
images_dir = output_dir
images_dir.mkdir(parents=True, exist_ok=True)

# Load dataset
dataset = load_dataset(
    "laion/laion2B-en-aesthetic",
    split="train",
    streaming=True
)

# Store metadata
metadata_rows = []
successful = 0
failed = 0

print("Downloading images and collecting metadata...")

for example in dataset:
    if successful >= 1000:  # Change to 1000 if you only want 1000 images
        break
    
    try:
        # Download image
        response = requests.get(example['URL'], timeout=5)
        img = Image.open(BytesIO(response.content))
        
        # Convert to RGB if needed
        if img.mode != 'RGB':
            img = img.convert('RGB')
        
        # Save image with zero-padded index
        img_filename = f"image_{successful:05d}.png"
        img.save(images_dir / img_filename)
        
        # Collect all metadata from the example
        metadata_row = {
            'index': successful,
            'filename': img_filename,
            **example  # Add all fields from the dataset
        }
        metadata_rows.append(metadata_row)
        
        successful += 1
        
        if successful % 20 == 0:
            print(f"Progress: {successful} images saved, {failed} failed")
            
    except Exception as e:
        failed += 1
        continue

# Save metadata to CSV
df = pd.DataFrame(metadata_rows)
op = Path("/home/sandeep_b/qualcomm_sd/fid")
df.to_csv(op / "metadata.csv", index=False)

print(f"\nComplete!")
print(f"Successfully saved: {successful} images")
print(f"Failed downloads: {failed}")
print(f"Images saved to: {images_dir}")
print(f"Metadata saved to: {output_dir / 'metadata.csv'}")
print(f"\nCSV columns: {list(df.columns)}")