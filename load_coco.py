import os
import json
import csv
import shutil
import urllib.request
import zipfile
from tqdm import tqdm

class DownloadProgressBar(tqdm):
    """Progress bar for download"""
    def update_to(self, b=1, bsize=1, tsize=None):
        if tsize is not None:
            self.total = tsize
        self.update(b * bsize - self.n)

def download_url(url, output_path):
    """Download file with progress bar"""
    with DownloadProgressBar(unit='B', unit_scale=True, miniters=1, desc=url.split('/')[-1]) as t:
        urllib.request.urlretrieve(url, filename=output_path, reporthook=t.update_to)

def extract_zip(zip_path, extract_to):
    """Extract zip file with progress bar"""
    print(f"Extracting {zip_path}...")
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(extract_to)
    print(f"Extraction complete!")

def download_coco_val(output_dir='./coco_data'):
    """
    Download COCO 2017 validation set with annotations
    
    Args:
        output_dir: Directory to save the dataset
    
    Returns:
        Tuple of (images_path, annotations_path)
    """
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # COCO 2017 validation images URL
    val_images_url = "http://images.cocodataset.org/zips/val2017.zip"
    annotations_url = "http://images.cocodataset.org/annotations/annotations_trainval2017.zip"
    
    val_zip_path = os.path.join(output_dir, "val2017.zip")
    annotations_zip_path = os.path.join(output_dir, "annotations_trainval2017.zip")
    
    # Download validation images
    if not os.path.exists(val_zip_path):
        print("Downloading COCO 2017 validation images (~1GB)...")
        download_url(val_images_url, val_zip_path)
        print("Download complete!")
    else:
        print(f"Validation images already downloaded at {val_zip_path}")
    
    # Extract validation images
    val_extract_path = os.path.join(output_dir, "val2017")
    if not os.path.exists(val_extract_path):
        extract_zip(val_zip_path, output_dir)
    else:
        print(f"Validation images already extracted at {val_extract_path}")
    
    # Download annotations
    if not os.path.exists(annotations_zip_path):
        print("Downloading COCO annotations (~250MB)...")
        download_url(annotations_url, annotations_zip_path)
        print("Download complete!")
    else:
        print(f"Annotations already downloaded at {annotations_zip_path}")
    
    # Extract annotations
    annotations_path = os.path.join(output_dir, "annotations")
    if not os.path.exists(annotations_path):
        extract_zip(annotations_zip_path, output_dir)
    else:
        print(f"Annotations already extracted at {annotations_path}")
    
    print(f"\nCOCO validation set ready!")
    print(f"Images: {val_extract_path}")
    print(f"Annotations: {annotations_path}")
    print(f"Total images: {len(os.listdir(val_extract_path))}")
    
    return val_extract_path, annotations_path

def export_coco_captions_to_csv(
    annotations_path,
    images_path,
    output_csv='coco_val_captions.csv',
    output_images_dir='coco_val_images',
    split='val2017',
    use_first_caption_only=True,
    copy_images=True
):
    """
    Export COCO captions to CSV file and copy images to output directory
    
    Args:
        annotations_path: Path to COCO annotations directory
        images_path: Path to COCO images directory (e.g., './coco_data/val2017')
        output_csv: Output CSV file path
        output_images_dir: Directory to copy images to
        split: 'val2017' or 'train2017'
        use_first_caption_only: If True, export only first caption per image.
                                If False, export all captions (one row per caption)
        copy_images: Whether to copy images to output directory
    """
    
    caption_file = os.path.join(annotations_path, f'captions_{split}.json')
    
    print(f"Loading captions from {caption_file}...")
    with open(caption_file, 'r') as f:
        coco_data = json.load(f)
    
    # Create mappings
    image_id_to_info = {img['id']: img for img in coco_data['images']}
    
    # Organize captions by image
    image_captions = {}
    for ann in coco_data['annotations']:
        image_id = ann['image_id']
        caption = ann['caption']
        
        if image_id not in image_captions:
            image_captions[image_id] = []
        image_captions[image_id].append(caption)
    
    # Create output images directory
    if copy_images:
        os.makedirs(output_images_dir, exist_ok=True)
        print(f"Copying images to {output_images_dir}...")
    
    # Write to CSV
    print(f"Writing captions to {output_csv}...")
    with open(output_csv, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        
        if use_first_caption_only:
            # One row per image
            writer.writerow(['image_id', 'filename', 'caption'])
            
            for image_id, captions in tqdm(image_captions.items()):
                img_info = image_id_to_info[image_id]
                filename = img_info['file_name']
                caption = captions[0]  # First caption only
                
                writer.writerow([image_id, filename, caption])
                
                # Copy image
                if copy_images:
                    src = os.path.join(images_path, filename)
                    dst = os.path.join(output_images_dir, filename)
                    if os.path.exists(src) and not os.path.exists(dst):
                        shutil.copy2(src, dst)
        else:
            # One row per caption (multiple rows per image)
            writer.writerow(['image_id', 'filename', 'caption', 'caption_index'])
            
            copied_images = set()
            for image_id, captions in tqdm(image_captions.items()):
                img_info = image_id_to_info[image_id]
                filename = img_info['file_name']
                
                for idx, caption in enumerate(captions):
                    writer.writerow([image_id, filename, caption, idx])
                
                # Copy image once per image (not per caption)
                if copy_images and filename not in copied_images:
                    src = os.path.join(images_path, filename)
                    dst = os.path.join(output_images_dir, filename)
                    if os.path.exists(src) and not os.path.exists(dst):
                        shutil.copy2(src, dst)
                    copied_images.add(filename)
    
    num_rows = len(image_captions) if use_first_caption_only else sum(len(v) for v in image_captions.values())
    print(f"\nExported {num_rows} rows to {output_csv}")
    print(f"Total unique images: {len(image_captions)}")
    if copy_images:
        print(f"Copied {len(image_captions)} images to {output_images_dir}")
    
    # Show sample
    print("\nFirst 5 rows preview:")
    with open(output_csv, 'r', encoding='utf-8') as f:
        reader = csv.reader(f)
        for i, row in enumerate(reader):
            if i >= 6:  # Header + 5 rows
                break
            print(f"  {row}")
    
    return output_csv, output_images_dir

if __name__ == "__main__":
    # Download COCO dataset
    print("="*60)
    print("Step 1: Downloading COCO dataset")
    print("="*60)
    images_path, annotations_path = download_coco_val('./coco_data')
    
    print("\n" + "="*60)
    print("Step 2: Exporting captions and copying images")
    print("="*60)
    
    # Export with first caption only (5000 rows - one per image)
    # AND copy all images to output directory
    export_coco_captions_to_csv(
        annotations_path=annotations_path,
        images_path=images_path,
        output_csv='coco_val_captions.csv',
        output_images_dir='coco_val_images',
        use_first_caption_only=True,
        copy_images=True
    )
    
    print("\n" + "="*60)
    print("ALL DONE!")
    print("="*60)
    print(f"CSV file: coco_val_captions.csv")
    print(f"Images directory: coco_val_images/")
    print("="*60)
    
    # Uncomment to export all captions (~25000 rows - all caption variations)
    # export_coco_captions_to_csv(
    #     annotations_path=annotations_path,
    #     images_path=images_path,
    #     output_csv='coco_val_all_captions.csv',
    #     output_images_dir='coco_val_images',
    #     use_first_caption_only=False,
    #     copy_images=True
    # )