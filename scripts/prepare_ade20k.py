#!/usr/bin/env python3
"""
Dataset preparation script for ADE20K.

Downloads and prepares the ADE20K dataset for training.

Usage:
    python scripts/prepare_ade20k.py --data_dir data/ade20k
"""

import argparse
import os
import urllib.request
import zipfile
from pathlib import Path


def download_file(url: str, filepath: str):
    """Download file with progress bar."""
    def progress_hook(block_num, block_size, total_size):
        downloaded = block_num * block_size
        percent = min(downloaded / total_size * 100, 100)
        print(f"\rDownloading: {percent:.1f}%", end="", flush=True)
    
    urllib.request.urlretrieve(url, filepath, progress_hook)
    print()  # New line after progress


def extract_zip(zip_path: str, extract_to: str):
    """Extract zip file."""
    print(f"Extracting {zip_path}...")
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(extract_to)
    print("Extraction complete.")


def prepare_ade20k(data_dir: str):
    """Download and prepare ADE20K dataset."""
    data_dir = Path(data_dir)
    data_dir.mkdir(parents=True, exist_ok=True)
    
    # URLs for ADE20K dataset
    dataset_url = "http://data.csail.mit.edu/places/ADEchallenge/ADEChallengeData2016.zip"
    zip_path = data_dir / "ADEChallengeData2016.zip"
    
    # Download if not exists
    if not zip_path.exists():
        print(f"Downloading ADE20K dataset to {zip_path}")
        download_file(dataset_url, str(zip_path))
    else:
        print(f"Dataset already downloaded: {zip_path}")
    
    # Extract if not already extracted
    extracted_dir = data_dir / "ADEChallengeData2016"
    if not extracted_dir.exists():
        extract_zip(str(zip_path), str(data_dir))
    else:
        print(f"Dataset already extracted: {extracted_dir}")
    
    # Reorganize directory structure for easier access
    images_dir = data_dir / "images"
    annotations_dir = data_dir / "annotations"
    
    if not images_dir.exists():
        print("Reorganizing directory structure...")
        
        # Create organized structure
        images_dir.mkdir()
        annotations_dir.mkdir()
        
        # Move training data
        src_train_img = extracted_dir / "images" / "training"
        src_train_ann = extracted_dir / "annotations" / "training"
        
        if src_train_img.exists():
            (images_dir / "training").mkdir()
            os.system(f"cp -r {src_train_img}/* {images_dir / 'training'}/")
        
        if src_train_ann.exists():
            (annotations_dir / "training").mkdir()
            os.system(f"cp -r {src_train_ann}/* {annotations_dir / 'training'}/")
        
        # Move validation data
        src_val_img = extracted_dir / "images" / "validation"
        src_val_ann = extracted_dir / "annotations" / "validation"
        
        if src_val_img.exists():
            (images_dir / "validation").mkdir()
            os.system(f"cp -r {src_val_img}/* {images_dir / 'validation'}/")
        
        if src_val_ann.exists():
            (annotations_dir / "validation").mkdir()
            os.system(f"cp -r {src_val_ann}/* {annotations_dir / 'validation'}/")
        
        print("Directory reorganization complete.")
    
    # Print dataset statistics
    print("\nDataset Statistics:")
    for split in ["training", "validation"]:
        img_dir = images_dir / split
        ann_dir = annotations_dir / split
        
        if img_dir.exists():
            num_images = len(list(img_dir.glob("*.jpg")))
            print(f"  {split}: {num_images} images")
        
        if ann_dir.exists():
            num_annotations = len(list(ann_dir.glob("*.png")))
            print(f"  {split}: {num_annotations} annotations")
    
    print(f"\nDataset prepared successfully at {data_dir}")
    print("You can now run training with:")
    print(f"python cli.py fit --config configs/ade20k.yaml --data.data_dir {data_dir}")


def main():
    parser = argparse.ArgumentParser(description="Prepare ADE20K dataset")
    parser.add_argument("--data_dir", default="data/ade20k", 
                       help="Directory to download and extract dataset")
    
    args = parser.parse_args()
    prepare_ade20k(args.data_dir)


if __name__ == "__main__":
    main()
