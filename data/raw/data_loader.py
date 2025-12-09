"""
Data Loading and Preprocessing Helper

This module provides functionality to:
1. Download the breast cancer dataset using kagglehub
2. Unzip the dataset if it's compressed
3. Create train/test split with stratified sampling (90/10)
4. Organize data with class-prefixed filenames to avoid naming conflicts

All output folders will be created in the same directory where this script is executed.
"""

import os
import shutil
import zipfile
from pathlib import Path
from collections import Counter
from typing import Tuple, List, Dict

import numpy as np
from sklearn.model_selection import train_test_split


def download_dataset(dataset_slug: str = "amirbnsl/breast-cancer-padded-interpolated-720p") -> str:
    """
    Download the breast cancer dataset using kagglehub.
    
    Args:
        dataset_slug: The kagglehub dataset identifier.
                     Default: "amirbnsl/breast-cancer-padded-interpolated-720p"
    
    Returns:
        Path to the downloaded dataset directory.
    """
    try:
        import kagglehub
        print(f"Downloading dataset: {dataset_slug}...")
        path = kagglehub.dataset_download(dataset_slug)
        print(f"Dataset downloaded to: {path}")
        return path
    except ImportError:
        raise ImportError(
            "kagglehub is not installed. Install it with: pip install kagglehub"
        )
    except Exception as e:
        raise RuntimeError(f"Error downloading dataset: {e}")


def unzip_dataset(zip_path: str, extract_to: str = None) -> str:
    """
    Unzip a dataset if it's compressed.
    
    Args:
        zip_path: Path to the zip file.
        extract_to: Directory to extract files to. If None, extracts to the same directory as the zip.
    
    Returns:
        Path to the extracted dataset directory.
    """
    zip_path = Path(zip_path)
    
    if not zip_path.exists():
        raise FileNotFoundError(f"Zip file not found: {zip_path}")
    
    if not zipfile.is_zipfile(zip_path):
        print(f"File is not a zip archive: {zip_path}")
        return str(zip_path.parent)
    
    if extract_to is None:
        extract_to = zip_path.parent
    else:
        extract_to = Path(extract_to)
        extract_to.mkdir(parents=True, exist_ok=True)
    
    print(f"Unzipping {zip_path} to {extract_to}...")
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(extract_to)
    print(f"Dataset unzipped successfully")
    
    return str(extract_to)


def collect_images(dataset_path: str, class_folders: List[str] = None) -> List[Dict]:
    """
    Collect all image paths and labels from the dataset directory.
    
    Args:
        dataset_path: Path to the dataset root directory.
        class_folders: List of class folder names. 
                      Default: ['0_N', '1_PB', '2_UDH', '3_FEA', '4_ADH', '5_DCIS', '6_IC']
    
    Returns:
        List of dictionaries containing image information (path, filename, label).
    """
    if class_folders is None:
        class_folders = ['0_N', '1_PB', '2_UDH', '3_FEA', '4_ADH', '5_DCIS', '6_IC']
    
    dataset_path = Path(dataset_path)
    image_data = []
    
    for class_folder in class_folders:
        class_path = dataset_path / class_folder
        
        if not class_path.exists():
            print(f"Warning: {class_path} does not exist!")
            continue
        
        # Get all PNG files in the folder
        image_files = [f for f in os.listdir(class_path) if f.endswith('.png')]
        
        for img_file in image_files:
            original_path = class_path / img_file
            # Create new filename: classname_originalname.png
            new_filename = f"{class_folder}_{img_file}"
            image_data.append({
                'original_path': str(original_path),
                'new_filename': new_filename,
                'label': class_folder
            })
        
        print(f"Class {class_folder}: {len(image_files)} images")
    
    print(f"\nTotal images collected: {len(image_data)}")
    return image_data


def create_train_test_split(
    image_data: List[Dict],
    output_dir: str = None,
    test_size: float = 0.1,
    random_state: int = 42
) -> Tuple[str, str]:
    """
    Create train/test split with stratified sampling and organize files.
    
    Args:
        image_data: List of image dictionaries from collect_images().
        output_dir: Directory where to create train/test folders.
                   If None, uses the current working directory.
        test_size: Proportion of dataset to include in test split (default: 0.1 for 90/10).
        random_state: Random seed for reproducibility.
    
    Returns:
        Tuple of (train_dir_path, test_dir_path).
    """
    if output_dir is None:
        output_dir = os.getcwd()
    
    output_dir = Path(output_dir)
    train_dir = output_dir / 'train'
    test_dir = output_dir / 'test'
    
    # Create directories
    train_dir.mkdir(parents=True, exist_ok=True)
    test_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"\nTrain directory: {train_dir}")
    print(f"Test directory: {test_dir}")
    
    # Extract labels for stratified sampling
    labels = [item['label'] for item in image_data]
    
    # Perform stratified split
    train_data, test_data = train_test_split(
        image_data,
        test_size=test_size,
        stratify=labels,
        random_state=random_state
    )
    
    print(f"\nTrain set: {len(train_data)} images")
    print(f"Test set: {len(test_data)} images")
    
    # Verify distribution
    train_labels = [item['label'] for item in train_data]
    test_labels = [item['label'] for item in test_data]
    
    print("\nTrain set distribution:")
    for label, count in Counter(train_labels).items():
        print(f"  {label}: {count}")
    
    print("\nTest set distribution:")
    for label, count in Counter(test_labels).items():
        print(f"  {label}: {count}")
    
    # Copy files to train directory with new names
    print("\n" + "="*60)
    print("Copying training images...")
    for item in train_data:
        src = item['original_path']
        dst = train_dir / item['new_filename']
        shutil.copy2(src, dst)
    
    print(f"Copied {len(train_data)} images to {train_dir}")
    
    # Copy files to test directory with new names
    print("Copying test images...")
    for item in test_data:
        src = item['original_path']
        dst = test_dir / item['new_filename']
        shutil.copy2(src, dst)
    
    print(f"Copied {len(test_data)} images to {test_dir}")
    print("="*60)
    
    return str(train_dir), str(test_dir)


def process_dataset(
    dataset_slug: str = "amirbnsl/breast-cancer-padded-interpolated-720p",
    output_dir: str = None,
    class_folders: List[str] = None,
    test_size: float = 0.1,
    random_state: int = 42
) -> Tuple[str, str]:
    """
    Complete pipeline: Download -> Unzip -> Create Train/Test Split
    
    Args:
        dataset_slug: Kagglehub dataset identifier.
        output_dir: Directory to save train/test folders. If None, uses current directory.
        class_folders: List of class folder names in the dataset.
        test_size: Proportion for test set.
        random_state: Random seed.
    
    Returns:
        Tuple of (train_dir_path, test_dir_path).
    """
    print("\n" + "="*60)
    print("BREAST CANCER DATASET PREPROCESSING")
    print("="*60)
    
    # Step 1: Download dataset
    print("\nStep 1: Downloading dataset...")
    dataset_path = download_dataset(dataset_slug)
    
    # Step 2: Unzip if necessary
    print("\nStep 2: Checking for compressed files...")
    dataset_path = unzip_dataset(dataset_path)
    
    # Step 3: Collect images
    print("\nStep 3: Collecting image paths and labels...")
    image_data = collect_images(dataset_path, class_folders)
    
    # Step 4: Create train/test split
    print("\nStep 4: Creating train/test split with stratified sampling...")
    train_dir, test_dir = create_train_test_split(
        image_data,
        output_dir=output_dir,
        test_size=test_size,
        random_state=random_state
    )
    
    print("\n" + "="*60)
    print("PREPROCESSING COMPLETE!")
    print(f"Train directory: {train_dir}")
    print(f"Test directory: {test_dir}")
    print("="*60 + "\n")
    
    return train_dir, test_dir


if __name__ == "__main__":
    """
    Example usage: Run this script to download, unzip, and organize the dataset
    """
    # Run the complete pipeline with default settings
    # This will create 'train' and 'test' folders in the current working directory
    train_dir, test_dir = process_dataset()
    
    # Alternative: Specify custom output directory
    # train_dir, test_dir = process_dataset(output_dir="/path/to/output")
