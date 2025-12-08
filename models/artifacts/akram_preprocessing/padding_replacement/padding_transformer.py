"""
Padding Replacement Transformer

This module provides functionality to replace black pixels (padding) in X-ray images
with the average non-black pixel value calculated from training data.
"""

import numpy as np
from PIL import Image
from pathlib import Path
from typing import Union, Tuple


class PaddingReplacementTransformer:
    """
    A transformer that replaces black pixels in images with an average pixel value
    loaded from a pre-calculated .npy file.
    
    The average pixel value is calculated from all non-black pixels in training images,
    ensuring consistent and realistic padding replacement across the dataset.
    """
    
    def __init__(self, avg_pixel_path: str = "average_pixel_value.npy"):
        """
        Initialize the transformer with the average pixel value.
        
        Args:
            avg_pixel_path: Path to the average_pixel_value.npy file.
                           Defaults to "average_pixel_value.npy" in the same directory as this script.
        
        Raises:
            FileNotFoundError: If the average pixel value file is not found.
            ValueError: If the loaded average pixel value has an invalid shape.
        """
        # Look for average_pixel_value.npy in the same directory as this script if using default
        if avg_pixel_path == "average_pixel_value.npy":
            script_dir = Path(__file__).parent
            avg_pixel_path = script_dir / avg_pixel_path
        else:
            avg_pixel_path = Path(avg_pixel_path)
        
        if not avg_pixel_path.exists():
            raise FileNotFoundError(
                f"Average pixel value file not found at: {avg_pixel_path}\n"
                f"Expected file: average_pixel_value.npy"
            )
        
        # Load the average pixel value
        self.average_pixel_value = np.load(avg_pixel_path)
        
        # Validate the loaded value
        if self.average_pixel_value.shape != (3,):
            raise ValueError(
                f"Invalid average pixel value shape: {self.average_pixel_value.shape}. "
                f"Expected shape: (3,) for RGB channels"
            )
        
        print(f"Loaded average pixel value from: {avg_pixel_path}")
        print(f"  R: {self.average_pixel_value[0]:.2f}")
        print(f"  G: {self.average_pixel_value[1]:.2f}")
        print(f"  B: {self.average_pixel_value[2]:.2f}")
    
    def transform(self, image: Union[np.ndarray, Image.Image, str, Path]) -> np.ndarray:
        """
        Replace black pixels in an image with the average pixel value.
        
        Args:
            image: Input image as:
                  - numpy array (H, W, C) with values in [0, 255]
                  - PIL Image object
                  - path to image file (str or Path)
        
        Returns:
            Transformed image as numpy array with dtype uint8, same shape as input.
        
        Raises:
            ValueError: If image has invalid shape or type.
        """
        # Convert input to numpy array
        if isinstance(image, (str, Path)):
            image = Image.open(image)
        
        if isinstance(image, Image.Image):
            img_array = np.array(image, dtype=np.float64)
        elif isinstance(image, np.ndarray):
            img_array = image.astype(np.float64)
        else:
            raise ValueError(
                f"Invalid image type: {type(image)}. "
                f"Expected numpy array, PIL Image, or file path."
            )
        
        # Validate image shape
        if len(img_array.shape) != 3 or img_array.shape[2] != 3:
            raise ValueError(
                f"Invalid image shape: {img_array.shape}. "
                f"Expected (H, W, 3) for RGB images."
            )
        
        # Create a copy to avoid modifying original
        img_replaced = img_array.copy()
        
        # Find black pixels (where sum of all channels equals 0)
        black_mask = img_array.sum(axis=2) == 0
        
        # Replace black pixels with average value
        img_replaced[black_mask] = self.average_pixel_value
        
        # Convert back to uint8
        return img_replaced.astype(np.uint8)
    
    def transform_batch(self, image_paths: list) -> list:
        """
        Transform a batch of images.
        
        Args:
            image_paths: List of paths to image files or numpy arrays.
        
        Returns:
            List of transformed images as numpy arrays.
        """
        transformed_images = []
        for image_path in image_paths:
            transformed_image = self.transform(image_path)
            transformed_images.append(transformed_image)
        return transformed_images
    
    def save_transformed_image(
        self, 
        image: Union[np.ndarray, Image.Image, str, Path],
        output_path: Union[str, Path]
    ) -> None:
        """
        Transform an image and save it to disk.
        
        Args:
            image: Input image.
            output_path: Path where to save the transformed image.
        """
        transformed = self.transform(image)
        transformed_image = Image.fromarray(transformed)
        transformed_image.save(output_path)
        print(f"Saved transformed image to: {output_path}")


# Convenient function for quick usage
def replace_padding(
    image: Union[np.ndarray, Image.Image, str, Path],
    avg_pixel_path: Union[str, Path] = None
) -> np.ndarray:
    """
    Quick function to replace black pixels in a single image.
    
    Args:
        image: Input image.
        avg_pixel_path: Path to average_pixel_value.npy file.
    
    Returns:
        Transformed image as numpy array.
    """
    transformer = PaddingReplacementTransformer(avg_pixel_path)
    return transformer.transform(image)


if __name__ == "__main__":
    """
    Example usage of the PaddingReplacementTransformer
    """
    import os
    
    # Initialize transformer
    transformer = PaddingReplacementTransformer()
    
    # Example: Transform a single image
    # image_path = "path/to/image.png"
    # transformed = transformer.transform(image_path)
    # transformer.save_transformed_image(image_path, "path/to/output.png")
    
    print("\nTransformer initialized successfully!")
    print("Usage example:")
    print("  from transformer import PaddingReplacementTransformer")
    print("  transformer = PaddingReplacementTransformer()")
    print("  transformed_image = transformer.transform('path/to/image.png')")
