# src/data_preprocessing.py
import cv2
import numpy as np
import os

# Supported image extensions for dataset traversal
IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}

# Default dataset location (relative to repository root)
DEFAULT_DATASET_DIR = os.path.normpath(
    os.path.join(os.path.dirname(__file__), "..", "data", "external", "breast-cancer-padded-interpolated-720p")
)

# --- Existing functions ---
def crop_black(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    coords = cv2.findNonZero(gray)
    if coords is not None:
        x, y, w, h = cv2.boundingRect(coords)
        return img[y:y+h, x:x+w]
    else:
        return img

def resize_image(img, size=(224, 224)):
    return cv2.resize(img, size, interpolation=cv2.INTER_AREA)

def normalize_image(img):
    return img.astype(np.float32) / 255.0

# --- Helpers ---
def _is_image_file(path):
    return os.path.splitext(path)[1].lower() in IMAGE_EXTENSIONS


def _collect_image_paths(source):
    """Gather image file paths from a file path or a directory tree."""
    if os.path.isfile(source):
        return [source] if _is_image_file(source) else []

    image_paths = []
    for root, _, files in os.walk(source):
        for fname in files:
            candidate = os.path.join(root, fname)
            if _is_image_file(candidate):
                image_paths.append(candidate)

    image_paths.sort()
    return image_paths

# --- Pipeline function ---
def preprocess_pipeline(input_data, target_size=(224, 224)):
    """
    Preprocess one image, a list, or all images inside a directory tree.
    
    Args:
        input_data: str (file path or directory), numpy array (loaded image), 
                    or list/tuple of str/numpy arrays/directories
        target_size: tuple, size to resize images

    Returns:
        List of normalized images as numpy arrays (or single array if input is single)
    """
    # Detect whether the caller expects a single image back
    single_input = isinstance(input_data, np.ndarray) or (
        isinstance(input_data, str) and not os.path.isdir(input_data)
    )

    if isinstance(input_data, (str, np.ndarray)):
        items = [input_data]
    elif isinstance(input_data, (list, tuple)):
        items = list(input_data)
    else:
        raise TypeError("input_data must be a path, numpy array, or list/tuple of them")

    processed_images = []

    # Expand any directory inputs into individual image paths
    expanded_items = []
    for item in items:
        if isinstance(item, str) and os.path.isdir(item):
            expanded_items.extend(_collect_image_paths(item))
        else:
            expanded_items.append(item)

    for item in expanded_items:
        # Load image if item is a path
        if isinstance(item, str):
            img = cv2.imread(item)
            if img is None:
                print(f"Failed to read {item}")
                continue
        else:
            img = item  # already loaded image

        # Apply preprocessing steps
        img = crop_black(img)
        img = resize_image(img, target_size)
        img = normalize_image(img)

        processed_images.append(img)

    if single_input and len(processed_images) > 0:
        return processed_images[0]  # return single image
    return processed_images


