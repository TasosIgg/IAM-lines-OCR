# img_preprocess.py

import cv2
from PIL import Image
import numpy as np
import logging

TARGET_HEIGHT = 128
TARGET_WIDTH = 1024


def preprocess_image(img, target_height=TARGET_HEIGHT, target_width=TARGET_WIDTH,
                     enhance_contrast=False, normalization_range="0_1",
                     log_rescaling=True, binarization=None, binary_threshold=127,
                     adaptive_threshold_block_size=11, adaptive_threshold_c=2):
    
    # Convert to grayscale numpy array
    if isinstance(img, Image.Image):
        img = np.array(img.convert("L"))
    elif len(img.shape) == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Handle edge cases
    if img.size == 0:
        logging.warning("Empty image provided, returning zero array")
        return np.zeros((target_height, target_width, 1), dtype=np.float32)
    
    h, w = img.shape
    if h == 0 or w == 0:
        logging.warning(f"Invalid image dimensions: {h}x{w}, returning zero array")
        return np.zeros((target_height, target_width, 1), dtype=np.float32)
    
    # Optional: Light contrast enhancement for faded lines
    if enhance_contrast:
        img = cv2.convertScaleAbs(img, alpha=1.1, beta=5)
    
    # Apply binarization if requested
    if binarization is not None:
        if binarization == "otsu":
            # Otsu's method automatically finds optimal threshold
            _, img = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        elif binarization == "adaptive_mean":
            # Adaptive threshold using mean of neighborhood
            img = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_MEAN_C, 
                                       cv2.THRESH_BINARY, adaptive_threshold_block_size, 
                                       adaptive_threshold_c)
        elif binarization == "adaptive_gaussian":
            # Adaptive threshold using Gaussian-weighted sum of neighborhood
            img = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                       cv2.THRESH_BINARY, adaptive_threshold_block_size, 
                                       adaptive_threshold_c)
        elif binarization == "simple":
            # Simple binary threshold
            _, img = cv2.threshold(img, binary_threshold, 255, cv2.THRESH_BINARY)
        else:
            raise ValueError("binarization must be None, 'otsu', 'adaptive_mean', 'adaptive_gaussian', or 'simple'")
    
    # Calculate scaling factors for both dimensions
    scale_h = target_height / h
    scale_w = target_width / w
    
    # Use the smaller scale to ensure the image fits within target dimensions
    scale = min(scale_h, scale_w)
    
    # Calculate new dimensions
    new_h = int(h * scale)
    new_w = int(w * scale)
    
    # Log rescaling if it occurs and aspect ratio changes significantly
    if log_rescaling:
        original_aspect = w / h
        target_aspect = target_width / target_height
        if abs(original_aspect - target_aspect) / target_aspect > 0.1:  # >10% difference
            logging.info(f"Image rescaled from {w}x{h} to {new_w}x{new_h} "
                        f"(aspect ratio changed from {original_aspect:.2f} to {new_w/new_h:.2f})")
    
    # Resize image while preserving aspect ratio
    img = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)
    
    # Create padded image with white background
    padded = np.full((target_height, target_width), 255, dtype=np.uint8)
    
    # Center the resized image in the padded canvas
    y_offset = (target_height - new_h) // 2
    x_offset = (target_width - new_w) // 2
    padded[y_offset:y_offset + new_h, x_offset:x_offset + new_w] = img
    
    # Normalize based on specified range
    img = padded.astype(np.float32) / 255.0
    
    if normalization_range == "-1_1":
        img = (img - 0.5) * 2  # Convert [0,1] to [-1,1]
    elif normalization_range != "0_1":
        raise ValueError("normalization_range must be '0_1' or '-1_1'")
    
    # Add channel dimension
    img = np.expand_dims(img, axis=-1)
    
    return img
