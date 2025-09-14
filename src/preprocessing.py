import cv2
import logging
import numpy as np
from PIL import Image
import albumentations as A
import matplotlib.pyplot as plt

# Augmentation pipeline
ALBUMENTATIONS_TRANSFORM = A.Compose([
    A.Rotate(limit=2, border_mode=cv2.BORDER_CONSTANT, p=0.4),
    A.RandomBrightnessContrast(brightness_limit=0.1, contrast_limit=0.1, p=0.4),
    A.ElasticTransform(alpha=1, sigma=50, p=0.3),
    A.GridDistortion(num_steps=5, distort_limit=0.1, p=0.3),
    A.GaussNoise(noise_scale_factor=0.1, p=0.2),
    A.MotionBlur(blur_limit=3, p=0.2),
])

def preprocess_image(img, target_height=128, target_width=1028,
                     enhance_contrast=False, normalization_range="0_1",
                     log_rescaling=True, binarization=None, binary_threshold=127,
                     adaptive_threshold_block_size=11, adaptive_threshold_c=2):
    """
    Preprocess image for IAM OCR.
    
    Args:
        img: PIL Image or numpy array
        target_height: Target height in pixels
        target_width: Target width in pixels
        enhance_contrast: Whether to apply contrast enhancement
        normalization_range: "0_1" for [0,1] or "-1_1" for [-1,1]
        log_rescaling: Whether to log rescaling operations
        binarization: Binarization method (None, "otsu", "adaptive_mean", etc.)
        binary_threshold: Threshold for simple binarization
        adaptive_threshold_block_size: Block size for adaptive thresholding
        adaptive_threshold_c: Constant for adaptive thresholding
    
    Returns:
        Preprocessed image as numpy array with shape (height, width, 1)
    """
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
    
    # Optional contrast enhancement
    if enhance_contrast:
        img = cv2.convertScaleAbs(img, alpha=1.1, beta=5)
    
    # Apply binarization
    if binarization == "otsu":
        _, img = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    elif binarization == "adaptive_mean":
        img = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_MEAN_C, 
                                   cv2.THRESH_BINARY, adaptive_threshold_block_size, 
                                   adaptive_threshold_c)
    elif binarization == "adaptive_gaussian":
        img = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                   cv2.THRESH_BINARY, adaptive_threshold_block_size, 
                                   adaptive_threshold_c)
    elif binarization == "simple":
        _, img = cv2.threshold(img, binary_threshold, 255, cv2.THRESH_BINARY)
    elif binarization is not None:
        raise ValueError("Invalid binarization method")
    
    # Calculate scaling factors
    scale_h = target_height / h
    scale_w = target_width / w
    scale = min(scale_h, scale_w)
    
    new_h = int(h * scale)
    new_w = int(w * scale)
    
    # Log aspect ratio changes
    if log_rescaling:
        original_aspect = w / h
        target_aspect = target_width / target_height
        if abs(original_aspect - target_aspect) / target_aspect > 0.1:
            logging.info(f"Image rescaled from {w}x{h} to {new_w}x{new_h}")
    
    # Resize and pad
    img = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)
    padded = np.full((target_height, target_width), 255, dtype=np.uint8)
    
    y_offset = (target_height - new_h) // 2
    x_offset = (target_width - new_w) // 2
    padded[y_offset:y_offset + new_h, x_offset:x_offset + new_w] = img
    
    # Normalize
    img = padded.astype(np.float32) / 255.0
    if normalization_range == "-1_1":
        img = (img - 0.5) * 2
    elif normalization_range != "0_1":
        raise ValueError("normalization_range must be '0_1' or '-1_1'")
    
    return np.expand_dims(img, axis=-1)

def show_preprocessed_samples(dataset, n=5):
    """Show original vs preprocessed images."""
    samples = dataset.shuffle(seed=42).select(range(n))
    
    for sample in samples:
        original_img = sample["image"]
        text = sample["text"]
        
        processed_img = preprocess_image(
            original_img, 128, 1028, True, "0_1", True, "adaptive_gaussian"
        )
        processed_img_vis = processed_img.squeeze()
        
        plt.figure(figsize=(12, 3))
        
        plt.subplot(1, 2, 1)
        plt.imshow(original_img, cmap="gray")
        plt.title("Original")
        plt.axis("off")
        
        plt.subplot(1, 2, 2)
        plt.imshow(processed_img_vis, cmap="gray")
        plt.title(f"Preprocessed\n{text}")
        plt.axis("off")
        
        plt.tight_layout()
        plt.show()
