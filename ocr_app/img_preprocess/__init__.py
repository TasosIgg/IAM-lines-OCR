# __init__.py

"""
Image preprocessing package for OCR datasets (Teklia IAM line-level).
Provides utilities for resizing, normalization, contrast enhancement, 
and binarization.
"""

from .img_preprocess import (
    preprocess_image,
    TARGET_HEIGHT,
    TARGET_WIDTH,
)

__all__ = [
    "preprocess_image",
    "TARGET_HEIGHT",
    "TARGET_WIDTH",
]
