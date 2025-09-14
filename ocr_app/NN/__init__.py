# __init__.py

"""
Neural network architectures for handwriting recognition.

This package currently provides:
- CRNN (Convolutional Recurrent Neural Network) with residual CNN backbone,
  spatial attention, and bidirectional LSTMs.
- ResidualBlock utility module.
- MultiHeadSpatialAttention for fine-grained feature focusing.
"""

from .NN import (
    CRNN,
    ResidualBlock,
    MultiHeadSpatialAttention,
)

__all__ = [
    "CRNN",
    "ResidualBlock",
    "MultiHeadSpatialAttention",
]
