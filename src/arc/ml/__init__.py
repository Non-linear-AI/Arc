"""Arc ML package for PyTorch model building and training."""

from .builder import ModelBuilder
from .layers import LAYER_REGISTRY, get_layer_class
from .utils import auto_detect_input_size, validate_tensor_shape

__all__ = [
    "ModelBuilder",
    "LAYER_REGISTRY",
    "get_layer_class",
    "auto_detect_input_size",
    "validate_tensor_shape",
]
