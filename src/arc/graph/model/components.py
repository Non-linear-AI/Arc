"""Core PyTorch model components with torch.nn prefix for Arc-Graph."""

from __future__ import annotations

from typing import Any

import torch.nn as nn

# Direct PyTorch mapping with torch.nn prefix for LLM clarity
CORE_LAYERS = {
    # Linear layers
    "torch.nn.Linear": nn.Linear,
    "torch.nn.Bilinear": nn.Bilinear,
    # Convolution layers
    "torch.nn.Conv1d": nn.Conv1d,
    "torch.nn.Conv2d": nn.Conv2d,
    "torch.nn.Conv3d": nn.Conv3d,
    "torch.nn.ConvTranspose1d": nn.ConvTranspose1d,
    "torch.nn.ConvTranspose2d": nn.ConvTranspose2d,
    "torch.nn.ConvTranspose3d": nn.ConvTranspose3d,
    # Activation functions
    "torch.nn.ReLU": nn.ReLU,
    "torch.nn.Sigmoid": nn.Sigmoid,
    "torch.nn.Tanh": nn.Tanh,
    "torch.nn.Softmax": nn.Softmax,
    "torch.nn.LogSoftmax": nn.LogSoftmax,
    "torch.nn.LeakyReLU": nn.LeakyReLU,
    "torch.nn.ELU": nn.ELU,
    "torch.nn.GELU": nn.GELU,
    "torch.nn.SiLU": nn.SiLU,  # Swish activation
    "torch.nn.Mish": nn.Mish,
    "torch.nn.Hardswish": nn.Hardswish,
    "torch.nn.Hardsigmoid": nn.Hardsigmoid,
    # Normalization layers
    "torch.nn.BatchNorm1d": nn.BatchNorm1d,
    "torch.nn.BatchNorm2d": nn.BatchNorm2d,
    "torch.nn.BatchNorm3d": nn.BatchNorm3d,
    "torch.nn.LayerNorm": nn.LayerNorm,
    "torch.nn.GroupNorm": nn.GroupNorm,
    "torch.nn.InstanceNorm1d": nn.InstanceNorm1d,
    "torch.nn.InstanceNorm2d": nn.InstanceNorm2d,
    "torch.nn.InstanceNorm3d": nn.InstanceNorm3d,
    "torch.nn.LocalResponseNorm": nn.LocalResponseNorm,
    # Dropout and regularization
    "torch.nn.Dropout": nn.Dropout,
    "torch.nn.Dropout1d": nn.Dropout1d,
    "torch.nn.Dropout2d": nn.Dropout2d,
    "torch.nn.Dropout3d": nn.Dropout3d,
    "torch.nn.AlphaDropout": nn.AlphaDropout,
    "torch.nn.FeatureAlphaDropout": nn.FeatureAlphaDropout,
    # Pooling layers
    "torch.nn.MaxPool1d": nn.MaxPool1d,
    "torch.nn.MaxPool2d": nn.MaxPool2d,
    "torch.nn.MaxPool3d": nn.MaxPool3d,
    "torch.nn.AvgPool1d": nn.AvgPool1d,
    "torch.nn.AvgPool2d": nn.AvgPool2d,
    "torch.nn.AvgPool3d": nn.AvgPool3d,
    "torch.nn.AdaptiveAvgPool1d": nn.AdaptiveAvgPool1d,
    "torch.nn.AdaptiveAvgPool2d": nn.AdaptiveAvgPool2d,
    "torch.nn.AdaptiveAvgPool3d": nn.AdaptiveAvgPool3d,
    "torch.nn.AdaptiveMaxPool1d": nn.AdaptiveMaxPool1d,
    "torch.nn.AdaptiveMaxPool2d": nn.AdaptiveMaxPool2d,
    "torch.nn.AdaptiveMaxPool3d": nn.AdaptiveMaxPool3d,
    "torch.nn.MaxUnpool1d": nn.MaxUnpool1d,
    "torch.nn.MaxUnpool2d": nn.MaxUnpool2d,
    "torch.nn.MaxUnpool3d": nn.MaxUnpool3d,
    # Recurrent layers
    "torch.nn.LSTM": nn.LSTM,
    "torch.nn.GRU": nn.GRU,
    "torch.nn.RNN": nn.RNN,
    # Attention and transformer layers
    "torch.nn.MultiheadAttention": nn.MultiheadAttention,
    "torch.nn.TransformerEncoderLayer": nn.TransformerEncoderLayer,
    "torch.nn.TransformerDecoderLayer": nn.TransformerDecoderLayer,
    "torch.nn.TransformerEncoder": nn.TransformerEncoder,
    "torch.nn.TransformerDecoder": nn.TransformerDecoder,
    "torch.nn.Transformer": nn.Transformer,
    # Embedding layers
    "torch.nn.Embedding": nn.Embedding,
    "torch.nn.EmbeddingBag": nn.EmbeddingBag,
    # Utility layers
    "torch.nn.Flatten": nn.Flatten,
    "torch.nn.Unflatten": nn.Unflatten,
    "torch.nn.Identity": nn.Identity,
    "torch.nn.Upsample": nn.Upsample,
    "torch.nn.UpsamplingNearest2d": nn.UpsamplingNearest2d,
    "torch.nn.UpsamplingBilinear2d": nn.UpsamplingBilinear2d,
    # Padding layers
    "torch.nn.ReflectionPad1d": nn.ReflectionPad1d,
    "torch.nn.ReflectionPad2d": nn.ReflectionPad2d,
    "torch.nn.ReplicationPad1d": nn.ReplicationPad1d,
    "torch.nn.ReplicationPad2d": nn.ReplicationPad2d,
    "torch.nn.ReplicationPad3d": nn.ReplicationPad3d,
    "torch.nn.ZeroPad2d": nn.ZeroPad2d,
    "torch.nn.ConstantPad1d": nn.ConstantPad1d,
    "torch.nn.ConstantPad2d": nn.ConstantPad2d,
    "torch.nn.ConstantPad3d": nn.ConstantPad3d,
}


def get_layer_class(layer_type: str) -> type[nn.Module]:
    """Get PyTorch layer class by type name.

    Args:
        layer_type: Layer type name (e.g., "pytorch.Linear")

    Returns:
        PyTorch layer class

    Raises:
        ValueError: If layer type is not supported
    """
    if layer_type not in CORE_LAYERS:
        raise ValueError(f"Unsupported layer type: {layer_type}")

    return CORE_LAYERS[layer_type]


def validate_layer_params(layer_type: str, params: dict[str, Any]) -> bool:
    """Validate layer parameters for a given layer type.

    Args:
        layer_type: Layer type name (e.g., "pytorch.Linear")
        params: Parameters dictionary

    Returns:
        True if parameters are valid

    Raises:
        ValueError: If parameters are invalid
    """
    if layer_type == "torch.nn.Linear":
        required = ["in_features", "out_features"]
        missing = [p for p in required if p not in params]
        if missing:
            raise ValueError(f"Missing required parameters for {layer_type}: {missing}")

    elif layer_type == "torch.nn.Conv2d":
        required = ["in_channels", "out_channels", "kernel_size"]
        missing = [p for p in required if p not in params]
        if missing:
            raise ValueError(f"Missing required parameters for {layer_type}: {missing}")

    elif layer_type == "torch.nn.LSTM":
        required = ["input_size", "hidden_size"]
        missing = [p for p in required if p not in params]
        if missing:
            raise ValueError(f"Missing required parameters for {layer_type}: {missing}")

    elif layer_type == "torch.nn.MultiheadAttention":
        required = ["embed_dim", "num_heads"]
        missing = [p for p in required if p not in params]
        if missing:
            raise ValueError(f"Missing required parameters for {layer_type}: {missing}")

    elif layer_type == "torch.nn.TransformerEncoderLayer":
        required = ["d_model", "nhead"]
        missing = [p for p in required if p not in params]
        if missing:
            raise ValueError(f"Missing required parameters for {layer_type}: {missing}")

    elif layer_type == "torch.nn.Embedding":
        required = ["num_embeddings", "embedding_dim"]
        missing = [p for p in required if p not in params]
        if missing:
            raise ValueError(f"Missing required parameters for {layer_type}: {missing}")

    elif layer_type == "torch.nn.BatchNorm1d":
        required = ["num_features"]
        missing = [p for p in required if p not in params]
        if missing:
            raise ValueError(f"Missing required parameters for {layer_type}: {missing}")

    elif layer_type == "torch.nn.LayerNorm":
        required = ["normalized_shape"]
        missing = [p for p in required if p not in params]
        if missing:
            raise ValueError(f"Missing required parameters for {layer_type}: {missing}")

    # For other layers, assume parameters are valid if present
    return True


def get_supported_layer_types() -> list[str]:
    """Get list of all supported layer types.

    Returns:
        List of supported layer type names
    """
    return list(CORE_LAYERS.keys())
