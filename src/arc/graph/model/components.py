"""Core PyTorch model components with torch.nn prefix for Arc-Graph."""

from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

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

# PyTorch functions registry for functional operations
TORCH_FUNCTIONS = {
    # Core tensor operations
    "torch.add": torch.add,
    "torch.sub": torch.sub,
    "torch.mul": torch.mul,
    "torch.div": torch.div,
    "torch.cat": torch.cat,
    "torch.stack": torch.stack,
    "torch.unsqueeze": torch.unsqueeze,
    "torch.squeeze": torch.squeeze,
    "torch.reshape": torch.reshape,
    "torch.view": torch.Tensor.view,  # Note: this is a method, handled specially
    "torch.mean": torch.mean,
    "torch.sum": torch.sum,
    "torch.max": torch.max,
    "torch.min": torch.min,
    "torch.transpose": torch.transpose,
    "torch.permute": torch.permute,
    "torch.matmul": torch.matmul,
    "torch.mm": torch.mm,
    "torch.bmm": torch.bmm,
    # Functional activation functions
    "torch.nn.functional.relu": F.relu,
    "torch.nn.functional.leaky_relu": F.leaky_relu,
    "torch.nn.functional.elu": F.elu,
    "torch.nn.functional.gelu": F.gelu,
    "torch.nn.functional.silu": F.silu,
    "torch.nn.functional.sigmoid": torch.sigmoid,
    "torch.nn.functional.tanh": torch.tanh,
    "torch.nn.functional.softmax": F.softmax,
    "torch.nn.functional.log_softmax": F.log_softmax,
    "torch.nn.functional.dropout": F.dropout,
    # Normalization functions
    "torch.nn.functional.layer_norm": F.layer_norm,
    "torch.nn.functional.batch_norm": F.batch_norm,
    "torch.nn.functional.group_norm": F.group_norm,
    # Pooling functions
    "torch.nn.functional.max_pool1d": F.max_pool1d,
    "torch.nn.functional.max_pool2d": F.max_pool2d,
    "torch.nn.functional.avg_pool1d": F.avg_pool1d,
    "torch.nn.functional.avg_pool2d": F.avg_pool2d,
    "torch.nn.functional.adaptive_avg_pool1d": F.adaptive_avg_pool1d,
    "torch.nn.functional.adaptive_avg_pool2d": F.adaptive_avg_pool2d,
    # Loss functions (functional versions)
    "torch.nn.functional.mse_loss": F.mse_loss,
    "torch.nn.functional.l1_loss": F.l1_loss,
    "torch.nn.functional.smooth_l1_loss": F.smooth_l1_loss,
    "torch.nn.functional.huber_loss": F.huber_loss,
    "torch.nn.functional.cross_entropy": F.cross_entropy,
    "torch.nn.functional.nll_loss": F.nll_loss,
    "torch.nn.functional.binary_cross_entropy": F.binary_cross_entropy,
    "torch.nn.functional.binary_cross_entropy_with_logits": (
        F.binary_cross_entropy_with_logits
    ),
    "torch.nn.functional.kl_div": F.kl_div,
    "torch.nn.functional.poisson_nll_loss": F.poisson_nll_loss,
    # Convolution functions
    "torch.nn.functional.conv1d": F.conv1d,
    "torch.nn.functional.conv2d": F.conv2d,
    "torch.nn.functional.conv3d": F.conv3d,
    "torch.nn.functional.linear": F.linear,
    # Attention functions
    "torch.nn.functional.scaled_dot_product_attention": F.scaled_dot_product_attention,
}


def get_component_class_or_function(component_type: str) -> tuple[Any, str]:
    """Get PyTorch component (class or function) and its type.

    Args:
        component_type: Component type name
            (e.g., "torch.nn.Linear", "torch.add", "module.MyModule", "arc.stack")

    Returns:
        Tuple of (component, component_kind) where component_kind is one of:
        - "module": torch.nn.Module class
        - "function": torch function or torch.nn.functional function
        - "custom_module": reference to custom module
        - "stack": arc.stack special node type

    Raises:
        ValueError: If component type is not supported
    """
    if component_type in CORE_LAYERS:
        return CORE_LAYERS[component_type], "module"
    elif component_type in TORCH_FUNCTIONS:
        return TORCH_FUNCTIONS[component_type], "function"
    elif component_type.startswith("module."):
        return component_type, "custom_module"
    elif component_type == "arc.stack":
        return component_type, "stack"
    else:
        raise ValueError(f"Unsupported component type: {component_type}")


def get_layer_class(layer_type: str) -> type[nn.Module]:
    """Get PyTorch layer class by type name (backward compatibility).

    Args:
        layer_type: Layer type name (e.g., "torch.nn.Linear")

    Returns:
        PyTorch layer class

    Raises:
        ValueError: If layer type is not supported or is not a module
    """
    component, component_kind = get_component_class_or_function(layer_type)
    if component_kind != "module":
        raise ValueError(f"Layer type {layer_type} is not a PyTorch module")
    return component


def validate_component_params(component_type: str, params: dict[str, Any]) -> bool:
    """Validate component parameters for a given component type.

    Args:
        component_type: Component type name
            (e.g., "torch.nn.Linear", "torch.cat", "arc.stack")
        params: Parameters dictionary

    Returns:
        True if parameters are valid

    Raises:
        ValueError: If parameters are invalid
    """
    # Handle arc.stack special validation
    if component_type == "arc.stack":
        required = ["module", "count"]
        missing = [p for p in required if p not in params]
        if missing:
            raise ValueError(
                f"Missing required parameters for {component_type}: {missing}"
            )
        if not isinstance(params["count"], int) or params["count"] < 1:
            raise ValueError("arc.stack count must be a positive integer")
        return True

    # Module parameter validation
    if component_type == "torch.nn.Linear":
        required = ["in_features", "out_features"]
        missing = [p for p in required if p not in params]
        if missing:
            raise ValueError(
                f"Missing required parameters for {component_type}: {missing}"
            )

    elif component_type == "torch.nn.Conv2d":
        required = ["in_channels", "out_channels", "kernel_size"]
        missing = [p for p in required if p not in params]
        if missing:
            raise ValueError(
                f"Missing required parameters for {component_type}: {missing}"
            )

    elif component_type == "torch.nn.LSTM":
        required = ["input_size", "hidden_size"]
        missing = [p for p in required if p not in params]
        if missing:
            raise ValueError(
                f"Missing required parameters for {component_type}: {missing}"
            )

    elif component_type == "torch.nn.MultiheadAttention":
        required = ["embed_dim", "num_heads"]
        missing = [p for p in required if p not in params]
        if missing:
            raise ValueError(
                f"Missing required parameters for {component_type}: {missing}"
            )

    elif component_type == "torch.nn.TransformerEncoderLayer":
        required = ["d_model", "nhead"]
        missing = [p for p in required if p not in params]
        if missing:
            raise ValueError(
                f"Missing required parameters for {component_type}: {missing}"
            )

    elif component_type == "torch.nn.Embedding":
        required = ["num_embeddings", "embedding_dim"]
        missing = [p for p in required if p not in params]
        if missing:
            raise ValueError(
                f"Missing required parameters for {component_type}: {missing}"
            )

    elif component_type == "torch.nn.BatchNorm1d":
        required = ["num_features"]
        missing = [p for p in required if p not in params]
        if missing:
            raise ValueError(
                f"Missing required parameters for {component_type}: {missing}"
            )

    elif component_type == "torch.nn.LayerNorm":
        required = ["normalized_shape"]
        missing = [p for p in required if p not in params]
        if missing:
            raise ValueError(
                f"Missing required parameters for {component_type}: {missing}"
            )

    # Function parameter validation (most functions accept any keyword params)
    # We could add specific validation for torch functions here if needed

    # For other components, assume parameters are valid if present
    return True


def validate_layer_params(layer_type: str, params: dict[str, Any]) -> bool:
    """Validate layer parameters for a given layer type (backward compatibility).

    Args:
        layer_type: Layer type name (e.g., "torch.nn.Linear")
        params: Parameters dictionary

    Returns:
        True if parameters are valid

    Raises:
        ValueError: If parameters are invalid
    """
    return validate_component_params(layer_type, params)


def get_supported_layer_types() -> list[str]:
    """Get list of all supported layer types (backward compatibility).

    Returns:
        List of supported layer type names
    """
    return list(CORE_LAYERS.keys())


def get_supported_component_types() -> dict[str, list[str]]:
    """Get list of all supported component types organized by category.

    Returns:
        Dictionary with component categories and their supported types
    """
    return {
        "modules": list(CORE_LAYERS.keys()),
        "functions": list(TORCH_FUNCTIONS.keys()),
        "special": ["arc.stack"],
        "custom": ["module.*"],  # Pattern for custom modules
    }
