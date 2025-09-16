"""Arc-Graph layer implementations for PyTorch."""

from __future__ import annotations

import math

import torch
import torch.nn as nn


class ArcLayerBase(nn.Module):
    """Base class for all Arc-Graph layers."""

    def __init__(self, **params):
        super().__init__()
        self.params = params

    def forward(self, *args, **kwargs) -> torch.Tensor:
        """Forward pass with flexible input handling.

        Supports both positional and keyword arguments for multi-input layers.
        """
        # Handle keyword arguments by extracting primary input
        if kwargs and not args:
            # Multi-input layer case - delegate to _forward_multi
            if len(kwargs) > 1:
                return self._forward_multi(**kwargs)
            # Single input via keyword
            elif "input" in kwargs:
                return self._forward_single(kwargs["input"])
            else:
                # Use first available input
                input_tensor = next(iter(kwargs.values()))
                return self._forward_single(input_tensor)

        # Handle positional arguments (traditional case)
        elif args and not kwargs:
            if len(args) == 1:
                return self._forward_single(args[0])
            else:
                # Multiple positional args - convert to kwargs for consistency
                input_names = self._get_input_names()
                kwargs_from_args = dict(zip(input_names, args, strict=False))
                return self._forward_multi(**kwargs_from_args)

        # Mixed args/kwargs not supported
        elif args and kwargs:
            raise ValueError("Cannot mix positional and keyword arguments")

        # No arguments
        else:
            raise ValueError("No input arguments provided")

    def _forward_single(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass for single input (default behavior)."""
        raise NotImplementedError

    def _forward_multi(self, **inputs) -> torch.Tensor:
        """Forward pass for multiple inputs (override for multi-input layers)."""
        # Default: extract primary input and use single-input forward
        if "input" in inputs:
            return self._forward_single(inputs["input"])
        else:
            # Use first available input
            input_tensor = next(iter(inputs.values()))
            return self._forward_single(input_tensor)

    def _get_input_names(self) -> list[str]:
        """Get expected input names for multi-input layers."""
        return ["input"]  # Default single input name


class LinearLayer(ArcLayerBase):
    """Linear (fully connected) layer implementation."""

    def __init__(
        self, in_features: int, out_features: int, bias: bool = True, **params
    ):
        super().__init__(**params)
        self.linear = nn.Linear(in_features, out_features, bias=bias)

    def _forward_single(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear(x)


class ReLULayer(ArcLayerBase):
    """ReLU activation layer implementation."""

    def __init__(self, inplace: bool = False, **params):
        super().__init__(**params)
        self.relu = nn.ReLU(inplace=inplace)

    def _forward_single(self, x: torch.Tensor) -> torch.Tensor:
        return self.relu(x)


class SigmoidLayer(ArcLayerBase):
    """Sigmoid activation layer implementation."""

    def __init__(self, **params):
        super().__init__(**params)
        self.sigmoid = nn.Sigmoid()

    def _forward_single(self, x: torch.Tensor) -> torch.Tensor:
        return self.sigmoid(x)


class DropoutLayer(ArcLayerBase):
    """Dropout layer implementation."""

    def __init__(self, p: float = 0.5, inplace: bool = False, **params):
        super().__init__(**params)
        self.dropout = nn.Dropout(p=p, inplace=inplace)

    def _forward_single(self, x: torch.Tensor) -> torch.Tensor:
        return self.dropout(x)


class BatchNorm1dLayer(ArcLayerBase):
    """1D Batch Normalization layer implementation."""

    def __init__(
        self,
        num_features: int,
        eps: float = 1e-5,
        momentum: float = 0.1,
        **params,
    ):
        super().__init__(**params)
        self.batch_norm = nn.BatchNorm1d(num_features, eps=eps, momentum=momentum)

    def _forward_single(self, x: torch.Tensor) -> torch.Tensor:
        return self.batch_norm(x)


class EmbeddingLayer(ArcLayerBase):
    """Embedding layer for discrete inputs."""

    def __init__(
        self,
        num_embeddings: int,
        embedding_dim: int,
        padding_idx: int | None = None,
        max_norm: float | None = None,
        norm_type: float = 2.0,
        scale_grad_by_freq: bool = False,
        sparse: bool = False,
        **params,
    ):
        super().__init__(**params)
        self.embedding = nn.Embedding(
            num_embeddings=num_embeddings,
            embedding_dim=embedding_dim,
            padding_idx=padding_idx,
            max_norm=max_norm,
            norm_type=norm_type,
            scale_grad_by_freq=scale_grad_by_freq,
            sparse=sparse,
        )

    def _forward_single(self, x: torch.Tensor) -> torch.Tensor:
        return self.embedding(x)


class MultiHeadAttentionLayer(ArcLayerBase):
    """Multi-head attention layer."""

    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        dropout: float = 0.0,
        bias: bool = True,
        batch_first: bool = True,
        **params,
    ):
        super().__init__(**params)
        self.attention = nn.MultiheadAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            dropout=dropout,
            bias=bias,
            batch_first=batch_first,
        )

    def _forward_single(self, x: torch.Tensor) -> torch.Tensor:
        # Self-attention: query, key, value are all the same
        output, _ = self.attention(x, x, x)
        return output

    def _forward_multi(self, **inputs) -> torch.Tensor:
        """Support explicit query, key, value inputs."""
        if "query" in inputs and "key" in inputs and "value" in inputs:
            output, _ = self.attention(inputs["query"], inputs["key"], inputs["value"])
            return output
        elif "input" in inputs:
            return self._forward_single(inputs["input"])
        else:
            # Default to self-attention with first input
            x = next(iter(inputs.values()))
            return self._forward_single(x)

    def _get_input_names(self) -> list[str]:
        return ["query", "key", "value"]


class TransformerEncoderLayerCustom(ArcLayerBase):
    """Single transformer encoder layer."""

    def __init__(
        self,
        d_model: int,
        nhead: int,
        dim_feedforward: int = 2048,
        dropout: float = 0.1,
        activation: str = "relu",
        batch_first: bool = True,
        **params,
    ):
        super().__init__(**params)
        self.transformer_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation=activation,
            batch_first=batch_first,
        )

    def _forward_single(self, x: torch.Tensor) -> torch.Tensor:
        return self.transformer_layer(x)


class PositionalEncodingLayer(ArcLayerBase):
    """Positional encoding for transformer models."""

    def __init__(
        self, d_model: int, max_len: int = 5000, dropout: float = 0.1, **params
    ):
        super().__init__(**params)
        self.dropout = nn.Dropout(p=dropout)
        self.d_model = d_model

        # Create positional encoding matrix
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer("pe", pe)

    def _forward_single(self, x: torch.Tensor) -> torch.Tensor:
        # x shape: [batch_size, seq_len, d_model] (batch_first=True)
        seq_len = x.size(1)
        x = x + self.pe[:seq_len, :].transpose(0, 1)
        return self.dropout(x)


class LayerNormLayer(ArcLayerBase):
    """Layer normalization."""

    def __init__(
        self,
        normalized_shape: int | list[int],
        eps: float = 1e-5,
        elementwise_affine: bool = True,
        **params,
    ):
        super().__init__(**params)
        self.layer_norm = nn.LayerNorm(
            normalized_shape=normalized_shape,
            eps=eps,
            elementwise_affine=elementwise_affine,
        )

    def _forward_single(self, x: torch.Tensor) -> torch.Tensor:
        return self.layer_norm(x)


class ConcatenateLayer(ArcLayerBase):
    """Concatenate multiple tensors along a specified dimension."""

    def __init__(self, dim: int = -1, **params):
        super().__init__(**params)
        self.dim = dim

    def _forward_multi(self, **inputs) -> torch.Tensor:
        """Concatenate all input tensors."""
        tensors = list(inputs.values())
        if len(tensors) < 2:
            raise ValueError("ConcatenateLayer requires at least 2 inputs")
        return torch.cat(tensors, dim=self.dim)

    def _forward_single(self, x: torch.Tensor) -> torch.Tensor:
        # Single input - just pass through
        return x


class AddLayer(ArcLayerBase):
    """Element-wise addition of multiple tensors."""

    def __init__(self, **params):
        super().__init__(**params)

    def _forward_multi(self, **inputs) -> torch.Tensor:
        """Add all input tensors element-wise."""
        tensors = list(inputs.values())
        if len(tensors) < 2:
            raise ValueError("AddLayer requires at least 2 inputs")
        result = tensors[0]
        for tensor in tensors[1:]:
            result = result + tensor
        return result

    def _forward_single(self, x: torch.Tensor) -> torch.Tensor:
        # Single input - just pass through
        return x


class LSTMLayer(ArcLayerBase):
    """LSTM layer for sequence modeling."""

    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        num_layers: int = 1,
        bias: bool = True,
        batch_first: bool = True,
        dropout: float = 0.0,
        bidirectional: bool = False,
        **params,
    ):
        super().__init__(**params)
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            bias=bias,
            batch_first=batch_first,
            dropout=dropout,
            bidirectional=bidirectional,
        )

    def _forward_single(self, x: torch.Tensor) -> torch.Tensor:
        output, _ = self.lstm(x)
        return output


class GRULayer(ArcLayerBase):
    """GRU layer for sequence modeling."""

    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        num_layers: int = 1,
        bias: bool = True,
        batch_first: bool = True,
        dropout: float = 0.0,
        bidirectional: bool = False,
        **params,
    ):
        super().__init__(**params)
        self.gru = nn.GRU(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            bias=bias,
            batch_first=batch_first,
            dropout=dropout,
            bidirectional=bidirectional,
        )

    def _forward_single(self, x: torch.Tensor) -> torch.Tensor:
        output, _ = self.gru(x)
        return output


# Registry mapping Arc-Graph layer types to implementation classes
LAYER_REGISTRY: dict[str, type[ArcLayerBase]] = {
    # Basic layers
    "core.Linear": LinearLayer,
    "core.ReLU": ReLULayer,
    "core.Sigmoid": SigmoidLayer,
    "core.Dropout": DropoutLayer,
    "core.BatchNorm1d": BatchNorm1dLayer,
    "core.LayerNorm": LayerNormLayer,
    # Embedding layers
    "core.Embedding": EmbeddingLayer,
    # Attention and transformer layers
    "core.MultiHeadAttention": MultiHeadAttentionLayer,
    "core.TransformerEncoderLayer": TransformerEncoderLayerCustom,
    "core.PositionalEncoding": PositionalEncodingLayer,
    # Sequence modeling layers
    "core.LSTM": LSTMLayer,
    "core.GRU": GRULayer,
    # Routing and combination layers
    "core.Concatenate": ConcatenateLayer,
    "core.Add": AddLayer,
}


def get_layer_class(layer_type: str) -> type[ArcLayerBase]:
    """Get layer class by Arc-Graph type name.

    Args:
        layer_type: Arc-Graph layer type (e.g., "core.Linear")

    Returns:
        Layer class

    Raises:
        ValueError: If layer type is not registered
    """
    if layer_type not in LAYER_REGISTRY:
        available_types = ", ".join(LAYER_REGISTRY.keys())
        raise ValueError(
            f"Unknown layer type '{layer_type}'. Available types: {available_types}"
        )

    return LAYER_REGISTRY[layer_type]


def register_layer(layer_type: str, layer_class: type[ArcLayerBase]) -> None:
    """Register a new layer type.

    Args:
        layer_type: Arc-Graph layer type name
        layer_class: Layer implementation class
    """
    LAYER_REGISTRY[layer_type] = layer_class
