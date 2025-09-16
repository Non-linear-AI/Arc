"""Built-in plugins for Arc Graph.

This module provides built-in plugin implementations that register
Arc Graph's existing layer types, optimizers, and loss functions
through the plugin system.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import pluggy

if TYPE_CHECKING:
    pass

# Create hook implementation markers
hookimpl = pluggy.HookimplMarker("arc")


class BuiltinLayerPlugin:
    """Plugin that registers Arc Graph's built-in layer types."""

    __plugin_metadata__ = {
        "name": "builtin_layers",
        "version": "1.0.0",
        "description": "Built-in Arc Graph layer types",
    }

    @hookimpl
    def register_layers(self) -> dict[str, type]:
        """Register built-in layer implementations."""
        from ..ml.layers import (
            AddLayer,
            BatchNorm1dLayer,
            ConcatenateLayer,
            DropoutLayer,
            EmbeddingLayer,
            GRULayer,
            LayerNormLayer,
            LinearLayer,
            LSTMLayer,
            MultiHeadAttentionLayer,
            PositionalEncodingLayer,
            ReLULayer,
            SigmoidLayer,
            TransformerEncoderLayerCustom,
        )

        return {
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
            # Aliases for common layer names
            "Linear": LinearLayer,
            "ReLU": ReLULayer,
            "Sigmoid": SigmoidLayer,
            "Dropout": DropoutLayer,
            "BatchNorm": BatchNorm1dLayer,
            "LayerNorm": LayerNormLayer,
            "Embedding": EmbeddingLayer,
            "MultiHeadAttention": MultiHeadAttentionLayer,
            "TransformerLayer": TransformerEncoderLayerCustom,
            "PositionalEncoding": PositionalEncodingLayer,
            "LSTM": LSTMLayer,
            "GRU": GRULayer,
            "Concat": ConcatenateLayer,
            "Add": AddLayer,
        }

    @hookimpl
    def validate_layer_config(self, layer_type: str, config: dict[str, Any]) -> bool:
        """Validate layer configuration for built-in layers."""
        # Basic validation for built-in layers (handle both core. and alias names)
        if layer_type in ("core.Linear", "Linear"):
            required = ["in_features", "out_features"]
            return all(param in config for param in required)
        elif layer_type in ("core.Embedding", "Embedding"):
            required = ["num_embeddings", "embedding_dim"]
            return all(param in config for param in required)
        elif layer_type in ("core.LSTM", "LSTM") or layer_type in ("core.GRU", "GRU"):
            required = ["input_size", "hidden_size"]
            return all(param in config for param in required)
        elif layer_type in ("core.MultiHeadAttention", "MultiHeadAttention"):
            required = ["embed_dim", "num_heads"]
            return all(param in config for param in required)
        elif layer_type in ("core.TransformerEncoderLayer", "TransformerLayer"):
            required = ["d_model", "nhead"]
            return all(param in config for param in required)
        elif layer_type in ("core.PositionalEncoding", "PositionalEncoding"):
            required = ["d_model"]
            return all(param in config for param in required)
        elif layer_type in ("core.BatchNorm1d", "BatchNorm"):
            required = ["num_features"]
            return all(param in config for param in required)
        elif layer_type in ("core.LayerNorm", "LayerNorm"):
            required = ["normalized_shape"]
            return all(param in config for param in required)

        # For other layers, basic validation passes
        return True


class BuiltinOptimizerPlugin:
    """Plugin that registers Arc Graph's built-in optimizer types."""

    __plugin_metadata__ = {
        "name": "builtin_optimizers",
        "version": "1.0.0",
        "description": "Built-in PyTorch optimizers",
    }

    @hookimpl
    def register_optimizers(self) -> dict[str, type]:
        """Register built-in optimizer implementations."""
        import torch.optim as optim

        return {
            "SGD": optim.SGD,
            "Adam": optim.Adam,
            "AdamW": optim.AdamW,
            "RMSprop": optim.RMSprop,
            "Adagrad": optim.Adagrad,
            "Adadelta": optim.Adadelta,
            "AdamAx": optim.Adamax,
            "ASGD": optim.ASGD,
            "LBFGS": optim.LBFGS,
        }

    @hookimpl
    def validate_optimizer_config(
        self, optimizer_name: str, config: dict[str, Any]
    ) -> bool:
        """Validate optimizer configuration."""
        # All optimizers require learning rate
        if "lr" not in config:
            return False

        # Specific validation for certain optimizers
        return not (
            optimizer_name == "SGD"
            and "momentum" in config
            and not (0.0 <= config["momentum"] <= 1.0)
        )


class BuiltinLossPlugin:
    """Plugin that registers Arc Graph's built-in loss function types."""

    __plugin_metadata__ = {
        "name": "builtin_losses",
        "version": "1.0.0",
        "description": "Built-in PyTorch loss functions",
    }

    @hookimpl
    def register_losses(self) -> dict[str, type]:
        """Register built-in loss function implementations."""
        import torch.nn as nn

        return {
            "MSELoss": nn.MSELoss,
            "CrossEntropyLoss": nn.CrossEntropyLoss,
            "NLLLoss": nn.NLLLoss,
            "BCELoss": nn.BCELoss,
            "BCEWithLogitsLoss": nn.BCEWithLogitsLoss,
            "L1Loss": nn.L1Loss,
            "SmoothL1Loss": nn.SmoothL1Loss,
            "HuberLoss": nn.HuberLoss,
            "PoissonNLLLoss": nn.PoissonNLLLoss,
            "KLDivLoss": nn.KLDivLoss,
            "MarginRankingLoss": nn.MarginRankingLoss,
            "HingeEmbeddingLoss": nn.HingeEmbeddingLoss,
            "MultiLabelMarginLoss": nn.MultiLabelMarginLoss,
            "MultiLabelSoftMarginLoss": nn.MultiLabelSoftMarginLoss,
            "MultiMarginLoss": nn.MultiMarginLoss,
            "TripletMarginLoss": nn.TripletMarginLoss,
            "CosineEmbeddingLoss": nn.CosineEmbeddingLoss,
        }

    @hookimpl
    def validate_loss_config(self, loss_name: str, config: dict[str, Any]) -> bool:
        """Validate loss function configuration."""
        # Most loss functions can be instantiated without parameters
        # More specific validation can be added here as needed

        if (
            loss_name == "CrossEntropyLoss"
            and "weight" in config
            and config["weight"] is not None
        ):
            # Could validate weight tensor dimensions
            pass

        return True
