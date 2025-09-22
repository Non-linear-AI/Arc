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
    """Plugin that registers PyTorch layer types with exact PyTorch names."""

    __plugin_metadata__ = {
        "name": "torch_layers",
        "version": "1.0.0",
        "description": "PyTorch layer types using torch.nn namespace",
        "namespace": "torch.nn",
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
            # Basic layers with exact PyTorch names
            "Linear": LinearLayer,
            "ReLU": ReLULayer,
            "Sigmoid": SigmoidLayer,
            "Dropout": DropoutLayer,
            "BatchNorm1d": BatchNorm1dLayer,
            "LayerNorm": LayerNormLayer,
            # Embedding layers
            "Embedding": EmbeddingLayer,
            # Attention and transformer layers
            "MultiheadAttention": MultiHeadAttentionLayer,  # Exact PyTorch name
            "TransformerEncoderLayer": TransformerEncoderLayerCustom,
            "PositionalEncoding": PositionalEncodingLayer,  # Custom but clear name
            # Sequence modeling layers
            "LSTM": LSTMLayer,
            "GRU": GRULayer,
            # Routing and combination layers (custom)
            "Concatenate": ConcatenateLayer,
            "Add": AddLayer,
        }

    @hookimpl
    def validate_layer_config(self, layer_type: str, config: dict[str, Any]) -> bool:
        """Validate layer configuration for PyTorch layers."""
        # Basic validation for PyTorch layers using exact PyTorch parameter names
        if layer_type in ("torch.nn.Linear", "Linear"):
            required = ["in_features", "out_features"]
            return all(param in config for param in required)
        elif layer_type in ("torch.nn.Embedding", "Embedding"):
            required = ["num_embeddings", "embedding_dim"]
            return all(param in config for param in required)
        elif layer_type in ("torch.nn.LSTM", "LSTM") or layer_type in (
            "torch.nn.GRU",
            "GRU",
        ):
            required = ["input_size", "hidden_size"]
            return all(param in config for param in required)
        elif layer_type in ("torch.nn.MultiheadAttention", "MultiheadAttention"):
            required = ["embed_dim", "num_heads"]
            return all(param in config for param in required)
        elif layer_type in (
            "torch.nn.TransformerEncoderLayer",
            "TransformerEncoderLayer",
        ):
            required = ["d_model", "nhead"]
            return all(param in config for param in required)
        elif layer_type in ("torch.nn.PositionalEncoding", "PositionalEncoding"):
            required = ["d_model"]
            return all(param in config for param in required)
        elif layer_type in ("torch.nn.BatchNorm1d", "BatchNorm1d"):
            required = ["num_features"]
            return all(param in config for param in required)
        elif layer_type in ("torch.nn.LayerNorm", "LayerNorm"):
            required = ["normalized_shape"]
            return all(param in config for param in required)

        # For other layers, basic validation passes
        return True


class BuiltinOptimizerPlugin:
    """Plugin that registers PyTorch optimizers with exact PyTorch names."""

    __plugin_metadata__ = {
        "name": "torch_optimizers",
        "version": "1.0.0",
        "description": "PyTorch optimizers using torch.optim namespace",
        "namespace": "torch.optim",
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
            "Adamax": optim.Adamax,  # Exact PyTorch name
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
    """Plugin that registers PyTorch loss functions with exact PyTorch names."""

    __plugin_metadata__ = {
        "name": "torch_losses",
        "version": "1.0.0",
        "description": "PyTorch loss functions using torch.nn namespace",
        "namespace": "torch.nn",
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


class BuiltinProcessorPlugin:
    """Plugin that registers Arc Graph's built-in data processors."""

    __plugin_metadata__ = {
        "name": "builtin_processors",
        "version": "1.0.0",
        "description": "Built-in data processors for Arc Graph",
        "namespace": "core",
    }

    @hookimpl
    def register_processors(self) -> dict[str, type]:
        """Register built-in processor implementations."""
        from ..ml.processors.builtin import (
            CategoricalEncodingProcessor,
            MinMaxNormalizationProcessor,
            RobustNormalizationProcessor,
            StandardNormalizationProcessor,
        )

        return {
            # Standard normalization processors
            "StandardNormalization": StandardNormalizationProcessor,
            "MinMaxNormalization": MinMaxNormalizationProcessor,
            "RobustNormalization": RobustNormalizationProcessor,
            "CategoricalEncoding": CategoricalEncodingProcessor,
            # Aliases with descriptive names
            "ZScoreNormalization": StandardNormalizationProcessor,
            "OneHotEncoding": CategoricalEncodingProcessor,
        }

    @hookimpl
    def validate_processor_config(self, op_name: str, config: dict[str, Any]) -> bool:
        """Validate processor configuration."""
        processors = self.register_processors()
        if op_name not in processors:
            return True  # Not our processor

        try:
            # Basic validation
            if "table_name" not in config or "columns" not in config:
                return False

            columns = config["columns"]
            if not isinstance(columns, list) or not columns:
                return False

            # Processor-specific validation
            processor_class = processors[op_name]

            if processor_class.__name__ == "MinMaxNormalizationProcessor":
                feature_range = config.get("feature_range", (0.0, 1.0))
                if (
                    not isinstance(feature_range, (list, tuple))
                    or len(feature_range) != 2
                    or feature_range[0] >= feature_range[1]
                ):
                    return False

            elif processor_class.__name__ == "CategoricalEncodingProcessor":
                handle_unknown = config.get("handle_unknown", "ignore")
                if handle_unknown not in ("ignore", "error"):
                    return False

            return True

        except Exception:
            return False
