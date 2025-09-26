"""Tests for Arc-Graph component system."""

import pytest
import torch

from arc.graph.model.components import (
    CORE_LAYERS,
    TORCH_FUNCTIONS,
    get_component_class_or_function,
    get_supported_component_types,
    validate_component_params,
)


class TestComponentRegistry:
    """Test component registry functionality."""

    def test_core_layers_registry(self):
        """Test that core layers are properly registered."""
        # Test a few key layer types
        assert "torch.nn.Linear" in CORE_LAYERS
        assert "torch.nn.Conv2d" in CORE_LAYERS
        assert "torch.nn.LSTM" in CORE_LAYERS
        assert "torch.nn.MultiheadAttention" in CORE_LAYERS

        # Verify they map to actual PyTorch classes
        assert CORE_LAYERS["torch.nn.Linear"] is torch.nn.Linear
        assert CORE_LAYERS["torch.nn.Conv2d"] is torch.nn.Conv2d

    def test_torch_functions_registry(self):
        """Test that PyTorch functions are properly registered."""
        # Test core tensor operations
        assert "torch.cat" in TORCH_FUNCTIONS
        assert "torch.stack" in TORCH_FUNCTIONS
        assert "torch.mean" in TORCH_FUNCTIONS
        assert "torch.matmul" in TORCH_FUNCTIONS

        # Test functional operations
        assert "torch.nn.functional.relu" in TORCH_FUNCTIONS
        assert "torch.nn.functional.softmax" in TORCH_FUNCTIONS

        # Verify they map to actual functions
        assert TORCH_FUNCTIONS["torch.cat"] is torch.cat
        assert TORCH_FUNCTIONS["torch.mean"] is torch.mean

    def test_get_supported_component_types(self):
        """Test getting all supported component types."""
        component_types = get_supported_component_types()

        assert "modules" in component_types
        assert "functions" in component_types
        assert "special" in component_types
        assert "custom" in component_types

        # Check that we have a reasonable number of components
        assert len(component_types["modules"]) > 50  # Lots of torch.nn modules
        assert len(component_types["functions"]) > 30  # Many torch functions
        assert "arc.stack" in component_types["special"]
        assert "module.*" in component_types["custom"]


class TestComponentLookup:
    """Test component lookup functionality."""

    def test_get_module_component(self):
        """Test getting PyTorch module components."""
        component, kind = get_component_class_or_function("torch.nn.Linear")
        assert component is torch.nn.Linear
        assert kind == "module"

        component, kind = get_component_class_or_function("torch.nn.Conv2d")
        assert component is torch.nn.Conv2d
        assert kind == "module"

    def test_get_function_component(self):
        """Test getting PyTorch function components."""
        component, kind = get_component_class_or_function("torch.cat")
        assert component is torch.cat
        assert kind == "function"

        component, kind = get_component_class_or_function("torch.nn.functional.relu")
        assert component is torch.nn.functional.relu
        assert kind == "function"

    def test_get_custom_module_component(self):
        """Test getting custom module references."""
        component, kind = get_component_class_or_function("module.MyCustomModule")
        assert component == "module.MyCustomModule"
        assert kind == "custom_module"

    def test_get_arc_stack_component(self):
        """Test getting arc.stack special component."""
        component, kind = get_component_class_or_function("arc.stack")
        assert component == "arc.stack"
        assert kind == "stack"

    def test_unsupported_component_error(self):
        """Test error for unsupported component types."""
        with pytest.raises(ValueError, match="Unsupported component type"):
            get_component_class_or_function("unsupported.component")


class TestComponentValidation:
    """Test component parameter validation."""

    def test_validate_linear_params(self):
        """Test validation of Linear layer parameters."""
        # Valid parameters
        valid_params = {"in_features": 10, "out_features": 5, "bias": True}
        assert validate_component_params("torch.nn.Linear", valid_params)

        # Missing required parameters
        with pytest.raises(ValueError, match="Missing required parameters"):
            validate_component_params("torch.nn.Linear", {"bias": True})

    def test_validate_conv2d_params(self):
        """Test validation of Conv2d layer parameters."""
        # Valid parameters
        valid_params = {
            "in_channels": 3,
            "out_channels": 64,
            "kernel_size": 3,
            "padding": 1
        }
        assert validate_component_params("torch.nn.Conv2d", valid_params)

        # Missing required parameters
        with pytest.raises(ValueError, match="Missing required parameters"):
            validate_component_params("torch.nn.Conv2d", {"padding": 1})

    def test_validate_lstm_params(self):
        """Test validation of LSTM layer parameters."""
        # Valid parameters
        valid_params = {
            "input_size": 100,
            "hidden_size": 256,
            "num_layers": 2,
            "batch_first": True
        }
        assert validate_component_params("torch.nn.LSTM", valid_params)

        # Missing required parameters
        with pytest.raises(ValueError, match="Missing required parameters"):
            validate_component_params("torch.nn.LSTM", {"num_layers": 2})

    def test_validate_multihead_attention_params(self):
        """Test validation of MultiheadAttention parameters."""
        # Valid parameters
        valid_params = {"embed_dim": 512, "num_heads": 8}
        assert validate_component_params("torch.nn.MultiheadAttention", valid_params)

        # Missing required parameters
        with pytest.raises(ValueError, match="Missing required parameters"):
            validate_component_params("torch.nn.MultiheadAttention", {"dropout": 0.1})

    def test_validate_transformer_encoder_layer_params(self):
        """Test validation of TransformerEncoderLayer parameters."""
        # Valid parameters
        valid_params = {
            "d_model": 512,
            "nhead": 8,
            "dim_feedforward": 2048
        }
        assert validate_component_params("torch.nn.TransformerEncoderLayer", valid_params)

        # Missing required parameters
        with pytest.raises(ValueError, match="Missing required parameters"):
            validate_component_params("torch.nn.TransformerEncoderLayer", {"dim_feedforward": 2048})

    def test_validate_embedding_params(self):
        """Test validation of Embedding layer parameters."""
        # Valid parameters
        valid_params = {"num_embeddings": 10000, "embedding_dim": 300}
        assert validate_component_params("torch.nn.Embedding", valid_params)

        # Missing required parameters
        with pytest.raises(ValueError, match="Missing required parameters"):
            validate_component_params("torch.nn.Embedding", {"padding_idx": 0})

    def test_validate_normalization_params(self):
        """Test validation of normalization layer parameters."""
        # BatchNorm1d
        valid_params = {"num_features": 128}
        assert validate_component_params("torch.nn.BatchNorm1d", valid_params)

        # LayerNorm
        valid_params = {"normalized_shape": [512]}
        assert validate_component_params("torch.nn.LayerNorm", valid_params)

        # Missing required parameters
        with pytest.raises(ValueError, match="Missing required parameters"):
            validate_component_params("torch.nn.BatchNorm1d", {"eps": 1e-5})

    def test_validate_arc_stack_params(self):
        """Test validation of arc.stack parameters."""
        # Valid parameters
        valid_params = {"module": "ResidualBlock", "count": 6}
        assert validate_component_params("arc.stack", valid_params)

        # Missing required parameters
        with pytest.raises(ValueError, match="Missing required parameters"):
            validate_component_params("arc.stack", {"count": 3})

        # Invalid count
        with pytest.raises(ValueError, match="count must be a positive integer"):
            validate_component_params("arc.stack", {"module": "Block", "count": 0})

        with pytest.raises(ValueError, match="count must be a positive integer"):
            validate_component_params("arc.stack", {"module": "Block", "count": -1})

    def test_validate_function_params(self):
        """Test validation of function parameters (most accept any params)."""
        # Functions generally accept any parameters, so should not raise errors
        assert validate_component_params("torch.cat", {"dim": 1})
        assert validate_component_params("torch.mean", {"dim": 1, "keepdim": True})
        assert validate_component_params("torch.nn.functional.relu", {"inplace": False})

    def test_validate_unknown_component_params(self):
        """Test validation of unknown component types (should pass)."""
        # Unknown components assume parameters are valid
        assert validate_component_params("some.unknown.component", {"param": "value"})


class TestComponentIntegration:
    """Test component integration with real PyTorch operations."""

    def test_linear_component_instantiation(self):
        """Test that Linear components can be instantiated and used."""
        component, kind = get_component_class_or_function("torch.nn.Linear")
        assert kind == "module"

        # Create instance
        layer = component(in_features=10, out_features=5)
        assert isinstance(layer, torch.nn.Linear)

        # Test forward pass
        x = torch.randn(2, 10)
        output = layer(x)
        assert output.shape == (2, 5)

    def test_function_component_usage(self):
        """Test that function components can be called directly."""
        component, kind = get_component_class_or_function("torch.cat")
        assert kind == "function"

        # Test function call
        x1 = torch.randn(2, 3)
        x2 = torch.randn(2, 3)
        output = component([x1, x2], dim=1)
        assert output.shape == (2, 6)

    def test_functional_component_usage(self):
        """Test that functional components work correctly."""
        component, kind = get_component_class_or_function("torch.nn.functional.relu")
        assert kind == "function"

        # Test function call
        x = torch.randn(2, 5)
        x[0, 0] = -1.0  # Ensure some negative values
        output = component(x)
        assert output.shape == (2, 5)
        assert (output >= 0).all()  # ReLU should zero out negative values

    def test_complex_component_instantiation(self):
        """Test complex components like MultiheadAttention."""
        component, kind = get_component_class_or_function("torch.nn.MultiheadAttention")
        assert kind == "module"

        # Create instance
        attention = component(embed_dim=64, num_heads=8)
        assert isinstance(attention, torch.nn.MultiheadAttention)

        # Test forward pass
        x = torch.randn(10, 2, 64)  # seq_len, batch_size, embed_dim
        output, weights = attention(x, x, x)
        assert output.shape == (10, 2, 64)
        assert weights.shape == (2, 10, 10)  # batch_size, seq_len, seq_len


class TestBackwardCompatibility:
    """Test backward compatibility features."""

    def test_legacy_layer_class_function(self):
        """Test that legacy get_layer_class function still works."""
        from arc.graph.model.components import get_layer_class

        # Should work for module types
        layer_class = get_layer_class("torch.nn.Linear")
        assert layer_class is torch.nn.Linear

        # Should fail for function types
        with pytest.raises(ValueError, match="is not a PyTorch module"):
            get_layer_class("torch.cat")

    def test_legacy_layer_params_validation(self):
        """Test that legacy validate_layer_params function still works."""
        from arc.graph.model.components import validate_layer_params

        # Should work the same as validate_component_params for modules
        assert validate_layer_params("torch.nn.Linear", {"in_features": 10, "out_features": 5})

        with pytest.raises(ValueError, match="Missing required parameters"):
            validate_layer_params("torch.nn.Linear", {"bias": True})

    def test_legacy_supported_types_function(self):
        """Test that legacy get_supported_layer_types function still works."""
        from arc.graph.model.components import get_supported_layer_types

        layer_types = get_supported_layer_types()
        assert isinstance(layer_types, list)
        assert "torch.nn.Linear" in layer_types
        assert "torch.nn.Conv2d" in layer_types
        # Should only contain module types, not functions
        assert "torch.cat" not in layer_types