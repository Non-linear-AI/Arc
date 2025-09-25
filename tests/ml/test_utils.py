import pytest
import torch

from arc.ml.utils import (
    auto_detect_input_size,
    count_parameters,
    create_sample_data,
    resolve_variable_references,
    validate_tensor_shape,
)


class TestUtilityFunctions:
    """Test ML utility functions."""

    def test_auto_detect_input_size(self):
        """Test automatic input size detection."""
        # Test 2D tensor
        data = torch.randn(100, 10)
        size = auto_detect_input_size(data)
        assert size == 10

        # Test different sizes
        data = torch.randn(50, 25)
        size = auto_detect_input_size(data)
        assert size == 25

    def test_auto_detect_input_size_invalid_shape(self):
        """Test auto detection with invalid tensor shape."""
        # Test 1D tensor
        data = torch.randn(100)
        with pytest.raises(ValueError, match="Expected 2D tensor"):
            auto_detect_input_size(data)

        # Test 3D tensor
        data = torch.randn(10, 5, 3)
        with pytest.raises(ValueError, match="Expected 2D tensor"):
            auto_detect_input_size(data)

    def test_validate_tensor_shape_valid(self):
        """Test tensor shape validation with valid shapes."""
        # Test exact match
        tensor = torch.randn(32, 10)
        validate_tensor_shape(tensor, [32, 10])  # Should not raise

        # Test with None for variable dimensions
        tensor = torch.randn(64, 5, 3)
        validate_tensor_shape(tensor, [None, 5, 3])  # Should not raise

        # Test all None
        tensor = torch.randn(16, 8)
        validate_tensor_shape(tensor, [None, None])  # Should not raise

    def test_validate_tensor_shape_invalid(self):
        """Test tensor shape validation with invalid shapes."""
        tensor = torch.randn(32, 10)

        # Wrong number of dimensions
        with pytest.raises(ValueError, match="dimension mismatch"):
            validate_tensor_shape(tensor, [32])

        # Wrong dimension size
        with pytest.raises(ValueError, match="Shape mismatch at dimension"):
            validate_tensor_shape(tensor, [32, 5])

    def test_resolve_variable_references(self):
        """Test resolving variable references in parameters."""
        var_registry = {
            "vars.n_features": 10,
            "vars.hidden_size": 64,
            "vars.learning_rate": 0.001,
        }

        # Test with variable references
        params = {
            "in_features": "vars.n_features",
            "out_features": "vars.hidden_size",
            "bias": True,
            "dropout_rate": 0.1,  # Non-variable parameter
        }

        resolved = resolve_variable_references(params, var_registry)

        assert resolved["in_features"] == 10
        assert resolved["out_features"] == 64
        assert resolved["bias"] is True
        assert resolved["dropout_rate"] == 0.1

    def test_resolve_variable_references_undefined(self):
        """Test resolving undefined variable references."""
        var_registry = {"vars.n_features": 10}

        params = {"in_features": "vars.undefined_var"}

        with pytest.raises(ValueError, match="Cannot resolve variable reference"):
            resolve_variable_references(params, var_registry)

    def test_resolve_variable_references_no_vars(self):
        """Test resolving parameters with no variable references."""
        var_registry = {"vars.n_features": 10}

        params = {
            "in_features": 5,
            "out_features": 3,
            "bias": False,
        }

        resolved = resolve_variable_references(params, var_registry)

        # Should return unchanged
        assert resolved == params

    def test_create_sample_data(self):
        """Test sample data creation."""
        features, targets = create_sample_data(100, 5, binary_classification=True)

        assert features.shape == (100, 5)
        assert targets.shape == (100,)
        assert features.dtype == torch.float32
        assert targets.dtype == torch.float32

    def test_count_parameters(self):
        """Test parameter counting for models."""
        # Create a simple model
        model = torch.nn.Sequential(
            torch.nn.Linear(10, 5),  # 10*5 + 5 = 55 parameters
            torch.nn.Linear(5, 1),  # 5*1 + 1 = 6 parameters
        )

        param_count = count_parameters(model)
        assert param_count == 61  # 55 + 6

    def test_count_parameters_no_grad(self):
        """Test parameter counting with non-trainable parameters."""
        model = torch.nn.Linear(10, 5)

        # Disable gradients for all parameters
        for param in model.parameters():
            param.requires_grad = False

        param_count = count_parameters(model)
        assert param_count == 0  # No trainable parameters

    def test_count_parameters_mixed(self):
        """Test parameter counting with mixed trainable/non-trainable."""
        model = torch.nn.Sequential(
            torch.nn.Linear(10, 5),
            torch.nn.Linear(5, 1),
        )

        # Disable gradients for first layer only
        for param in model[0].parameters():
            param.requires_grad = False

        param_count = count_parameters(model)
        assert param_count == 6  # Only second layer parameters
