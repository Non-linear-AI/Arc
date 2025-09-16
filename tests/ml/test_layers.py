import pytest
import torch

from src.arc.ml.layers import (
    DropoutLayer,
    LinearLayer,
    ReLULayer,
    SigmoidLayer,
    get_layer_class,
)
from src.arc.plugins import get_plugin_manager


class TestLayerImplementations:
    """Test individual layer implementations."""

    def test_linear_layer(self):
        """Test LinearLayer implementation."""
        layer = LinearLayer(in_features=10, out_features=5)

        # Test forward pass
        x = torch.randn(32, 10)
        output = layer(x)

        assert output.shape == (32, 5)
        assert isinstance(output, torch.Tensor)

    def test_relu_layer(self):
        """Test ReLULayer implementation."""
        layer = ReLULayer()

        # Test with negative and positive values
        x = torch.tensor([[-1.0, 2.0, -3.0, 4.0]])
        output = layer(x)

        expected = torch.tensor([[0.0, 2.0, 0.0, 4.0]])
        assert torch.allclose(output, expected)

    def test_sigmoid_layer(self):
        """Test SigmoidLayer implementation."""
        layer = SigmoidLayer()

        # Test with known values
        x = torch.tensor([[0.0, 1.0, -1.0]])
        output = layer(x)

        # Sigmoid(0) = 0.5, Sigmoid(1) ≈ 0.731, Sigmoid(-1) ≈ 0.269
        assert output.shape == x.shape
        assert torch.all(output >= 0) and torch.all(output <= 1)
        assert torch.allclose(output[0, 0], torch.tensor(0.5), atol=1e-6)

    def test_dropout_layer(self):
        """Test DropoutLayer implementation."""
        layer = DropoutLayer(p=0.5)
        layer.train()  # Enable training mode for dropout

        x = torch.ones(100, 10)
        output = layer(x)

        assert output.shape == x.shape
        # In training mode, some values should be zeroed
        assert torch.any(output == 0)

    def test_dropout_layer_eval_mode(self):
        """Test DropoutLayer in evaluation mode."""
        layer = DropoutLayer(p=0.5)
        layer.eval()  # Enable evaluation mode

        x = torch.ones(100, 10)
        output = layer(x)

        # In eval mode, output should be unchanged
        assert torch.allclose(output, x)


class TestLayerRegistry:
    """Test layer registry functionality."""

    def test_get_layer_class_valid(self):
        """Test getting valid layer class."""
        layer_class = get_layer_class("core.Linear")
        assert layer_class == LinearLayer

        layer_class = get_layer_class("core.ReLU")
        assert layer_class == ReLULayer

    def test_get_layer_class_invalid(self):
        """Test getting invalid layer class."""
        with pytest.raises(ValueError, match="Unknown layer type"):
            get_layer_class("invalid.Layer")

    def test_register_new_layer_via_plugin(self):
        """Test registering a new layer via plugin system."""

        import pluggy

        class CustomLayer(torch.nn.Module):
            pass

        hookimpl = pluggy.HookimplMarker("arc")

        class TestPlugin:
            @hookimpl
            def register_layers(self):
                return {"custom.Layer": CustomLayer}

        pm = get_plugin_manager()
        pm.register_plugin(TestPlugin(), name="test_custom_layer_plugin")
        pm.refresh_registries()

        assert pm.get_layer("custom.Layer") is CustomLayer
        assert get_layer_class("custom.Layer") is CustomLayer

    def test_layer_registry_contents(self):
        """Test that core plugin exposes expected layers."""
        expected_layers = [
            "core.Linear",
            "core.ReLU",
            "core.Sigmoid",
            "core.Dropout",
            "core.BatchNorm1d",
        ]

        pm = get_plugin_manager()
        available = pm.get_layers().keys()
        for layer_type in expected_layers:
            assert layer_type in available
