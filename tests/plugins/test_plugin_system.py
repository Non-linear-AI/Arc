"""Tests for the Arc Graph plugin system."""

import pytest

from arc.ml.layers import get_layer_class
from arc.plugins import PluginManager, get_plugin_manager


class TestPluginSystem:
    """Test the plugin system functionality."""

    def test_plugin_manager_singleton(self):
        """Test that plugin manager is a singleton."""
        pm1 = get_plugin_manager()
        pm2 = get_plugin_manager()
        assert pm1 is pm2

    def test_builtin_plugins_loaded(self):
        """Test that built-in plugins are loaded."""
        pm = get_plugin_manager()

        # Check that built-in plugins are registered
        plugins = pm.list_plugins()
        assert "builtin_layers" in plugins
        assert "builtin_optimizers" in plugins
        assert "builtin_losses" in plugins

    def test_layer_registration(self):
        """Test that layers are registered through plugins."""
        pm = get_plugin_manager()

        # Get all registered layers
        layers = pm.get_layers()

        # Check that some core layers are present under core namespace
        assert "core.Linear" in layers
        assert "core.ReLU" in layers
        assert "core.Dropout" in layers
        assert "core.Embedding" in layers
        assert "core.LSTM" in layers

    def test_optimizer_registration(self):
        """Test that optimizers are registered through plugins."""
        pm = get_plugin_manager()

        # Get all registered optimizers
        optimizers = pm.get_optimizers()

        # Check that common optimizers are present
        assert "SGD" in optimizers
        assert "Adam" in optimizers
        assert "AdamW" in optimizers
        assert "RMSprop" in optimizers

    def test_loss_registration(self):
        """Test that loss functions are registered through plugins."""
        pm = get_plugin_manager()

        # Get all registered losses
        losses = pm.get_losses()

        # Check that common losses are present
        assert "MSELoss" in losses
        assert "CrossEntropyLoss" in losses
        assert "BCELoss" in losses

    def test_layer_retrieval(self):
        """Test that layers can be retrieved by name."""
        pm = get_plugin_manager()

        # Test retrieval of specific layers
        linear_layer = pm.get_layer("core.Linear")
        assert linear_layer is not None

        # Default namespace resolves to core plugin
        assert pm.get_layer("Linear") is linear_layer

        # Test non-existent layer
        nonexistent = pm.get_layer("NonExistent")
        assert nonexistent is None

    def test_get_layer_class_integration(self):
        """Test that get_layer_class works with the plugin system."""
        # Test core layer type
        linear_class = get_layer_class("core.Linear")
        assert linear_class.__name__ == "LinearLayer"

        # Unqualified name defaults to core
        assert get_layer_class("Linear") is linear_class

        # Test error for unknown layer
        with pytest.raises(ValueError, match="Unknown layer type"):
            get_layer_class("UnknownLayer")

    def test_component_validation(self):
        """Test plugin component configuration validation."""
        pm = get_plugin_manager()

        # Test valid layer config
        valid_config = {"in_features": 10, "out_features": 5}
        assert pm.validate_component_config("layer", "core.Linear", valid_config)

        # Test invalid layer config
        invalid_config = {"in_features": 10}  # Missing out_features
        assert not pm.validate_component_config("layer", "core.Linear", invalid_config)

        # Test valid optimizer config
        optimizer_config = {"lr": 0.001}
        assert pm.validate_component_config("optimizer", "Adam", optimizer_config)

        # Test invalid optimizer config
        invalid_optimizer_config = {}  # Missing lr
        assert not pm.validate_component_config(
            "optimizer", "Adam", invalid_optimizer_config
        )

    def test_plugin_metadata(self):
        """Test that plugin metadata is accessible."""
        pm = get_plugin_manager()

        # Check metadata for built-in layer plugin
        metadata = pm.get_plugin_metadata("builtin_layers")
        assert metadata["name"] == "builtin_layers"
        assert "version" in metadata
        assert "description" in metadata

    def test_fresh_plugin_manager(self):
        """Test creating a new plugin manager instance."""
        # Create a fresh plugin manager
        pm = PluginManager()

        # It should start with empty registries
        assert len(pm.get_layers()) == 0
        assert len(pm.get_optimizers()) == 0
        assert len(pm.get_losses()) == 0

        # After discovering plugins and refreshing, it should have components
        pm.discover_plugins()
        pm.refresh_registries()

        assert len(pm.get_layers()) > 0
        assert len(pm.get_optimizers()) > 0
        assert len(pm.get_losses()) > 0
