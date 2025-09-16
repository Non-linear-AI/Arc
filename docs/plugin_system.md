# Arc Graph Plugin System

Arc Graph includes a powerful plugin system built on top of the [pluggy](https://pluggy.readthedocs.io/) library, which is the same plugin system used by pytest. This system allows developers to extend Arc Graph with custom layers, optimizers, loss functions, validators, data sources, and exporters.

## Overview

The plugin system enables:

- **Layer plugins**: Add custom PyTorch layer implementations
- **Processor plugins**: Add custom data transformation operations
- **Optimizer plugins**: Add custom PyTorch optimizers
- **Loss function plugins**: Add custom loss functions
- **Validator plugins**: Add custom validation rules
- **Data source plugins**: Add custom data connectors
- **Export plugins**: Add custom model serialization formats
- **Lifecycle plugins**: Hook into various stages of Arc Graph operations

## Quick Start

### Using Existing Plugins

```python
from arc.plugins import get_plugin_manager

# Get the global plugin manager
pm = get_plugin_manager()

# List all available layers
layers = pm.get_layers()
print(f"Available layers: {list(layers.keys())}")

# Get a specific layer
linear_layer = pm.get_layer("Linear")
if linear_layer:
    layer_instance = linear_layer(in_features=10, out_features=5)
```

### Creating a Custom Plugin

```python
import pluggy
from arc.ml.layers import ArcLayerBase
from arc.plugins.hookspecs import LayerHookSpec

# Create hook implementation marker
hookimpl = pluggy.HookimplMarker("arc")

class MyCustomLayer(ArcLayerBase):
    def __init__(self, custom_param: int, **params):
        super().__init__(**params)
        self.custom_param = custom_param
        # Your layer implementation here

    def _forward_single(self, x):
        # Your forward pass implementation
        return x * self.custom_param

class MyPlugin:
    __plugin_metadata__ = {
        "name": "my_plugin",
        "version": "1.0.0",
        "description": "My custom Arc Graph plugin"
    }

    @hookimpl
    def register_layers(self):
        return {
            "custom.MyLayer": MyCustomLayer,
            "MyLayer": MyCustomLayer,  # Alias
        }

    @hookimpl
    def validate_layer_config(self, layer_type: str, config: dict):
        if layer_type in ("custom.MyLayer", "MyLayer"):
            return "custom_param" in config
        return True

# Register the plugin
from arc.plugins import get_plugin_manager
pm = get_plugin_manager()
pm.register_plugin(MyPlugin(), "my_plugin")
pm.refresh_registries()
```

## Plugin Architecture

### Hook Specifications

The plugin system defines several hook specifications (hookspecs) that plugins can implement:

#### LayerHookSpec
- `register_layers()`: Register custom layer implementations
- `validate_layer_config()`: Validate layer configuration

#### ProcessorHookSpec
- `register_processors()`: Register custom data processors
- `validate_processor_config()`: Validate processor configuration

#### OptimizerHookSpec
- `register_optimizers()`: Register custom optimizers
- `validate_optimizer_config()`: Validate optimizer configuration

#### LossHookSpec
- `register_losses()`: Register custom loss functions
- `validate_loss_config()`: Validate loss function configuration

#### ValidatorHookSpec
- `register_validators()`: Register custom validators
- `validate_model()`: Perform custom model validation

#### DataSourceHookSpec
- `register_data_sources()`: Register custom data sources
- `validate_data_source_config()`: Validate data source configuration

#### ExportHookSpec
- `register_exporters()`: Register custom export formats
- `validate_export_config()`: Validate export configuration

#### LifecycleHookSpec
- `pre_model_build()`: Called before model building
- `post_model_build()`: Called after model building
- `pre_validation()`: Called before validation
- `post_validation()`: Called after validation

### Plugin Discovery

Plugins can be discovered automatically through Python entry points. Add this to your `pyproject.toml`:

```toml
[project.entry-points."arc.plugins"]
my_plugin = "my_package.my_plugin:MyPlugin"
```

### Plugin Manager API

The `PluginManager` class provides the main interface:

```python
from arc.plugins import PluginManager, get_plugin_manager

# Get singleton instance
pm = get_plugin_manager()

# Or create a new instance
pm = PluginManager()

# Discover plugins from entry points
pm.discover_plugins()

# Manually register a plugin
pm.register_plugin(plugin_instance, "plugin_name")

# Refresh component registries
pm.refresh_registries()

# Get components
layers = pm.get_layers()
optimizers = pm.get_optimizers()
losses = pm.get_losses()

# Get specific component
layer_class = pm.get_layer("LayerName")

# Validate configuration
is_valid = pm.validate_component_config("layer", "LayerName", config)

# Plugin metadata
metadata = pm.get_plugin_metadata("plugin_name")
```

## Built-in Plugins

Arc Graph includes built-in plugins that register all the standard components:

- **BuiltinLayerPlugin**: Registers all standard Arc Graph layers
- **BuiltinOptimizerPlugin**: Registers standard PyTorch optimizers
- **BuiltinLossPlugin**: Registers standard PyTorch loss functions

## Best Practices

1. **Use namespaced names**: Prefix your layer types with your plugin namespace (e.g., `"myplugin.CustomLayer"`)

2. **Provide aliases**: Offer convenient aliases for commonly used layers

3. **Include metadata**: Add `__plugin_metadata__` to your plugin classes

4. **Validate configurations**: Implement configuration validation hooks

5. **Handle errors gracefully**: Use try/except blocks in your implementations

6. **Follow Arc Graph conventions**: Inherit from `ArcLayerBase` for custom layers

## Example Plugins

See `examples/custom_plugin_example.py` for a complete example of creating custom layers including:

- GELU activation layer
- SwiGLU activation layer
- RMS normalization layer
- Rotary positional embedding layer

## Integration with Arc Graph

The plugin system is fully integrated with Arc Graph's model building process. When you specify a layer type in your Arc Graph YAML specification, the system will:

1. Check the legacy registry for backward compatibility
2. Query all registered plugins for the layer type
3. Return the first matching implementation found
4. Validate the layer configuration using plugin validators

This ensures that custom plugins work seamlessly with existing Arc Graph workflows.