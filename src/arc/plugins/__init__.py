"""Arc Graph Plugin System.

This module provides a pluggy-based plugin system that allows users to extend
Arc Graph with custom layers, processors, validators, and other components.

Key Features:
- Layer plugins for custom PyTorch layer implementations
- Processor plugins for custom data transformation operations
- Optimizer and loss function plugins
- Validator plugins for custom validation rules
- Data source plugins for custom data connectors
- Export plugins for custom model serialization formats

Example Usage:
    # Register a plugin manager
    from arc.plugins import get_plugin_manager

    pm = get_plugin_manager()

    # Discover and load plugins
    pm.discover_plugins()

    # Get all registered layers
    layers = pm.get_layers()
"""

from arc.plugins.hookspecs import (
    DataSourceHookSpec,
    ExportHookSpec,
    LayerHookSpec,
    LossHookSpec,
    OptimizerHookSpec,
    ProcessorHookSpec,
    ValidatorHookSpec,
)
from arc.plugins.manager import PluginManager, get_plugin_manager

__all__ = [
    "PluginManager",
    "get_plugin_manager",
    "LayerHookSpec",
    "ProcessorHookSpec",
    "OptimizerHookSpec",
    "LossHookSpec",
    "ValidatorHookSpec",
    "DataSourceHookSpec",
    "ExportHookSpec",
]
