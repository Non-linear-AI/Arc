"""Plugin manager for Arc Graph.

This module provides the central PluginManager class that orchestrates
plugin discovery, loading, and lifecycle management.
"""

from __future__ import annotations

import logging
from importlib.metadata import entry_points
from typing import TYPE_CHECKING, Any

import pluggy

from .hookspecs import (
    DataSourceHookSpec,
    ExportHookSpec,
    LayerHookSpec,
    LifecycleHookSpec,
    LossHookSpec,
    OptimizerHookSpec,
    ProcessorHookSpec,
    ValidatorHookSpec,
)

if TYPE_CHECKING:
    from ..ml.layers import ArcLayerBase

logger = logging.getLogger(__name__)

# Global plugin manager instance
_plugin_manager: PluginManager | None = None


class PluginManager:
    """Central manager for Arc Graph plugins.

    The PluginManager handles:
    - Plugin discovery from entry points
    - Plugin loading and registration
    - Component registry management
    - Error handling and logging
    - Plugin lifecycle events
    """

    def __init__(self):
        """Initialize the plugin manager."""
        # Create pluggy plugin manager
        self.pm = pluggy.PluginManager("arc")

        # Register hook specifications
        self.pm.add_hookspecs(LayerHookSpec)
        self.pm.add_hookspecs(ProcessorHookSpec)
        self.pm.add_hookspecs(OptimizerHookSpec)
        self.pm.add_hookspecs(LossHookSpec)
        self.pm.add_hookspecs(ValidatorHookSpec)
        self.pm.add_hookspecs(DataSourceHookSpec)
        self.pm.add_hookspecs(ExportHookSpec)
        self.pm.add_hookspecs(LifecycleHookSpec)

        # Component registries
        self._layers: dict[str, type[ArcLayerBase]] = {}
        self._processors: dict[str, type] = {}
        self._optimizers: dict[str, type] = {}
        self._losses: dict[str, type] = {}
        self._validators: dict[str, type] = {}
        self._data_sources: dict[str, type] = {}
        self._exporters: dict[str, type] = {}

        # Plugin metadata
        self._plugin_metadata: dict[str, dict[str, Any]] = {}
        self._loaded_plugins: set[str] = set()

        # Register built-in plugins
        self._register_builtin_plugins()

    def discover_plugins(self, group: str = "arc.plugins") -> None:
        """Discover and load plugins from entry points.

        Args:
            group: Entry point group name to discover plugins from
        """
        logger.info(f"Discovering plugins from entry point group: {group}")

        try:
            # Get entry points for the specified group
            eps = entry_points(group=group)

            for ep in eps:
                try:
                    self._load_plugin_from_entry_point(ep)
                except Exception as e:
                    logger.error(f"Failed to load plugin {ep.name}: {e}")

        except Exception as e:
            logger.error(f"Failed to discover plugins: {e}")

    def _load_plugin_from_entry_point(self, entry_point) -> None:
        """Load a plugin from an entry point.

        Args:
            entry_point: The entry point to load
        """
        plugin_name = entry_point.name

        if plugin_name in self._loaded_plugins:
            logger.debug(f"Plugin {plugin_name} already loaded")
            return

        logger.info(f"Loading plugin: {plugin_name}")

        try:
            # Load the plugin module/class
            plugin = entry_point.load()

            # Register the plugin
            self.pm.register(plugin, name=plugin_name)
            self._loaded_plugins.add(plugin_name)

            # Store metadata if available
            if hasattr(plugin, "__plugin_metadata__"):
                self._plugin_metadata[plugin_name] = plugin.__plugin_metadata__

            logger.info(f"Successfully loaded plugin: {plugin_name}")

        except Exception as e:
            logger.error(f"Failed to load plugin {plugin_name}: {e}")
            raise

    def register_plugin(self, plugin: Any, name: str | None = None) -> None:
        """Manually register a plugin.

        Args:
            plugin: The plugin instance or class
            name: Optional name for the plugin
        """
        if name is None:
            name = getattr(plugin, "__name__", str(plugin))

        if name in self._loaded_plugins:
            logger.warning(f"Plugin {name} is already registered")
            return

        logger.info(f"Registering plugin: {name}")

        try:
            self.pm.register(plugin, name=name)
            self._loaded_plugins.add(name)

            # Store metadata if available
            if hasattr(plugin, "__plugin_metadata__"):
                self._plugin_metadata[name] = plugin.__plugin_metadata__

        except Exception as e:
            logger.error(f"Failed to register plugin {name}: {e}")
            raise

    def refresh_registries(self) -> None:
        """Refresh all component registries by calling plugin hooks."""
        logger.debug("Refreshing component registries")

        try:
            # Clear existing registries
            self._layers.clear()
            self._processors.clear()
            self._optimizers.clear()
            self._losses.clear()
            self._validators.clear()
            self._data_sources.clear()
            self._exporters.clear()

            # Collect components from all plugins
            self._collect_layers()
            self._collect_processors()
            self._collect_optimizers()
            self._collect_losses()
            self._collect_validators()
            self._collect_data_sources()
            self._collect_exporters()

            logger.info(
                f"Refreshed registries: "
                f"{len(self._layers)} layers, "
                f"{len(self._processors)} processors, "
                f"{len(self._optimizers)} optimizers, "
                f"{len(self._losses)} losses, "
                f"{len(self._validators)} validators, "
                f"{len(self._data_sources)} data sources, "
                f"{len(self._exporters)} exporters"
            )

        except Exception as e:
            logger.error(f"Failed to refresh registries: {e}")
            raise

    def _collect_layers(self) -> None:
        """Collect layer implementations from all plugins with namespace context."""
        for plugin in self.pm.get_plugins():
            register = getattr(plugin, "register_layers", None)
            if not callable(register):
                continue
            try:
                ns = getattr(plugin, "__plugin_metadata__", {}).get("namespace", "core")
                mapping = register() or {}
                for key, layer_class in mapping.items():
                    full_key = key if "." in key else f"{ns}.{key}"
                    if full_key in self._layers:
                        logger.warning(f"Duplicate layer registration: {full_key}")
                    self._layers[full_key] = layer_class
                    logger.debug(f"Registered layer: {full_key}")
            except Exception as e:
                logger.error(f"Failed collecting layers from {plugin}: {e}")

    def _collect_processors(self) -> None:
        """Collect processor implementations with namespace context."""
        for plugin in self.pm.get_plugins():
            register = getattr(plugin, "register_processors", None)
            if not callable(register):
                continue
            try:
                ns = getattr(plugin, "__plugin_metadata__", {}).get("namespace", "core")
                mapping = register() or {}
                for key, cls in mapping.items():
                    full_key = key if "." in key else f"{ns}.{key}"
                    if full_key in self._processors:
                        logger.warning(f"Duplicate processor registration: {full_key}")
                    self._processors[full_key] = cls
                    logger.debug(f"Registered processor: {full_key}")
            except Exception as e:
                logger.error(f"Failed collecting processors from {plugin}: {e}")

    def _collect_optimizers(self) -> None:
        """Collect optimizer implementations with namespace context."""
        for plugin in self.pm.get_plugins():
            register = getattr(plugin, "register_optimizers", None)
            if not callable(register):
                continue
            try:
                ns = getattr(plugin, "__plugin_metadata__", {}).get("namespace", "core")
                mapping = register() or {}
                for key, cls in mapping.items():
                    full_key = key if "." in key else f"{ns}.{key}"
                    if full_key in self._optimizers:
                        logger.warning(f"Duplicate optimizer registration: {full_key}")
                    self._optimizers[full_key] = cls
                    logger.debug(f"Registered optimizer: {full_key}")
            except Exception as e:
                logger.error(f"Failed collecting optimizers from {plugin}: {e}")

    def _collect_losses(self) -> None:
        """Collect loss function implementations with namespace context."""
        for plugin in self.pm.get_plugins():
            register = getattr(plugin, "register_losses", None)
            if not callable(register):
                continue
            try:
                ns = getattr(plugin, "__plugin_metadata__", {}).get("namespace", "core")
                mapping = register() or {}
                for key, cls in mapping.items():
                    full_key = key if "." in key else f"{ns}.{key}"
                    if full_key in self._losses:
                        logger.warning(f"Duplicate loss registration: {full_key}")
                    self._losses[full_key] = cls
                    logger.debug(f"Registered loss: {full_key}")
            except Exception as e:
                logger.error(f"Failed collecting losses from {plugin}: {e}")

    def _collect_validators(self) -> None:
        """Collect validator implementations with namespace context."""
        for plugin in self.pm.get_plugins():
            register = getattr(plugin, "register_validators", None)
            if not callable(register):
                continue
            try:
                ns = getattr(plugin, "__plugin_metadata__", {}).get("namespace", "core")
                mapping = register() or {}
                for key, cls in mapping.items():
                    full_key = key if "." in key else f"{ns}.{key}"
                    if full_key in self._validators:
                        logger.warning(f"Duplicate validator registration: {full_key}")
                    self._validators[full_key] = cls
                    logger.debug(f"Registered validator: {full_key}")
            except Exception as e:
                logger.error(f"Failed collecting validators from {plugin}: {e}")

    def _collect_data_sources(self) -> None:
        """Collect data source implementations with namespace context."""
        for plugin in self.pm.get_plugins():
            register = getattr(plugin, "register_data_sources", None)
            if not callable(register):
                continue
            try:
                ns = getattr(plugin, "__plugin_metadata__", {}).get("namespace", "core")
                mapping = register() or {}
                for key, cls in mapping.items():
                    full_key = key if "." in key else f"{ns}.{key}"
                    if full_key in self._data_sources:
                        logger.warning(
                            f"Duplicate data source registration: {full_key}"
                        )
                    self._data_sources[full_key] = cls
                    logger.debug(f"Registered data source: {full_key}")
            except Exception as e:
                logger.error(f"Failed collecting data sources from {plugin}: {e}")

    def _collect_exporters(self) -> None:
        """Collect exporter implementations with namespace context."""
        for plugin in self.pm.get_plugins():
            register = getattr(plugin, "register_exporters", None)
            if not callable(register):
                continue
            try:
                ns = getattr(plugin, "__plugin_metadata__", {}).get("namespace", "core")
                mapping = register() or {}
                for key, cls in mapping.items():
                    full_key = key if "." in key else f"{ns}.{key}"
                    if full_key in self._exporters:
                        logger.warning(f"Duplicate exporter registration: {full_key}")
                    self._exporters[full_key] = cls
                    logger.debug(f"Registered exporter: {full_key}")
            except Exception as e:
                logger.error(f"Failed collecting exporters from {plugin}: {e}")

    def _register_builtin_plugins(self) -> None:
        """Register built-in Arc Graph plugins."""
        from .builtin import (
            BuiltinLayerPlugin,
            BuiltinLossPlugin,
            BuiltinOptimizerPlugin,
            BuiltinProcessorPlugin,
        )

        # Register built-in plugins
        self.register_plugin(BuiltinLayerPlugin(), "builtin_layers")
        self.register_plugin(BuiltinOptimizerPlugin(), "builtin_optimizers")
        self.register_plugin(BuiltinLossPlugin(), "builtin_losses")
        self.register_plugin(BuiltinProcessorPlugin(), "builtin_processors")

    # Public API methods

    def get_layers(self) -> dict[str, type]:
        """Get all registered layer implementations."""
        return self._layers.copy()

    def get_layer(self, name: str) -> type | None:
        """Get a specific layer implementation by name."""
        # If no namespace specified, prefer core.<name> then plain name
        if "." not in name:
            return self._layers.get(f"core.{name}") or self._layers.get(name)
        return self._layers.get(name)

    def get_processors(self) -> dict[str, type]:
        """Get all registered processor implementations."""
        return self._processors.copy()

    def get_processor(self, name: str) -> type | None:
        """Get a specific processor implementation by name."""
        if "." not in name:
            return self._processors.get(f"core.{name}") or self._processors.get(name)
        return self._processors.get(name)

    def get_optimizers(self) -> dict[str, type]:
        """Get all registered optimizer implementations."""
        return self._optimizers.copy()

    def get_optimizer(self, name: str) -> type | None:
        """Get a specific optimizer implementation by name."""
        if "." not in name:
            return self._optimizers.get(f"core.{name}") or self._optimizers.get(name)
        return self._optimizers.get(name)

    def get_losses(self) -> dict[str, type]:
        """Get all registered loss function implementations."""
        return self._losses.copy()

    def get_loss(self, name: str) -> type | None:
        """Get a specific loss function implementation by name."""
        if "." not in name:
            return self._losses.get(f"core.{name}") or self._losses.get(name)
        return self._losses.get(name)

    def get_validators(self) -> dict[str, type]:
        """Get all registered validator implementations."""
        return self._validators.copy()

    def get_validator(self, name: str) -> type | None:
        """Get a specific validator implementation by name."""
        if "." not in name:
            return self._validators.get(f"core.{name}") or self._validators.get(name)
        return self._validators.get(name)

    def get_data_sources(self) -> dict[str, type]:
        """Get all registered data source implementations."""
        return self._data_sources.copy()

    def get_data_source(self, name: str) -> type | None:
        """Get a specific data source implementation by name."""
        if "." not in name:
            return self._data_sources.get(f"core.{name}") or self._data_sources.get(
                name
            )
        return self._data_sources.get(name)

    def get_exporters(self) -> dict[str, type]:
        """Get all registered exporter implementations."""
        return self._exporters.copy()

    def get_exporter(self, name: str) -> type | None:
        """Get a specific exporter implementation by name."""
        if "." not in name:
            return self._exporters.get(f"core.{name}") or self._exporters.get(name)
        return self._exporters.get(name)

    def get_plugin_metadata(self, plugin_name: str) -> dict[str, Any]:
        """Get metadata for a specific plugin."""
        return self._plugin_metadata.get(plugin_name, {})

    def list_plugins(self) -> list[str]:
        """Get list of all loaded plugin names."""
        return list(self._loaded_plugins)

    def validate_component_config(
        self, component_type: str, name: str, config: dict[str, Any]
    ) -> bool:
        """Validate configuration for a component.

        Args:
            component_type: Type of component (layer, processor, etc.)
            name: Name of the specific component
            config: Configuration to validate

        Returns:
            True if configuration is valid
        """
        try:
            if component_type == "layer":
                results = self.pm.hook.validate_layer_config(
                    layer_type=name, config=config
                )
            elif component_type == "processor":
                results = self.pm.hook.validate_processor_config(
                    op_name=name, config=config
                )
            elif component_type == "optimizer":
                results = self.pm.hook.validate_optimizer_config(
                    optimizer_name=name, config=config
                )
            elif component_type == "loss":
                results = self.pm.hook.validate_loss_config(
                    loss_name=name, config=config
                )
            elif component_type == "data_source":
                results = self.pm.hook.validate_data_source_config(
                    source_name=name, config=config
                )
            elif component_type == "export":
                results = self.pm.hook.validate_export_config(
                    format_name=name, config=config
                )
            else:
                logger.warning(
                    f"Unknown component type for validation: {component_type}"
                )
                return True

            # If any plugin returned False, validation failed
            return all(result is not False for result in results if result is not None)

        except Exception as e:
            logger.error(f"Error validating {component_type} config for {name}: {e}")
            return False

    # Lifecycle hooks

    def call_pre_model_build(
        self, model_spec: Any, context: dict[str, Any]
    ) -> dict[str, Any]:
        """Call pre-model-build hooks."""
        results = self.pm.hook.pre_model_build(model_spec=model_spec, context=context)

        # Merge all context modifications
        for result in results:
            if result:
                context.update(result)

        return context

    def call_post_model_build(self, model: Any, context: dict[str, Any]) -> None:
        """Call post-model-build hooks."""
        self.pm.hook.post_model_build(model=model, context=context)

    def call_pre_validation(
        self, model_spec: Any, context: dict[str, Any]
    ) -> dict[str, Any]:
        """Call pre-validation hooks."""
        results = self.pm.hook.pre_validation(model_spec=model_spec, context=context)

        # Merge all context modifications
        for result in results:
            if result:
                context.update(result)

        return context

    def call_post_validation(
        self, model_spec: Any, results: list[str], context: dict[str, Any]
    ) -> list[str]:
        """Call post-validation hooks."""
        hook_results = self.pm.hook.post_validation(
            model_spec=model_spec, results=results, context=context
        )

        # Use the last non-None result
        for result in reversed(hook_results):
            if result is not None:
                return result

        return results

    def call_validate_model(
        self, model_spec: Any, context: dict[str, Any]
    ) -> list[str]:
        """Call model validation hooks."""
        all_errors = []

        results = self.pm.hook.validate_model(model_spec=model_spec, context=context)
        for errors in results:
            if errors:
                all_errors.extend(errors)

        return all_errors


def get_plugin_manager() -> PluginManager:
    """Get the global plugin manager instance."""
    global _plugin_manager

    if _plugin_manager is None:
        _plugin_manager = PluginManager()

        # Auto-discover plugins and refresh registries
        _plugin_manager.discover_plugins()
        _plugin_manager.refresh_registries()

    return _plugin_manager


def reset_plugin_manager() -> None:
    """Reset the global plugin manager (mainly for testing)."""
    global _plugin_manager
    _plugin_manager = None
