"""Hook specifications for Arc Graph plugins.

This module defines the plugin interfaces using pluggy hookspecs.
Each hookspec defines what methods a plugin must implement for a specific
component type.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import pluggy

if TYPE_CHECKING:
    pass

# Create hookspec markers
hookspec = pluggy.HookspecMarker("arc")


class LayerHookSpec:
    """Hook specification for layer plugins.

    Layer plugins can register custom PyTorch layer implementations
    that can be used in Arc Graph model specifications.
    """

    @hookspec
    def register_layers(self) -> dict[str, type]:
        """Register custom layer implementations.

        Returns:
            Dictionary mapping layer type names to layer classes.
            Example: {"custom.MyLayer": MyLayerClass}
        """

    @hookspec
    def validate_layer_config(self, layer_type: str, config: dict[str, Any]) -> bool:
        """Validate layer configuration before instantiation.

        Args:
            layer_type: Type name of the layer
            config: Configuration parameters for the layer

        Returns:
            True if configuration is valid, False otherwise
        """


class ProcessorHookSpec:
    """Hook specification for processor plugins.

    Processor plugins can register custom data transformation operations
    for the Arc Graph features pipeline.
    """

    @hookspec
    def register_processors(self) -> dict[str, type]:
        """Register custom processor implementations.

        Returns:
            Dictionary mapping processor operation names to processor classes.
            Example: {"transform.custom_normalize": CustomNormalizeProcessor}
        """

    @hookspec
    def validate_processor_config(self, op_name: str, config: dict[str, Any]) -> bool:
        """Validate processor configuration.

        Args:
            op_name: Name of the processor operation
            config: Configuration parameters for the processor

        Returns:
            True if configuration is valid, False otherwise
        """


class OptimizerHookSpec:
    """Hook specification for optimizer plugins.

    Optimizer plugins can register custom PyTorch optimizers.
    """

    @hookspec
    def register_optimizers(self) -> dict[str, type]:
        """Register custom optimizer implementations.

        Returns:
            Dictionary mapping optimizer names to optimizer classes.
            Example: {"custom.AdaBound": AdaBoundOptimizer}
        """

    @hookspec
    def validate_optimizer_config(
        self, optimizer_name: str, config: dict[str, Any]
    ) -> bool:
        """Validate optimizer configuration.

        Args:
            optimizer_name: Name of the optimizer
            config: Configuration parameters for the optimizer

        Returns:
            True if configuration is valid, False otherwise
        """


class LossHookSpec:
    """Hook specification for loss function plugins.

    Loss function plugins can register custom PyTorch loss functions.
    """

    @hookspec
    def register_losses(self) -> dict[str, type]:
        """Register custom loss function implementations.

        Returns:
            Dictionary mapping loss function names to loss classes.
            Example: {"custom.FocalLoss": FocalLossClass}
        """

    @hookspec
    def validate_loss_config(self, loss_name: str, config: dict[str, Any]) -> bool:
        """Validate loss function configuration.

        Args:
            loss_name: Name of the loss function
            config: Configuration parameters for the loss function

        Returns:
            True if configuration is valid, False otherwise
        """


class ValidatorHookSpec:
    """Hook specification for validator plugins.

    Validator plugins can register custom validation rules for Arc Graph models.
    """

    @hookspec
    def register_validators(self) -> dict[str, type]:
        """Register custom validator implementations.

        Returns:
            Dictionary mapping validator names to validator classes.
            Example: {"custom.performance_validator": PerformanceValidator}
        """

    @hookspec
    def validate_model(self, model_spec: Any, context: dict[str, Any]) -> list[str]:
        """Perform custom model validation.

        Args:
            model_spec: The Arc Graph model specification
            context: Additional context for validation

        Returns:
            List of validation error messages (empty if valid)
        """


class DataSourceHookSpec:
    """Hook specification for data source plugins.

    Data source plugins can register custom data connectors and loaders.
    """

    @hookspec
    def register_data_sources(self) -> dict[str, type]:
        """Register custom data source implementations.

        Returns:
            Dictionary mapping data source names to data source classes.
            Example: {"custom.s3_connector": S3DataConnector}
        """

    @hookspec
    def validate_data_source_config(
        self, source_name: str, config: dict[str, Any]
    ) -> bool:
        """Validate data source configuration.

        Args:
            source_name: Name of the data source
            config: Configuration parameters for the data source

        Returns:
            True if configuration is valid, False otherwise
        """


class ExportHookSpec:
    """Hook specification for export format plugins.

    Export plugins can register custom model serialization formats.
    """

    @hookspec
    def register_exporters(self) -> dict[str, type]:
        """Register custom export format implementations.

        Returns:
            Dictionary mapping export format names to exporter classes.
            Example: {"custom.tensorrt": TensorRTExporter}
        """

    @hookspec
    def validate_export_config(self, format_name: str, config: dict[str, Any]) -> bool:
        """Validate export configuration.

        Args:
            format_name: Name of the export format
            config: Configuration parameters for the export

        Returns:
            True if configuration is valid, False otherwise
        """


class LifecycleHookSpec:
    """Hook specification for lifecycle plugins.

    Lifecycle plugins can hook into various stages of Arc Graph operations.
    """

    @hookspec
    def pre_model_build(
        self, model_spec: Any, context: dict[str, Any]
    ) -> dict[str, Any]:
        """Called before model building starts.

        Args:
            model_spec: The Arc Graph model specification
            context: Build context

        Returns:
            Modified context (can modify or add to context)
        """

    @hookspec
    def post_model_build(self, model: Any, context: dict[str, Any]) -> None:
        """Called after model building completes.

        Args:
            model: The built PyTorch model
            context: Build context
        """

    @hookspec
    def pre_validation(
        self, model_spec: Any, context: dict[str, Any]
    ) -> dict[str, Any]:
        """Called before validation starts.

        Args:
            model_spec: The Arc Graph model specification
            context: Validation context

        Returns:
            Modified context
        """

    @hookspec
    def post_validation(
        self, model_spec: Any, results: list[str], context: dict[str, Any]
    ) -> list[str]:
        """Called after validation completes.

        Args:
            model_spec: The Arc Graph model specification
            results: Validation results (list of error messages)
            context: Validation context

        Returns:
            Modified validation results
        """
