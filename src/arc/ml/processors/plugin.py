"""Built-in processor plugin for Arc Graph."""

from __future__ import annotations

from typing import Any

import pluggy

from .builtin import (
    CategoricalEncodingProcessor,
    MinMaxNormalizationProcessor,
    RobustNormalizationProcessor,
    StandardNormalizationProcessor,
)

# Create hook implementation marker
hookimpl = pluggy.HookimplMarker("arc")


class BuiltinProcessorPlugin:
    """Plugin that registers built-in data processors."""

    __plugin_metadata__ = {
        "name": "builtin_processors",
        "version": "1.0.0",
        "description": "Built-in data processors for Arc Graph",
    }

    @hookimpl
    def register_processors(self) -> dict[str, type]:
        """Register built-in processor implementations.

        Returns:
            Dictionary mapping processor names to processor classes
        """
        return {
            # Standard normalization processors
            "normalize.standard": StandardNormalizationProcessor,
            "normalize.zscore": StandardNormalizationProcessor,  # Alias
            "normalize.minmax": MinMaxNormalizationProcessor,
            "normalize.robust": RobustNormalizationProcessor,
            # Encoding processors
            "encode.categorical": CategoricalEncodingProcessor,
            "encode.onehot": CategoricalEncodingProcessor,  # Alias
            # Core namespace for backward compatibility
            "core.StandardNormalization": StandardNormalizationProcessor,
            "core.MinMaxNormalization": MinMaxNormalizationProcessor,
            "core.RobustNormalization": RobustNormalizationProcessor,
            "core.CategoricalEncoding": CategoricalEncodingProcessor,
        }

    @hookimpl
    def validate_processor_config(self, op_name: str, config: dict[str, Any]) -> bool:
        """Validate processor configuration.

        Args:
            op_name: Name of the processor operation
            config: Configuration parameters

        Returns:
            True if configuration is valid, False otherwise
        """
        # Get the processor type from our registry
        processors = self.register_processors()
        if op_name not in processors:
            return True  # Not our processor, let others handle it

        processor_class = processors[op_name]

        try:
            # Basic validation - check required parameters
            if "table_name" not in config:
                return False

            if "columns" not in config:
                return False

            columns = config["columns"]
            if not isinstance(columns, list) or not columns:
                return False

            # Processor-specific validation
            if processor_class == MinMaxNormalizationProcessor:
                feature_range = config.get("feature_range", (0.0, 1.0))
                if (
                    not isinstance(feature_range, (list, tuple))
                    or len(feature_range) != 2
                    or feature_range[0] >= feature_range[1]
                ):
                    return False

            elif processor_class == CategoricalEncodingProcessor:
                handle_unknown = config.get("handle_unknown", "ignore")
                if handle_unknown not in ("ignore", "error"):
                    return False

            return True

        except Exception:
            return False
