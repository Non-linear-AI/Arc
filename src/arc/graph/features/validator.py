"""Validation for features specifications."""

from __future__ import annotations

from typing import Any

from arc.graph.features.components import get_processor_class, validate_processor_params


class FeaturesValidationError(ValueError):
    """Exception raised for features validation errors."""

    pass


def _require(obj: dict[str, Any], key: str, msg: str | None = None) -> Any:
    """Helper function for requiring dictionary keys."""
    if key not in obj:
        raise FeaturesValidationError(msg or f"Missing required field: {key}")
    return obj[key]


def validate_features_dict(data: dict[str, Any]) -> None:
    """Validate a parsed YAML dict for features specification.

    Args:
        data: Dictionary containing features specification

    Raises:
        FeaturesValidationError: If validation fails
    """
    # Validate feature columns
    feature_columns = _require(
        data, "feature_columns", "features.feature_columns required"
    )
    if not isinstance(feature_columns, list) or not feature_columns:
        raise FeaturesValidationError(
            "features.feature_columns must be a non-empty list"
        )

    for i, col in enumerate(feature_columns):
        if not isinstance(col, str):
            raise FeaturesValidationError(
                f"features.feature_columns[{i}] must be a string"
            )

    # Validate target columns (optional)
    if "target_columns" in data and data["target_columns"] is not None:
        target_columns = data["target_columns"]
        if not isinstance(target_columns, list):
            raise FeaturesValidationError("features.target_columns must be a list")

        for i, col in enumerate(target_columns):
            if not isinstance(col, str):
                raise FeaturesValidationError(
                    f"features.target_columns[{i}] must be a string"
                )

    # Validate processors (optional)
    if "processors" in data and data["processors"] is not None:
        processors = data["processors"]
        if not isinstance(processors, list):
            raise FeaturesValidationError("features.processors must be a list")

        processor_names = set()
        for i, processor in enumerate(processors):
            if not isinstance(processor, dict):
                raise FeaturesValidationError(
                    f"features.processors[{i}] must be a mapping"
                )

            # Validate required fields
            proc_name = _require(
                processor, "name", f"features.processors[{i}].name required"
            )
            proc_op = _require(processor, "op", f"features.processors[{i}].op required")

            # Check for duplicate processor names
            if proc_name in processor_names:
                raise FeaturesValidationError(f"Duplicate processor name: {proc_name}")
            processor_names.add(proc_name)

            # Validate processor type is supported
            try:
                get_processor_class(proc_op)
            except ValueError as e:
                raise FeaturesValidationError(f"features.processors[{i}]: {e}") from e

            # Validate processor parameters if present
            if "params" in processor and processor["params"] is not None:
                try:
                    validate_processor_params(proc_op, processor["params"])
                except ValueError as e:
                    raise FeaturesValidationError(
                        f"features.processors[{i}].params: {e}"
                    ) from e

            # Validate train_only field
            if "train_only" in processor:
                train_only = processor["train_only"]
                if not isinstance(train_only, bool):
                    raise FeaturesValidationError(
                        f"features.processors[{i}].train_only must be a boolean"
                    )

            # Validate inputs mapping
            if "inputs" in processor and processor["inputs"] is not None:
                inputs = processor["inputs"]
                if not isinstance(inputs, dict):
                    raise FeaturesValidationError(
                        f"features.processors[{i}].inputs must be a mapping"
                    )

            # Validate outputs mapping
            if "outputs" in processor and processor["outputs"] is not None:
                outputs = processor["outputs"]
                if not isinstance(outputs, dict):
                    raise FeaturesValidationError(
                        f"features.processors[{i}].outputs must be a mapping"
                    )


def validate_features_components(data: dict[str, Any]) -> list[str]:
    """Validate features components against available types.

    Args:
        data: Features specification dictionary

    Returns:
        List of validation errors (empty if valid)
    """
    errors = []

    if "processors" not in data:
        return errors

    processors = data["processors"]
    if not isinstance(processors, list):
        return errors

    for i, processor in enumerate(processors):
        if not isinstance(processor, dict):
            continue

        proc_op = processor.get("op")
        if not proc_op:
            continue

        try:
            get_processor_class(proc_op)
        except ValueError:
            errors.append(f"Unsupported processor type in processors[{i}]: {proc_op}")

        # Validate parameters if present
        if "params" in processor and processor["params"] is not None:
            try:
                validate_processor_params(proc_op, processor["params"])
            except ValueError as e:
                errors.append(f"Invalid parameters for processors[{i}]: {e}")

    return errors


def validate_column_references(
    data: dict[str, Any], available_columns: list[str]
) -> list[str]:
    """Validate column references against available columns.

    Args:
        data: Features specification dictionary
        available_columns: List of available column names

    Returns:
        List of validation errors (empty if valid)
    """
    errors = []
    available_set = set(available_columns)

    # Check feature columns
    if "feature_columns" in data:
        for col in data["feature_columns"]:
            if col not in available_set:
                errors.append(f"Feature column not found: {col}")

    # Check target columns
    if "target_columns" in data and data["target_columns"]:
        for col in data["target_columns"]:
            if col not in available_set:
                errors.append(f"Target column not found: {col}")

    return errors
