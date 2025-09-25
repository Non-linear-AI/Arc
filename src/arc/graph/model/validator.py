"""Validation for model specifications."""

from __future__ import annotations

from typing import Any

from arc.graph.model.components import get_layer_class, validate_layer_params


class ModelValidationError(ValueError):
    """Exception raised for model validation errors."""

    pass


def _require(obj: dict[str, Any], key: str, msg: str | None = None) -> Any:
    """Helper function for requiring dictionary keys."""
    if key not in obj:
        raise ModelValidationError(msg or f"Missing required field: {key}")
    return obj[key]


def validate_model_dict(data: dict[str, Any]) -> None:
    """Validate a parsed YAML dict for model specification.

    Args:
        data: Dictionary containing model specification

    Raises:
        ModelValidationError: If validation fails
    """
    # Validate inputs
    inputs = _require(data, "inputs", "model.inputs section required")
    if not isinstance(inputs, dict):
        raise ModelValidationError("model.inputs must be a mapping")

    for input_name, input_spec in inputs.items():
        if not isinstance(input_spec, dict):
            raise ModelValidationError(f"model.inputs.{input_name} must be a mapping")
        _require(input_spec, "dtype", f"model.inputs.{input_name}.dtype required")
        shape = _require(
            input_spec, "shape", f"model.inputs.{input_name}.shape required"
        )
        if not isinstance(shape, list):
            raise ModelValidationError(
                f"model.inputs.{input_name}.shape must be a list"
            )
        # columns is optional for direct column references
        if "columns" in input_spec and not isinstance(input_spec["columns"], list):
            raise ModelValidationError(
                f"model.inputs.{input_name}.columns must be a list"
            )

    # Validate graph
    graph = _require(data, "graph", "model.graph section required")
    if not isinstance(graph, list):
        raise ModelValidationError("model.graph must be a list")

    available_nodes = set(inputs.keys())  # Input nodes are available
    for i, node in enumerate(graph):
        if not isinstance(node, dict):
            raise ModelValidationError(f"model.graph[{i}] must be a mapping")

        node_name = _require(node, "name", f"model.graph[{i}].name required")
        node_type = _require(node, "type", f"model.graph[{i}].type required")

        # Validate node type is supported
        try:
            get_layer_class(node_type)
        except ValueError as e:
            raise ModelValidationError(f"model.graph[{i}]: {e}") from e

        # Validate node parameters if present
        if "params" in node and node["params"] is not None:
            try:
                validate_layer_params(node_type, node["params"])
            except ValueError as e:
                raise ModelValidationError(f"model.graph[{i}].params: {e}") from e

        # Validate node inputs reference available nodes
        if "inputs" in node and node["inputs"] is not None:
            node_inputs = node["inputs"]
            if not isinstance(node_inputs, dict):
                raise ModelValidationError(f"model.graph[{i}].inputs must be a mapping")

            for input_key, source_ref in node_inputs.items():
                if "." in source_ref:
                    # Reference to specific output of a node (e.g., "layer1.output")
                    source_node = source_ref.split(".")[0]
                    if source_node not in available_nodes:
                        raise ModelValidationError(
                            f"model.graph[{i}].inputs.{input_key} references "
                            f"undefined node: {source_ref}"
                        )
                else:
                    # Direct reference to a node
                    if source_ref not in available_nodes:
                        raise ModelValidationError(
                            f"model.graph[{i}].inputs.{input_key} references "
                            f"undefined node: {source_ref}"
                        )

        # Add this node to available nodes
        available_nodes.add(node_name)

    # Validate outputs
    outputs = _require(data, "outputs", "model.outputs section required")
    if not isinstance(outputs, dict):
        raise ModelValidationError("model.outputs must be a mapping")

    for output_name, source_ref in outputs.items():
        if "." in source_ref:
            # Reference to specific output of a node
            source_node = source_ref.split(".")[0]
            if source_node not in available_nodes:
                raise ModelValidationError(
                    f"model.outputs.{output_name} references undefined node: "
                    f"{source_ref}"
                )
        else:
            # Direct reference to a node
            if source_ref not in available_nodes:
                raise ModelValidationError(
                    f"model.outputs.{output_name} references undefined node: "
                    f"{source_ref}"
                )


def validate_model_components(data: dict[str, Any]) -> list[str]:
    """Validate model components against available types.

    Args:
        data: Model specification dictionary

    Returns:
        List of validation errors (empty if valid)
    """
    errors = []

    if "graph" not in data:
        return errors

    for i, node in enumerate(data["graph"]):
        if not isinstance(node, dict):
            continue

        node_type = node.get("type")
        if not node_type:
            continue

        try:
            get_layer_class(node_type)
        except ValueError:
            errors.append(f"Unsupported layer type in graph[{i}]: {node_type}")

        # Validate parameters if present
        if "params" in node and node["params"] is not None:
            try:
                validate_layer_params(node_type, node["params"])
            except ValueError as e:
                errors.append(f"Invalid parameters for graph[{i}]: {e}")

    return errors
