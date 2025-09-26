"""Validation for model specifications."""

from __future__ import annotations

from typing import Any

from arc.graph.model.components import (
    get_component_class_or_function,
    validate_component_params,
)


class ModelValidationError(ValueError):
    """Exception raised for model validation errors."""

    pass


def _require(obj: dict[str, Any], key: str, msg: str | None = None) -> Any:
    """Helper function for requiring dictionary keys."""
    if key not in obj:
        raise ModelValidationError(msg or f"Missing required field: {key}")
    return obj[key]


def resolve_node_reference(ref: str) -> tuple[str, str, int | None]:
    """Parse node references like 'node.output', 'node.output.0', etc.

    Args:
        ref: Node reference string

    Returns:
        Tuple of (node_name, attribute, tuple_index)

    Raises:
        ModelValidationError: If reference format is invalid
    """
    parts = ref.split(".")
    if len(parts) == 1:
        # Direct node reference (assume .output)
        return parts[0], "output", None
    elif len(parts) == 2:
        # node.attribute
        return parts[0], parts[1], None
    elif len(parts) == 3 and parts[2].isdigit():
        # node.attribute.0 (tuple indexing)
        return parts[0], parts[1], int(parts[2])
    else:
        raise ModelValidationError(f"Invalid node reference format: {ref}")


def validate_module_definition(module_name: str, module_data: dict[str, Any]) -> None:
    """Validate a module definition.

    Args:
        module_name: Name of the module
        module_data: Module definition data

    Raises:
        ModelValidationError: If module definition is invalid
    """
    # Validate required fields
    inputs = _require(module_data, "inputs", f"module.{module_name}.inputs required")
    graph = _require(module_data, "graph", f"module.{module_name}.graph required")
    outputs = _require(module_data, "outputs", f"module.{module_name}.outputs required")

    # Validate inputs is a list of strings
    if not isinstance(inputs, list) or not all(isinstance(inp, str) for inp in inputs):
        raise ModelValidationError(
            f"module.{module_name}.inputs must be a list of strings"
        )

    # Validate graph is a list
    if not isinstance(graph, list):
        raise ModelValidationError(f"module.{module_name}.graph must be a list")

    # Validate outputs is a dict
    if not isinstance(outputs, dict):
        raise ModelValidationError(f"module.{module_name}.outputs must be a mapping")

    # Validate internal graph nodes (we'll do full validation later)
    available_nodes = set(inputs)  # Input parameters are available as nodes
    for i, node in enumerate(graph):
        if not isinstance(node, dict):
            raise ModelValidationError(
                f"module.{module_name}.graph[{i}] must be a mapping"
            )

        node_name = _require(
            node, "name", f"module.{module_name}.graph[{i}].name required"
        )
        available_nodes.add(node_name)

    # Validate output references point to valid nodes
    for output_name, source_ref in outputs.items():
        try:
            node_name, _, _ = resolve_node_reference(source_ref)
            if node_name not in available_nodes:
                raise ModelValidationError(
                    f"module.{module_name}.outputs.{output_name} references "
                    f"undefined node: {source_ref}"
                )
        except ModelValidationError:
            raise  # Re-raise reference parsing errors
        except Exception as e:
            raise ModelValidationError(
                f"module.{module_name}.outputs.{output_name} has invalid reference: "
                f"{source_ref}"
            ) from e


def validate_model_dict(data: dict[str, Any]) -> None:
    """Validate a parsed YAML dict for model specification with modules support.

    Args:
        data: Dictionary containing model specification

    Raises:
        ModelValidationError: If validation fails
    """
    # Validate inputs section
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

    # Validate modules section if present
    modules = data.get("modules", {})
    if modules is None:
        modules = {}  # Treat None as empty dict for backward compatibility
    if not isinstance(modules, dict):
        raise ModelValidationError("model.modules must be a mapping")

    for module_name, module_data in modules.items():
        validate_module_definition(module_name, module_data)

    # Validate main graph
    graph = _require(data, "graph", "model.graph section required")
    if not isinstance(graph, list):
        raise ModelValidationError("model.graph must be a list")

    available_modules = set(modules.keys())
    validate_graph_nodes(graph, set(inputs.keys()), available_modules, "model")

    # Validate outputs
    outputs = _require(data, "outputs", "model.outputs section required")
    if not isinstance(outputs, dict):
        raise ModelValidationError("model.outputs must be a mapping")

    # Collect all available nodes from inputs and graph
    all_available_nodes = set(inputs.keys())
    for node in graph:
        if isinstance(node, dict) and "name" in node:
            all_available_nodes.add(node["name"])

    for output_name, source_ref in outputs.items():
        try:
            node_name, _, _ = resolve_node_reference(source_ref)
            if node_name not in all_available_nodes:
                raise ModelValidationError(
                    f"model.outputs.{output_name} references undefined node: "
                    f"{source_ref}"
                )
        except ModelValidationError:
            raise  # Re-raise reference parsing errors
        except Exception as e:
            raise ModelValidationError(
                f"model.outputs.{output_name} has invalid reference: {source_ref}"
            ) from e


def validate_graph_nodes(
    graph: list[dict[str, Any]],
    initial_nodes: set[str],
    available_modules: set[str],
    context_prefix: str = "",
) -> None:
    """Validate a list of graph nodes with module and component support.

    Args:
        graph: List of graph node dictionaries
        initial_nodes: Set of initially available node names (e.g., inputs)
        available_modules: Set of available custom module names
        context_prefix: Prefix for error messages (e.g., "model" or "module.MyModule")

    Raises:
        ModelValidationError: If validation fails
    """
    available_nodes = initial_nodes.copy()

    for i, node in enumerate(graph):
        if not isinstance(node, dict):
            raise ModelValidationError(f"{context_prefix}.graph[{i}] must be a mapping")

        node_name = _require(node, "name", f"{context_prefix}.graph[{i}].name required")
        node_type = _require(node, "type", f"{context_prefix}.graph[{i}].type required")

        # Validate node type is supported
        try:
            get_component_class_or_function(node_type)
        except ValueError as e:
            # Special handling for custom modules
            if node_type.startswith("module."):
                module_name = node_type[7:]  # Remove "module." prefix
                if module_name not in available_modules:
                    raise ModelValidationError(
                        f"{context_prefix}.graph[{i}]: {e}"
                    ) from e
            else:
                raise ModelValidationError(f"{context_prefix}.graph[{i}]: {e}") from e

        # Validate node parameters if present
        if "params" in node and node["params"] is not None:
            try:
                validate_component_params(node_type, node["params"])

                # Special validation for arc.stack
                if node_type == "arc.stack":
                    module_name = node["params"]["module"]
                    if module_name not in available_modules:
                        raise ValueError(
                            f"arc.stack references undefined module: {module_name}"
                        )

            except ValueError as e:
                raise ModelValidationError(
                    f"{context_prefix}.graph[{i}].params: {e}"
                ) from e

        # Validate node inputs reference available nodes
        if "inputs" in node and node["inputs"] is not None:
            node_inputs = node["inputs"]

            # Support both dict and list input formats
            if isinstance(node_inputs, dict):
                # Dict format: {arg_name: source_ref}
                for input_key, source_ref in node_inputs.items():
                    validate_node_reference(
                        source_ref,
                        available_nodes,
                        f"{context_prefix}.graph[{i}].inputs.{input_key}",
                    )
            elif isinstance(node_inputs, list):
                # List format: [source_ref1, source_ref2, ...]
                for j, source_ref in enumerate(node_inputs):
                    validate_node_reference(
                        source_ref,
                        available_nodes,
                        f"{context_prefix}.graph[{i}].inputs[{j}]",
                    )
            else:
                raise ModelValidationError(
                    f"{context_prefix}.graph[{i}].inputs must be a mapping or list"
                )

        # Add this node to available nodes
        available_nodes.add(node_name)


def validate_node_reference(
    source_ref: str, available_nodes: set[str], error_context: str
) -> None:
    """Validate that a node reference points to an available node.

    Args:
        source_ref: Node reference string (e.g., "layer1.output", "layer2.output.0")
        available_nodes: Set of available node names
        error_context: Context string for error messages

    Raises:
        ModelValidationError: If reference is invalid
    """
    try:
        node_name, _, _ = resolve_node_reference(source_ref)
        if node_name not in available_nodes:
            raise ModelValidationError(
                f"{error_context} references undefined node: {source_ref}"
            )
    except ModelValidationError:
        raise  # Re-raise reference parsing errors
    except Exception as e:
        raise ModelValidationError(
            f"{error_context} has invalid reference: {source_ref}"
        ) from e


def validate_model_components(data: dict[str, Any]) -> list[str]:
    """Validate model components against available types.

    Args:
        data: Model specification dictionary

    Returns:
        List of validation errors (empty if valid)
    """
    errors = []

    # Check modules if present
    modules = data.get("modules", {})
    available_modules = set(modules.keys())

    for module_name, module_data in modules.items():
        try:
            validate_module_definition(module_name, module_data)
        except ModelValidationError as e:
            errors.append(f"Module {module_name}: {e}")

    # Check main graph
    if "graph" not in data:
        return errors

    for i, node in enumerate(data["graph"]):
        if not isinstance(node, dict):
            continue

        node_type = node.get("type")
        if not node_type:
            continue

        try:
            get_component_class_or_function(node_type)
        except ValueError:
            # Check for custom modules
            if node_type.startswith("module."):
                module_name = node_type[7:]
                if module_name not in available_modules:
                    errors.append(
                        f"Unsupported component type in graph[{i}]: {node_type}"
                    )
            else:
                errors.append(f"Unsupported component type in graph[{i}]: {node_type}")

        # Validate parameters if present
        if "params" in node and node["params"] is not None:
            try:
                validate_component_params(node_type, node["params"])
            except ValueError as e:
                errors.append(f"Invalid parameters for graph[{i}]: {e}")

    return errors
