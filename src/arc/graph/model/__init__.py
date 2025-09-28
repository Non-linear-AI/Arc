"""Model specification and components for Arc-Graph."""

from arc.graph.model.builder import (
    ArcGraphModel,
    build_model_from_file,
    build_model_from_spec,
    build_model_from_yaml,
)
from arc.graph.model.components import (
    CORE_LAYERS,
    TORCH_FUNCTIONS,
    get_component_class_or_function,
    get_layer_class,
    get_supported_component_types,
    validate_component_params,
)
from arc.graph.model.spec import (
    GraphNode,
    LossSpec,
    ModelInput,
    ModelSpec,
    ModuleDefinition,
)
from arc.graph.model.validator import ModelValidationError, validate_model_dict


def load_model_from_yaml(file_path: str) -> ModelSpec:
    """Load model specification from YAML file.

    Args:
        file_path: Path to YAML file

    Returns:
        Loaded and validated ModelSpec
    """
    return ModelSpec.from_yaml_file(file_path)


__all__ = [
    "ModelInput",
    "GraphNode",
    "ModuleDefinition",
    "LossSpec",
    "ModelSpec",
    "ArcGraphModel",
    "build_model_from_spec",
    "build_model_from_yaml",
    "build_model_from_file",
    "get_layer_class",
    "get_component_class_or_function",
    "validate_component_params",
    "get_supported_component_types",
    "CORE_LAYERS",
    "TORCH_FUNCTIONS",
    "validate_model_dict",
    "ModelValidationError",
    "load_model_from_yaml",
]
