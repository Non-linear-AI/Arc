"""Model specification and components for Arc-Graph."""

from .components import CORE_LAYERS, get_layer_class
from .spec import GraphNode, ModelInput, ModelSpec
from .validator import ModelValidationError, validate_model_dict


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
    "ModelSpec",
    "get_layer_class",
    "CORE_LAYERS",
    "validate_model_dict",
    "ModelValidationError",
    "load_model_from_yaml",
]
