"""Model specification for Arc-Graph."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

try:
    import yaml
except ImportError as e:
    raise RuntimeError(
        "PyYAML is required for Arc-Graph. "
        "Install with 'uv add pyyaml' or 'pip install pyyaml'."
    ) from e


@dataclass
class ModelInput:
    """Specification for model input tensor."""

    dtype: str
    shape: list[int | None | str]
    columns: list[str] | None = None  # Direct column references for data mapping


@dataclass
class GraphNode:
    """Specification for a model graph node (layer)."""

    name: str
    type: str  # Direct PyTorch class name with pytorch prefix
    params: dict[str, Any] | None = None
    inputs: dict[str, str] | None = None


@dataclass
class ModelSpec:
    """Complete model specification."""

    inputs: dict[str, ModelInput]
    graph: list[GraphNode]
    outputs: dict[str, str]

    @classmethod
    def from_yaml(cls, yaml_str: str) -> ModelSpec:
        """Parse ModelSpec from YAML string.

        Args:
            yaml_str: YAML string containing model specification

        Returns:
            ModelSpec: Parsed and validated model specification

        Raises:
            ValueError: If YAML is invalid or doesn't contain valid model spec
        """
        from .validator import validate_model_dict

        data = yaml.safe_load(yaml_str)
        if not isinstance(data, dict):
            raise ValueError("Top-level YAML must be a mapping")

        # Validate the model structure
        validate_model_dict(data)

        # Parse inputs
        inputs = {}
        for input_name, input_spec in data["inputs"].items():
            inputs[input_name] = ModelInput(
                dtype=input_spec["dtype"],
                shape=input_spec["shape"],
                columns=input_spec.get("columns"),
            )

        # Parse graph nodes
        graph = []
        for node_data in data["graph"]:
            graph.append(
                GraphNode(
                    name=node_data["name"],
                    type=node_data["type"],
                    params=node_data.get("params"),
                    inputs=node_data.get("inputs"),
                )
            )

        # Parse outputs
        outputs = data["outputs"]

        return cls(inputs=inputs, graph=graph, outputs=outputs)

    @classmethod
    def from_yaml_file(cls, path: str) -> ModelSpec:
        """Parse ModelSpec from YAML file.

        Args:
            path: Path to YAML file containing model specification

        Returns:
            ModelSpec: Parsed and validated model specification

        Raises:
            FileNotFoundError: If file doesn't exist
            ValueError: If YAML is invalid or doesn't contain valid model spec
        """
        with open(path, encoding="utf-8") as f:
            return cls.from_yaml(f.read())

    def to_yaml(self) -> str:
        """Convert ModelSpec to YAML string.

        Returns:
            YAML string representation of the model specification
        """
        return yaml.dump(asdict(self), default_flow_style=False)

    def to_yaml_file(self, path: str) -> None:
        """Save ModelSpec to YAML file.

        Args:
            path: Path to save the YAML file
        """
        with open(path, "w", encoding="utf-8") as f:
            f.write(self.to_yaml())

    def get_input_names(self) -> list[str]:
        """Get list of input names.

        Returns:
            List of input names
        """
        return list(self.inputs.keys())

    def get_output_names(self) -> list[str]:
        """Get list of output names.

        Returns:
            List of output names
        """
        return list(self.outputs.keys())

    def get_layer_names(self) -> list[str]:
        """Get list of layer names in execution order.

        Returns:
            List of layer names
        """
        return [node.name for node in self.graph]

    def get_layer_types(self) -> dict[str, str]:
        """Get mapping of layer names to types.

        Returns:
            Dictionary mapping layer names to their types
        """
        return {node.name: node.type for node in self.graph}
