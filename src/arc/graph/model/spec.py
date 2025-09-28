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
    type: str  # torch.nn.*, torch.nn.functional.*, torch.*, module.*, arc.stack
    params: dict[str, Any] | None = None
    inputs: dict[str, str] | list[str] | None = (
        None  # Support both dict and list formats
    )


@dataclass
class ModuleDefinition:
    """Specification for a reusable module/sub-graph."""

    inputs: list[str]  # Parameter names for the module
    graph: list[GraphNode]  # Internal computation graph
    outputs: dict[str, str]  # Named outputs from internal nodes


@dataclass
class LossSpec:
    """Specification for model loss function."""

    type: str
    inputs: dict[str, str] | None = None
    params: dict[str, Any] | None = None


@dataclass
class ModelSpec:
    """Complete model specification."""

    inputs: dict[str, ModelInput]
    graph: list[GraphNode]
    outputs: dict[str, str]
    modules: dict[str, ModuleDefinition] | None = None  # Optional reusable modules
    loss: LossSpec | None = None  # Optional loss function specification

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
        from arc.graph.model.validator import validate_model_dict

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

        # Parse modules section if present
        modules = None
        if "modules" in data and data["modules"] is not None:
            modules = {}
            for module_name, module_data in data["modules"].items():
                # Parse module graph nodes
                module_graph = []
                for node_data in module_data["graph"]:
                    module_graph.append(
                        GraphNode(
                            name=node_data["name"],
                            type=node_data["type"],
                            params=node_data.get("params"),
                            inputs=node_data.get("inputs"),
                        )
                    )

                modules[module_name] = ModuleDefinition(
                    inputs=module_data["inputs"],
                    graph=module_graph,
                    outputs=module_data["outputs"],
                )

        # Parse main graph nodes
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

        # Parse loss section if present
        loss = None
        if "loss" in data and data["loss"] is not None:
            loss_data = data["loss"]
            loss = LossSpec(
                type=loss_data["type"],
                inputs=loss_data.get("inputs"),
                params=loss_data.get("params"),
            )

        return cls(
            inputs=inputs, graph=graph, outputs=outputs, modules=modules, loss=loss
        )

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
        # Convert to dict, filtering out None modules
        spec_dict = asdict(self)
        if spec_dict.get("modules") is None:
            del spec_dict["modules"]
        return yaml.dump(spec_dict, default_flow_style=False)

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
