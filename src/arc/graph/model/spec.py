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
    categorical: bool = False  # Whether this is a categorical feature
    embedding_dim: int | None = None  # Embedding dim for categorical
    vocab_size: int | None = None  # Vocabulary size for categorical features


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
    name: str | None = None  # Model name (injected by tool)
    data_table: str | None = None  # Training data table (injected by tool)
    plan_id: str | None = None  # Optional ML plan ID for lineage tracking
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
                categorical=input_spec.get("categorical", False),
                embedding_dim=input_spec.get("embedding_dim"),
                vocab_size=input_spec.get("vocab_size"),
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

        # Parse metadata fields (optional, injected by tool)
        name = data.get("name")
        data_table = data.get("data_table")
        plan_id = data.get("plan_id")

        return cls(
            inputs=inputs,
            graph=graph,
            outputs=outputs,
            name=name,
            data_table=data_table,
            plan_id=plan_id,
            modules=modules,
            loss=loss,
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
        # Build dict with specific field order
        spec_dict = {}

        # Metadata fields first (if present)
        if self.name is not None:
            spec_dict["name"] = self.name
        if self.data_table is not None:
            spec_dict["data_table"] = self.data_table
        if self.plan_id is not None:
            spec_dict["plan_id"] = self.plan_id

        # Core spec fields
        spec_dict["inputs"] = asdict(self)["inputs"]
        spec_dict["graph"] = asdict(self)["graph"]
        spec_dict["outputs"] = self.outputs

        # Optional fields
        if self.modules is not None:
            spec_dict["modules"] = asdict(self)["modules"]
        if self.loss is not None:
            spec_dict["loss"] = asdict(self)["loss"]

        return yaml.dump(spec_dict, default_flow_style=False, sort_keys=False)

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
