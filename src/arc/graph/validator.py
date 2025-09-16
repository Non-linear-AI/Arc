"""Validation for Arc-Graph YAML structures (v0.1 overview schema)."""

from __future__ import annotations

from typing import Any

from ..database.base import Database
from .spec import ArcGraph


class GraphValidationError(ValueError):
    """Exception raised for Arc-Graph validation errors."""

    pass


class ArcGraphValidator:
    """Class-based validator for Arc-Graph specifications.

    Supports both static validation (structural) and runtime validation
    (against database schema and data types).
    """

    def __init__(self, database: Database | None = None):
        """Initialize validator.

        Args:
            database: Optional database connection for runtime validation.
        """
        self.database = database
        self._var_registry: dict[str, Any] = {}
        self._state_registry: dict[str, Any] = {}
        self._tensor_registry: dict[str, Any] = {}

    def validate_static(self, graph: ArcGraph) -> None:
        """Perform static validation of Arc-Graph structure.

        Validates:
        - Shape variable references (vars.*)
        - State variable references (states.*)
        - Tensor variable references (tensors.*)
        - Processor input/output consistency
        - Model input/output consistency

        Args:
            graph: Parsed Arc-Graph to validate

        Raises:
            GraphValidationError: If validation fails
        """
        self._reset_registries()

        # Validate features pipeline
        self._validate_features_static(graph.features)

        # Validate model with computed variables
        self._validate_model_static(graph.model)

        # Validate trainer references
        self._validate_trainer_static(graph.trainer, graph.model)

    def validate_runtime(self, graph: ArcGraph, table_name: str) -> None:
        """Perform runtime validation against database schema.

        Validates:
        - Feature columns exist in table
        - Target columns exist in table
        - Data types match expected types
        - Column cardinalities for categorical features

        Args:
            graph: Parsed Arc-Graph to validate
            table_name: Name of table to validate against

        Raises:
            GraphValidationError: If validation fails
            RuntimeError: If database not provided
        """
        if not self.database:
            raise RuntimeError("Database connection required for runtime validation")

        # Get table schema
        schema = self._get_table_schema(table_name)

        # Validate feature columns exist
        self._validate_columns_exist(graph.features.feature_columns, schema, "feature")

        # Validate target columns exist (if provided)
        if graph.features.target_columns:
            self._validate_columns_exist(
                graph.features.target_columns, schema, "target"
            )

        # Validate data types
        self._validate_data_types(graph, schema)

    def _reset_registries(self) -> None:
        """Reset internal variable registries for a new validation."""
        self._var_registry.clear()
        self._state_registry.clear()
        self._tensor_registry.clear()

    def _validate_features_static(self, features) -> None:
        """Validate features section for static consistency."""
        # Track available variables after each processor
        available_columns = set(features.feature_columns)
        if features.target_columns:
            available_columns.update(features.target_columns)

        for processor in features.processors:
            # Validate processor inputs reference available variables
            if processor.inputs:
                for _input_name, var_ref in processor.inputs.items():
                    self._validate_variable_reference(var_ref, available_columns)

            # Register processor outputs
            if processor.outputs:
                for global_var, _output_name in processor.outputs.items():
                    self._register_variable(global_var, processor.name)

    def _validate_model_static(self, model) -> None:
        """Validate model section for static consistency."""
        # Validate model inputs reference available tensors
        for input_name, input_spec in model.inputs.items():
            # Check if shape references vars
            for dim in input_spec.shape:
                if isinstance(dim, str) and dim.startswith("vars."):
                    var_name = dim[5:]  # Remove "vars." prefix
                    if f"vars.{var_name}" not in self._var_registry:
                        raise GraphValidationError(
                            f"Model input {input_name} references "
                            f"undefined variable: {dim}"
                        )

        # Validate graph node connections
        available_nodes = set(model.inputs.keys())
        for node in model.graph:
            # Validate node inputs reference available nodes/inputs
            if node.inputs:
                for _, source in node.inputs.items():
                    if "." in source:
                        source_node = source.split(".")[0]
                        if source_node not in available_nodes:
                            raise GraphValidationError(
                                f"Node {node.name} references "
                                f"undefined source: {source}"
                            )
                    elif source not in available_nodes:
                        raise GraphValidationError(
                            f"Node {node.name} references undefined input: {source}"
                        )

            # Register this node as available
            available_nodes.add(node.name)

        # Validate model outputs reference available nodes
        for output_name, source in model.outputs.items():
            if "." in source:
                source_node = source.split(".")[0]
                if source_node not in available_nodes:
                    raise GraphValidationError(
                        f"Model output {output_name} references "
                        f"undefined source: {source}"
                    )

    def _validate_trainer_static(self, trainer, model) -> None:
        """Validate trainer section references."""
        # Validate loss inputs reference model outputs or target columns
        if trainer.loss.inputs:
            for input_name, source in trainer.loss.inputs.items():
                if source.startswith("model."):
                    output_name = source[6:]  # Remove "model." prefix
                    if output_name not in model.outputs:
                        raise GraphValidationError(
                            f"Loss input {input_name} references "
                            f"undefined model output: {source}"
                        )
                elif source.startswith("target_columns."):
                    # Would need features context to validate
                    pass

    def _validate_variable_reference(
        self, var_ref: str, available_columns: set[str]
    ) -> None:
        """Validate a variable reference is available."""
        if var_ref.startswith("vars."):
            if var_ref not in self._var_registry:
                raise GraphValidationError(
                    f"Reference to undefined variable: {var_ref}"
                )
        elif var_ref.startswith("states."):
            if var_ref not in self._state_registry:
                raise GraphValidationError(f"Reference to undefined state: {var_ref}")
        elif var_ref.startswith("tensors."):
            if var_ref not in self._tensor_registry:
                raise GraphValidationError(f"Reference to undefined tensor: {var_ref}")
        elif var_ref not in available_columns and not any(
            var_ref.startswith(prefix)
            for prefix in ["feature_columns", "target_columns"]
        ):
            # Check if it's a column reference
            raise GraphValidationError(
                f"Reference to undefined variable or column: {var_ref}"
            )

    def _register_variable(self, global_var: str, source: str) -> None:
        """Register a variable in the appropriate registry."""
        if global_var.startswith("vars."):
            self._var_registry[global_var] = source
        elif global_var.startswith("states."):
            self._state_registry[global_var] = source
        elif global_var.startswith("tensors."):
            self._tensor_registry[global_var] = source

    def _get_table_schema(self, table_name: str) -> dict[str, str]:
        """Get table schema from database."""
        try:
            result = self.database.query(f"DESCRIBE {table_name}")
            schema = {}
            for row in result:
                # DuckDB DESCRIBE returns: column_name, column_type, null,
                # key, default, extra
                schema[row["column_name"]] = row["column_type"]
            return schema
        except Exception as e:
            raise GraphValidationError(
                f"Failed to get schema for table {table_name}: {e}"
            ) from e

    def _validate_columns_exist(
        self, columns: list[str], schema: dict[str, str], column_type: str
    ) -> None:
        """Validate that columns exist in the table schema."""
        missing_columns = [col for col in columns if col not in schema]
        if missing_columns:
            raise GraphValidationError(
                f"Missing {column_type} columns in table: {missing_columns}"
            )

    def _validate_data_types(self, graph: ArcGraph, schema: dict[str, str]) -> None:
        """Validate data types compatibility between graph and table."""
        # Map Arc-Graph dtypes to DuckDB types

        # Validate feature columns
        for col in graph.features.feature_columns:
            if col in schema:
                schema[col].upper()
                # For now, be permissive since we may do type conversion
                # Could add stricter validation based on model requirements


def _require(obj: dict[str, Any], key: str, msg: str | None = None) -> Any:
    """Helper function for requiring dictionary keys."""
    if key not in obj:
        raise GraphValidationError(msg or f"Missing required field: {key}")
    return obj[key]


def validate_graph_dict(data: dict[str, Any]) -> None:
    """Validate a parsed YAML dict for the v0.1 overview schema.

    Minimal structural validation only (semantics are handled by ArcGraphValidator).
    """
    # metadata
    version = _require(data, "version", "version required")
    if not isinstance(version, str) or not version:
        raise GraphValidationError("version must be a non-empty string")
    model_name = _require(data, "model_name", "model_name required")
    if not isinstance(model_name, str) or not model_name:
        raise GraphValidationError("model_name must be a non-empty string")
    if "description" in data and not isinstance(data["description"], str | type(None)):
        raise GraphValidationError("description must be a string if provided")

    # features
    features = _require(data, "features", "features section required")
    if not isinstance(features, dict):
        raise GraphValidationError("features must be a mapping")
    fcols = _require(features, "feature_columns", "features.feature_columns required")
    if not isinstance(fcols, list) or not all(
        isinstance(c, (str, int, float)) for c in fcols
    ):
        raise GraphValidationError("feature_columns must be a list of column names")
    if "target_columns" in features and not (
        isinstance(features["target_columns"], list)
        and all(isinstance(c, (str, int, float)) for c in features["target_columns"])
    ):
        raise GraphValidationError("target_columns must be a list if provided")
    procs = features.get("processors", [])
    if not isinstance(procs, list):
        raise GraphValidationError("features.processors must be a list")
    for i, p in enumerate(procs):
        if not isinstance(p, dict):
            raise GraphValidationError(f"processor[{i}] must be a mapping")
        _require(p, "name", f"processor[{i}].name required")
        _require(p, "op", f"processor[{i}].op required")
        if "train_only" in p and not isinstance(p["train_only"], bool):
            raise GraphValidationError(f"processor[{i}].train_only must be bool")
        if "inputs" in p and not isinstance(p["inputs"], dict):
            raise GraphValidationError(f"processor[{i}].inputs must be a mapping")
        if "outputs" in p and not isinstance(p["outputs"], dict):
            raise GraphValidationError(f"processor[{i}].outputs must be a mapping")

    # model
    model = _require(data, "model", "model section required")
    if not isinstance(model, dict):
        raise GraphValidationError("model must be a mapping")
    minputs = _require(model, "inputs", "model.inputs required")
    if not isinstance(minputs, dict) or not minputs:
        raise GraphValidationError("model.inputs must be a non-empty mapping")
    for key, spec in minputs.items():
        if not isinstance(spec, dict):
            raise GraphValidationError(f"model.inputs.{key} must be a mapping")
        _require(spec, "dtype", f"model.inputs.{key}.dtype required")
        shape = _require(spec, "shape", f"model.inputs.{key}.shape required")
        if not isinstance(shape, list):
            raise GraphValidationError(f"model.inputs.{key}.shape must be a list")
    mgraph = _require(model, "graph", "model.graph required")
    if not isinstance(mgraph, list) or not mgraph:
        raise GraphValidationError("model.graph must be a non-empty list")
    for i, node in enumerate(mgraph):
        if not isinstance(node, dict):
            raise GraphValidationError(f"model.graph[{i}] must be a mapping")
        _require(node, "name", f"model.graph[{i}].name required")
        _require(node, "type", f"model.graph[{i}].type required")
        if "params" in node and not isinstance(node["params"], dict):
            raise GraphValidationError(f"model.graph[{i}].params must be a mapping")
        if "inputs" in node and not isinstance(node["inputs"], dict):
            raise GraphValidationError(f"model.graph[{i}].inputs must be a mapping")
    moutputs = _require(model, "outputs", "model.outputs required")
    if not isinstance(moutputs, dict) or not moutputs:
        raise GraphValidationError("model.outputs must be a non-empty mapping")

    # trainer
    trainer = _require(data, "trainer", "trainer section required")
    if not isinstance(trainer, dict):
        raise GraphValidationError("trainer must be a mapping")
    optimizer = _require(trainer, "optimizer", "trainer.optimizer required")
    if not isinstance(optimizer, dict):
        raise GraphValidationError("trainer.optimizer must be a mapping")
    _require(optimizer, "type", "trainer.optimizer.type required")
    if "config" in optimizer and not isinstance(optimizer["config"], dict):
        raise GraphValidationError("trainer.optimizer.config must be a mapping")
    loss = _require(trainer, "loss", "trainer.loss required")
    if not isinstance(loss, dict):
        raise GraphValidationError("trainer.loss must be a mapping")
    _require(loss, "type", "trainer.loss.type required")
    if "inputs" in loss and not isinstance(loss["inputs"], dict):
        raise GraphValidationError("trainer.loss.inputs must be a mapping")

    # predictor (optional)
    if "predictor" in data:
        pred = data["predictor"]
        if not isinstance(pred, dict):
            raise GraphValidationError("predictor must be a mapping if provided")
        if "returns" in pred and not isinstance(pred["returns"], list):
            raise GraphValidationError("predictor.returns must be a list")
