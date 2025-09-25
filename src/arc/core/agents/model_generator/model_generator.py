"""Arc model specification generation agent."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from arc.core.agents.shared.base_agent import AgentError, BaseAgent
from arc.core.agents.shared.example_repository import ExampleRepository
from arc.database.services import ServiceContainer
from arc.graph.model import CORE_LAYERS, ModelSpec, validate_model_dict


class ModelGeneratorError(AgentError):
    """Raised when model generation fails."""


class ModelGeneratorAgent(BaseAgent):
    """Specialized agent for generating Arc model specifications using LLM."""

    def __init__(
        self,
        services: ServiceContainer,
        api_key: str,
        base_url: str | None = None,
        model: str | None = None,
    ):
        """Initialize model generator agent.

        Args:
            services: Service container for database access
            api_key: API key for LLM interactions
            base_url: Optional base URL
            model: Optional model name
        """
        super().__init__(services, api_key, base_url, model)
        self.example_repository = ExampleRepository()

    def get_template_directory(self) -> Path:
        """Get the template directory for model generation.

        Returns:
            Path to the model generator template directory
        """
        return Path(__file__).parent / "templates"

    async def generate_model(
        self,
        name: str,
        user_context: str,
        table_name: str,
        exclude_columns: list[str] | None = None,
    ) -> tuple[ModelSpec, str]:
        """Generate Arc model specification based on data and user context.

        Args:
            name: Model name for the specification
            user_context: User description of desired model
            table_name: Database table name for data exploration
            exclude_columns: Optional list of column names to exclude from model inputs

        Returns:
            Tuple of (parsed ModelSpec object, YAML string)

        Raises:
            ModelGeneratorError: If generation fails
        """
        # Build simple context for LLM
        data_profile = await self._profile_data(table_name, exclude_columns)
        context = {
            "model_name": name,
            "user_intent": user_context,
            "data_profile": data_profile,
            "available_components": self._get_model_components(),
            "examples": self._get_model_examples(user_context, data_profile),
        }

        # Use the base agent validation loop with default max_iterations
        try:
            model_spec, model_yaml = await self._generate_with_validation_loop(
                context, self._validate_model_comprehensive, 3
            )

            return model_spec, model_yaml

        except AgentError as e:
            raise ModelGeneratorError(str(e)) from e

    def _validate_model_comprehensive(
        self, model_yaml: str, context: dict[str, Any]
    ) -> dict[str, Any]:
        """Comprehensive validation of generated model with detailed error reporting.

        Args:
            model_yaml: Generated YAML model string
            context: Generation context for validation

        Returns:
            Dictionary with validation results:
            {"valid": bool, "object": ModelSpec, "error": str}
        """
        try:
            # Parse YAML
            model_dict = self._validate_yaml_syntax(model_yaml)

            # Check required top-level fields for model
            required_fields = ["inputs", "graph", "outputs"]
            missing_fields = [
                field for field in required_fields if field not in model_dict
            ]
            if missing_fields:
                return {
                    "valid": False,
                    "error": f"Missing required model fields: {missing_fields}",
                }

            # Validate model structure using dedicated validator
            validate_model_dict(model_dict)

            # Validate node types against available components
            node_errors = self._validate_node_types(model_dict, context)
            if node_errors:
                return {
                    "valid": False,
                    "error": f"Node validation errors: {node_errors}",
                }

            # Validate column references against actual data if available
            if "data_profile" in context and context["data_profile"]:
                column_errors = self._validate_model_column_references(
                    model_dict, context
                )
                if column_errors:
                    return {
                        "valid": False,
                        "error": f"Column validation errors: {column_errors}",
                    }

            # Parse into ModelSpec object
            try:
                model_spec = ModelSpec.from_yaml(model_yaml)
            except Exception as e:
                return {
                    "valid": False,
                    "error": f"Failed to parse into ModelSpec: {str(e)}",
                }

            return {"valid": True, "object": model_spec, "error": None}

        except Exception as e:
            return {
                "valid": False,
                "error": f"Validation exception: {str(e)}",
            }

    def _validate_node_types(
        self, model_dict: dict, context: dict[str, Any]
    ) -> list[str]:
        """Validate node types against available components."""
        errors = []

        graph = model_dict.get("graph", [])
        available_nodes = context.get("available_components", {}).get("node_types", [])

        for node in graph:
            node_type = node.get("type", "")

            # Validate against new pytorch prefix architecture
            if node_type and node_type not in available_nodes:
                errors.append(f"Unknown node type: {node_type}")

        return errors

    def _validate_model_column_references(
        self, model_dict: dict, context: dict[str, Any]
    ) -> list[str]:
        """Validate column references against actual data."""
        errors = []

        data_profile = context.get("data_profile", {})
        available_columns = []

        # Extract available columns from data profile
        if "columns" in data_profile:
            available_columns = [col["name"] for col in data_profile["columns"]]

        # Check input column references
        inputs = model_dict.get("inputs", {})
        for input_name, input_spec in inputs.items():
            columns = input_spec.get("columns", [])

            # Check for incomplete fields
            shape = input_spec.get("shape")
            if shape is None:
                errors.append(
                    f"Input '{input_name}' has incomplete 'shape' field - "
                    f"must specify [null, N] where N is column count"
                )

            if columns is None:
                errors.append(
                    f"Input '{input_name}' has incomplete 'columns' field - "
                    f"must specify actual column names"
                )
                continue

            if isinstance(columns, list) and len(columns) == 0:
                errors.append(
                    f"Input '{input_name}' has empty 'columns' field - "
                    f"must specify actual column names"
                )
                continue

            for col in columns:
                if col not in available_columns:
                    errors.append(
                        f"Input '{input_name}' references unknown column: {col}"
                    )

        return errors

    def _get_model_components(self) -> dict[str, Any]:
        """Get available model components from the new architecture."""
        try:
            return {"node_types": list(CORE_LAYERS.keys())}
        except Exception as e:
            raise RuntimeError(f"Failed to load model components: {e}") from e

    def _get_model_examples(
        self, user_context: str, data_profile: dict[str, Any]
    ) -> list[dict[str, Any]]:
        """Get relevant model examples."""
        examples = self.example_repository.retrieve_relevant_model_examples(
            user_context, data_profile, max_examples=1
        )
        return [{"schema": ex.schema, "name": ex.name} for ex in examples]

    async def _profile_data(
        self, table_name: str, exclude_columns: list[str] | None = None
    ) -> dict[str, Any]:
        """Get basic data profile for the table with optional column exclusion."""
        try:
            # Use ML data service instead of raw SQL/db manager
            dataset_info = self.services.ml_data.get_dataset_info(table_name)

            if dataset_info is None:
                return {"error": f"Table {table_name} not found or invalid"}

            # Filter out excluded columns if specified
            available_columns = dataset_info.columns
            if exclude_columns:
                excluded_set = set(exclude_columns)
                filtered_columns = [
                    col for col in available_columns if col["name"] not in excluded_set
                ]
            else:
                filtered_columns = available_columns

            # Return essential fields with filtered columns
            return {
                "table_name": dataset_info.name,
                "columns": filtered_columns,
            }

        except Exception as e:
            return {"error": f"Failed to analyze table {table_name}: {str(e)}"}
