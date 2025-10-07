"""Arc model specification generation agent."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from arc.core.agents.shared.base_agent import AgentError, BaseAgent
from arc.core.agents.shared.example_repository import ExampleRepository
from arc.database.services import ServiceContainer
from arc.graph.model import CORE_LAYERS, TORCH_FUNCTIONS, ModelSpec, validate_model_dict


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

    def _get_template_path(self, _category: str) -> Path:
        """Get the template path for a specific category.

        Args:
            category: Model category ("mlp", "transformer", etc.)

        Returns:
            Path to the base template (now architecture-agnostic)
        """
        base_dir = self.get_template_directory()
        return base_dir / "base.j2"

    def get_template_name(self) -> str:
        """Get the name of the template file based on current category.

        Returns:
            Template filename relative to the template directory
        """
        # Always use the base template now (architecture-agnostic)
        return "base.j2"

    async def generate_model(
        self,
        name: str,
        user_context: str,
        table_name: str,
        target_column: str | None = None,
        category: str | None = None,
        existing_yaml: str | None = None,
        editing_instructions: str | None = None,
        ml_plan_architecture: str | None = None,
    ) -> tuple[ModelSpec, str]:
        """Generate Arc model specification based on data and user context.

        Args:
            name: Model name for the specification
            user_context: User description of desired model
            table_name: Database table name for data exploration
            target_column: Optional target column name for task-aware generation
            category: Optional category hint ("mlp", "transformer", etc.)
            existing_yaml: Optional existing YAML to edit
            editing_instructions: Optional instructions for editing existing YAML
            ml_plan_architecture: Optional ML plan architecture guidance

        Returns:
            Tuple of (parsed ModelSpec object, YAML string)

        Raises:
            ModelGeneratorError: If generation fails
        """
        # Build unified data profile with target-aware analysis
        data_profile = await self._get_unified_data_profile(table_name, target_column)

        # Determine category (explicit or auto-detected)
        resolved_category = category or self._detect_category_from_context(
            user_context, data_profile
        )

        # Architecture display names
        display_names = {
            "mlp": "Multi-Layer Perceptron (Feedforward Neural Network)",
            "transformer": "Transformer (Attention-based Neural Network)",
            "dcn": "Deep & Cross Network",
            "mmoe": "Multi-gate Mixture of Experts",
        }

        context = {
            "model_name": name,
            "user_intent": user_context,
            "data_profile": data_profile,
            "architecture_type": resolved_category,
            "architecture_display_name": display_names.get(
                resolved_category, resolved_category.upper()
            ),
            "available_components": self._get_model_components(),
            "architecture_guides": self._load_architecture_guides([resolved_category]),
            "existing_yaml": existing_yaml,
            "editing_instructions": editing_instructions,
            "is_editing": existing_yaml is not None,
            "ml_plan_architecture": ml_plan_architecture,
        }

        # Store category for template selection
        self._current_category = resolved_category

        # Use the base agent validation loop with default max_iterations
        try:
            model_spec, model_yaml = await self._generate_with_validation_loop(
                context, self._validate_model_comprehensive, 3
            )

            return model_spec, model_yaml

        except AgentError as e:
            raise ModelGeneratorError(str(e)) from e

    def _detect_category_from_context(
        self, user_context: str, _data_profile: dict[str, Any]
    ) -> str:
        """Detect model category from context and data characteristics.

        Args:
            user_context: User description of desired model
            _data_profile: Data profile information (reserved for future use)

        Returns:
            Detected category: "mlp", "dcn", "mmoe", or "transformer"
        """
        context_lower = user_context.lower()

        # Multi-gate Mixture of Experts indicators (check first for priority)
        if any(
            keyword in context_lower
            for keyword in [
                "multi-task",
                "multiple task",
                "multitask",
                "shared representation",
            ]
        ):
            return "mmoe"

        # Deep & Cross Network indicators
        if any(
            keyword in context_lower
            for keyword in [
                "feature cross",
                "interaction",
                "ctr",
                "click-through",
                "recommendation",
            ]
        ):
            return "dcn"

        # Transformer indicators
        if any(
            keyword in context_lower
            for keyword in [
                "attention",
                "sequence",
                "transformer",
                "self-attention",
                "encoder",
            ]
        ):
            return "transformer"

        # Default to MLP for tabular data
        mlp_keywords = [
            "classify",
            "classification",
            "predict",
            "prediction",
            "regression",
            "binary",
            "multiclass",
            "categorical",
            "numerical",
            "features",
            "target",
            "label",
            "supervised",
            "risk",
            "score",
            "fraud",
        ]

        if any(keyword in context_lower for keyword in mlp_keywords):
            return "mlp"

        # Default to MLP for general cases
        return "mlp"

    def _validate_category(self, category: str) -> str:
        """Validate and normalize category input.

        Args:
            category: Category string to validate

        Returns:
            Validated category ("mlp", "dcn", "mmoe", "transformer")
        """
        valid_categories = {"mlp", "dcn", "mmoe", "transformer"}
        if category in valid_categories:
            return category

        return "mlp"  # Default to MLP

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
            required_fields = ["inputs", "graph", "outputs", "loss"]
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

        # Extract available columns from unified data profile structure
        if "feature_columns" in data_profile:
            available_columns = [col["name"] for col in data_profile["feature_columns"]]

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
            # Combine both CORE_LAYERS (nn.Module) and TORCH_FUNCTIONS (functional)
            all_components = list(CORE_LAYERS.keys()) + list(TORCH_FUNCTIONS.keys())
            return {
                "node_types": all_components,
                "description": (
                    "PyTorch components available in Arc-Graph include layers "
                    "(instantiated once, used in forward pass) and functions "
                    "(applied as operations). All standard PyTorch neural "
                    "network components are supported."
                ),
            }
        except Exception as e:
            raise RuntimeError(f"Failed to load model components: {e}") from e

    def _load_architecture_guides(
        self, architecture_types: list[str]
    ) -> dict[str, str]:
        """Load architecture-specific content from files."""
        architecture_guides = {}

        for arch_type in architecture_types:
            content_path = (
                self.get_template_directory() / "architectures" / f"{arch_type}.md"
            )
            try:
                with open(content_path) as f:
                    architecture_guides[arch_type.upper()] = f.read()
            except FileNotFoundError:
                architecture_guides[arch_type.upper()] = (
                    f"*Content not found for {arch_type}*"
                )

        return architecture_guides

    def _get_model_examples(
        self, user_context: str, data_profile: dict[str, Any], _category: str
    ) -> list[dict[str, Any]]:
        """Get relevant model examples based on category.

        Args:
            user_context: User description of desired model
            data_profile: Data profile information
            _category: Model category for targeted examples (reserved for future use)

        Returns:
            List of relevant model examples
        """
        # For now, use existing example retrieval but we can enhance this
        # to be category-aware in the future
        examples = self.example_repository.retrieve_relevant_model_examples(
            user_context, data_profile, max_examples=1
        )
        return [{"schema": ex.schema, "name": ex.name} for ex in examples]

    async def _get_unified_data_profile(
        self, table_name: str, target_column: str | None = None
    ) -> dict[str, Any]:
        """Get unified data profile with target-aware analysis for LLM context.

        Args:
            table_name: Database table name
            target_column: Optional target column for task-aware analysis

        Returns:
            Unified data profile with target and feature information
        """
        try:
            # Use ML data service for dataset information
            dataset_info = self.services.ml_data.get_dataset_info(table_name)

            if dataset_info is None:
                return {"error": f"Table {table_name} not found or invalid"}

            # Build exclude set (only target column)
            exclude_set = set()
            if target_column:
                exclude_set.add(target_column)

            # Separate target and feature columns
            all_columns = dataset_info.columns
            feature_columns = [
                col for col in all_columns if col["name"] not in exclude_set
            ]

            # Base profile structure
            profile = {
                "table_name": dataset_info.name,
                "feature_columns": feature_columns,
                "total_columns": len(all_columns),
                "feature_count": len(feature_columns),
            }

            # Add target analysis if target column specified
            if target_column:
                target_analysis = await self._analyze_target_column(
                    table_name, target_column
                )
                profile["target_analysis"] = target_analysis

            return profile

        except Exception as e:
            return {"error": f"Failed to analyze table {table_name}: {str(e)}"}

    async def _analyze_target_column(
        self, table_name: str, target_column: str
    ) -> dict[str, Any]:
        """Analyze target column to provide factual information for LLM decision making.

        Args:
            table_name: Database table name
            target_column: Target column to analyze

        Returns:
            Dictionary with target column facts for LLM context
        """
        try:
            # Get target column basic info
            dataset_info = self.services.ml_data.get_dataset_info(table_name)
            if not dataset_info:
                return {"error": f"Table {table_name} not found"}

            # Find target column in dataset
            target_col_info = None
            for col in dataset_info.columns:
                if col["name"] == target_column:
                    target_col_info = col
                    break

            if not target_col_info:
                return {"error": f"Target column '{target_column}' not found in table"}

            # Get statistical analysis using ML data service
            stats = self.services.ml_data.analyze_target_column(
                table_name, target_column
            )

            # Build factual analysis for LLM
            analysis = {
                "column_name": target_column,
                "data_type": target_col_info["type"],
                "unique_values": stats.get("unique_count", 0),
                "null_values": stats.get("null_count", 0),
                "total_rows": stats.get("total_count", 0),
            }

            # Add type-specific facts
            if stats.get("is_numeric", False):
                analysis.update(
                    {
                        "is_numeric": True,
                        "min_value": stats.get("min_value"),
                        "max_value": stats.get("max_value"),
                        "mean_value": stats.get("mean_value"),
                    }
                )
            else:
                analysis.update(
                    {
                        "is_numeric": False,
                        "sample_values": stats.get("sample_values", []),
                    }
                )

            return analysis

        except Exception as e:
            return {
                "error": f"Failed to analyze target column {target_column}: {str(e)}"
            }
