"""Arc ML model agent."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import Any

from arc.core.agents.shared.base_agent import AgentError, BaseAgent
from arc.core.agents.shared.example_repository import ExampleRepository
from arc.database.services import ServiceContainer
from arc.graph.model import CORE_LAYERS, TORCH_FUNCTIONS, ModelSpec, validate_model_dict


class MLModelError(AgentError):
    """Raised when model generation fails."""


class MLModelAgent(BaseAgent):
    """Specialized agent for generating Arc model specifications using LLM."""

    def __init__(
        self,
        services: ServiceContainer,
        api_key: str,
        base_url: str | None = None,
        model: str | None = None,
        progress_callback: Callable[[str], None] | None = None,
    ):
        """Initialize model generator agent.

        Args:
            services: Service container for database access
            api_key: API key for LLM interactions
            base_url: Optional base URL
            model: Optional model name
            progress_callback: Optional callback to report progress/tool usage
        """
        super().__init__(services, api_key, base_url, model)
        self.example_repository = ExampleRepository()
        self.progress_callback = progress_callback
        # Note: knowledge_loader is now initialized in BaseAgent

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
        target_column: str | None = None,
        existing_yaml: str | None = None,
        editing_instructions: str | None = None,
        model_plan: str | None = None,
        preloaded_knowledge: list[dict[str, str]] | None = None,
        conversation_history: list[dict[str, str]] | None = None,
    ) -> tuple[ModelSpec, str, list[dict[str, str]]]:
        """Generate unified model + training specification.

        Args:
            name: Model name for the specification
            user_context: User description of desired model and training
            table_name: Database table name for data exploration
            target_column: Optional target column name for task-aware generation
            existing_yaml: Optional existing YAML to edit
            editing_instructions: Optional instructions for editing existing YAML
            model_plan: Optional ML plan guidance (architecture + training unified)
            preloaded_knowledge: Optional list of preloaded knowledge docs
            conversation_history: Optional conversation history for editing workflow

        Returns:
            Tuple of (parsed ModelSpec object, unified YAML string, conversation_history)
            Note: The YAML includes both model and training sections, but only ModelSpec
            is returned for backward compatibility. Training config extracted separately by tool.

        Raises:
            MLModelError: If generation fails
        """
        # Route to appropriate generation path
        if conversation_history is None:
            # Fresh generation - build full context
            return await self._generate_fresh(
                name=name,
                user_context=user_context,
                table_name=table_name,
                target_column=target_column,
                existing_yaml=existing_yaml,
                editing_instructions=editing_instructions,
                model_plan=model_plan,
                preloaded_knowledge=preloaded_knowledge,
            )
        else:
            # Continue conversation - just append feedback
            return await self._continue_conversation(
                feedback=editing_instructions or "",
                conversation_history=conversation_history,
            )

    async def _generate_fresh(
        self,
        name: str,
        user_context: str,
        table_name: str,
        target_column: str | None = None,
        existing_yaml: str | None = None,
        editing_instructions: str | None = None,
        model_plan: str | None = None,
        preloaded_knowledge: list[dict[str, str]] | None = None,
    ) -> tuple[ModelSpec, str, list[dict[str, str]]]:
        """Fresh generation with full context building.

        This path is used for initial generation or when starting a new conversation.
        It builds the complete system message with data profiling and knowledge loading.
        """
        # Build unified data profile with target-aware analysis
        data_profile = await self._get_unified_data_profile(table_name, target_column)

        # Use preloaded knowledge if provided
        if preloaded_knowledge:
            # Knowledge already loaded by tool
            loaded_knowledge_ids = [doc["id"] for doc in preloaded_knowledge]
            for doc in preloaded_knowledge:
                self._loaded_knowledge.add((doc["id"], "model"))
        else:
            preloaded_knowledge = []
            loaded_knowledge_ids = []

        # Build system message with all context
        system_message = self._render_template(
            self.get_template_name(),
            {
                "model_name": name,
                "user_intent": user_context,
                "data_profile": data_profile,
                "available_components": self._get_model_components(),
                "model_plan": model_plan,
                "preloaded_knowledge": preloaded_knowledge,
                "existing_yaml": existing_yaml,
                "editing_instructions": editing_instructions,
                "is_editing": existing_yaml is not None,
            },
        )

        # User message guides tool usage and lists pre-loaded knowledge
        if existing_yaml:
            user_message = (
                f"Edit the existing Arc-Graph specification with these changes: "
                f"{editing_instructions}."
            )
        else:
            user_message = f"Generate the Arc-Graph model specification for '{name}'."

        # Tell agent which knowledge IDs are already provided
        if loaded_knowledge_ids:
            user_message += (
                f"\n\nPre-loaded knowledge (already in system message): "
                f"{', '.join(loaded_knowledge_ids)}. "
                f"Do NOT reload these. Only use knowledge tools for "
                f"additional guidance if needed."
            )
        else:
            user_message += (
                "\n\nNo knowledge was pre-loaded. Use list_available_knowledge "
                "and read_knowledge_content if you need architecture guidance."
            )

        # Get ML tools from BaseAgent
        tools = self._get_ml_tools()

        # Generate with multi-turn tool support
        try:
            (
                model_spec,
                model_yaml,
                conversation_history,
            ) = await self._generate_with_tools(
                system_message=system_message,
                user_message=user_message,
                tools=tools,
                tool_executor=self._execute_ml_tool,
                validator_func=self._validate_model_comprehensive,
                validation_context={
                    "data_profile": data_profile,
                    "available_components": self._get_model_components(),
                },
                max_iterations=3,
                conversation_history=None,  # Fresh start
                progress_callback=self.progress_callback,
            )

            return model_spec, model_yaml, conversation_history

        except AgentError as e:
            raise MLModelError(str(e)) from e

    async def _continue_conversation(
        self,
        feedback: str,
        conversation_history: list[dict[str, str]],
    ) -> tuple[ModelSpec, str, list[dict[str, str]]]:
        """Continue existing conversation with user feedback.

        This path is used during interactive editing when conversation history exists.
        It simply appends the user's feedback to the existing conversation without
        rebuilding the system message.
        """
        # Get ML tools from BaseAgent
        tools = self._get_ml_tools()

        # Continue conversation with feedback
        try:
            model_spec, model_yaml, updated_history = await self._generate_with_tools(
                system_message="",  # Not used - already in conversation_history
                user_message=feedback,
                tools=tools,
                tool_executor=self._execute_ml_tool,
                validator_func=self._validate_model_comprehensive,
                validation_context={
                    "data_profile": None,  # Already in conversation history
                    "available_components": self._get_model_components(),
                },
                max_iterations=3,
                conversation_history=conversation_history,
                progress_callback=self.progress_callback,
            )

            return model_spec, model_yaml, updated_history

        except AgentError as e:
            raise MLModelError(str(e)) from e

    def _validate_model_comprehensive(
        self, model_yaml: str, context: dict[str, Any]
    ) -> dict[str, Any]:
        """Comprehensive validation of generated model with detailed error reporting.

        Args:
            model_yaml: Generated YAML model string (unified spec with training section)
            context: Generation context for validation

        Returns:
            Dictionary with validation results:
            {"valid": bool, "object": ModelSpec, "error": str}
        """
        try:
            # Parse YAML
            model_dict = self._validate_yaml_syntax(model_yaml)

            # Check required top-level fields for unified specification
            required_fields = ["inputs", "graph", "outputs", "training"]
            missing_fields = [
                field for field in required_fields if field not in model_dict
            ]
            if missing_fields:
                return {
                    "valid": False,
                    "error": f"Missing required fields: {missing_fields}",
                }

            # Check that training section contains loss
            training = model_dict.get("training", {})
            if not training.get("loss"):
                return {
                    "valid": False,
                    "error": "Missing required 'loss' field inside 'training' section",
                }

            # Validate model structure (without training section) using dedicated validator
            # Create a copy with just the model fields for validation
            model_only = {
                k: v for k, v in model_dict.items() if k != "training"
            }
            validate_model_dict(model_only)

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

            # Parse into ModelSpec object (from model-only portion)
            try:
                import yaml
                model_yaml_str = yaml.dump(model_only, default_flow_style=False, sort_keys=False)
                model_spec = ModelSpec.from_yaml(model_yaml_str)
            except Exception as e:
                return {
                    "valid": False,
                    "error": f"Failed to parse into ModelSpec: {str(e)}",
                }

            return {"valid": True, "object": model_spec, "error": None}

        except AgentError as e:
            # AgentError messages are already well-formatted, don't wrap them
            return {
                "valid": False,
                "error": str(e),
            }
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

        # Get list of defined modules
        modules = model_dict.get("modules", {})
        module_names = list(modules.keys()) if modules else []

        for node in graph:
            node_type = node.get("type", "")

            if not node_type:
                continue

            # Allow special Arc components
            if node_type.startswith("arc."):
                continue

            # Allow references to custom modules
            if node_type.startswith("module."):
                module_name = node_type.split(".", 1)[1]
                if module_name not in module_names:
                    errors.append(
                        f"Unknown module reference: {node_type} "
                        f"(module '{module_name}' not defined in 'modules' section)"
                    )
                continue

            # Validate against available PyTorch components
            if node_type not in available_nodes:
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

    def _get_available_knowledge_metadata(self) -> str:
        """Get formatted knowledge metadata for LLM context.

        Returns:
            Formatted string with available knowledge information
        """
        return self.knowledge_loader.format_metadata_for_llm()

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
