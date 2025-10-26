"""Arc ML train agent."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import Any

import yaml

from arc.core.agents.shared.base_agent import AgentError, BaseAgent
from arc.core.agents.shared.example_repository import ExampleRepository
from arc.database.services import ServiceContainer
from arc.graph.trainer import (
    CORE_LOSSES,
    CORE_OPTIMIZERS,
    TrainerSpec,
    validate_trainer_dict,
)


class MLTrainError(AgentError):
    """Raised when trainer generation fails."""


class MLTrainAgent(BaseAgent):
    """Specialized agent for generating Arc trainer specifications using LLM."""

    def __init__(
        self,
        services: ServiceContainer,
        api_key: str,
        base_url: str | None = None,
        model: str | None = None,
        progress_callback: Callable[[str], None] | None = None,
    ):
        """Initialize trainer generator agent.

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
        """Get the template directory for trainer generation.

        Returns:
            Path to the trainer generator template directory
        """
        return Path(__file__).parent / "templates"

    async def generate_trainer(
        self,
        name: str,
        instruction: str,
        model_id: str,
        model_spec_yaml: str,
        existing_yaml: str | None = None,
        ml_plan_training_config: str | None = None,
        recommended_knowledge_ids: list[str] | None = None,
        conversation_history: list[dict[str, str]] | None = None,
    ) -> tuple[TrainerSpec, str, list[dict[str, str]]]:
        """Generate Arc trainer specification based on model and instruction.

        Args:
            name: Trainer name for the specification
            instruction: User's instruction for training setup.
                For generation: requirements for training configuration.
                For editing: changes to make to existing YAML.
            model_id: ID of the registered model (e.g., "diabetes-logistic-v1")
            model_spec_yaml: Model specification YAML content
            existing_yaml: Optional existing YAML to edit (editing mode)
            ml_plan_training_config: Optional training config from ML plan
            recommended_knowledge_ids: Optional list of knowledge IDs
                recommended by ML Plan
            conversation_history: Optional conversation history for editing workflow

        Returns:
            Tuple of (parsed TrainerSpec, raw YAML string, conversation_history)

        Raises:
            TrainerGeneratorError: If generation fails
        """
        # Route to appropriate generation path
        if conversation_history is None:
            # Fresh generation - build full context
            return await self._generate_fresh(
                name=name,
                instruction=instruction,
                model_id=model_id,
                model_spec_yaml=model_spec_yaml,
                existing_yaml=existing_yaml,
                ml_plan_training_config=ml_plan_training_config,
                recommended_knowledge_ids=recommended_knowledge_ids,
            )
        else:
            # Continue conversation - just append feedback
            return await self._continue_conversation(
                feedback=instruction,
                conversation_history=conversation_history,
            )

    async def _generate_fresh(
        self,
        name: str,
        instruction: str,
        model_id: str,
        model_spec_yaml: str,
        existing_yaml: str | None = None,
        ml_plan_training_config: str | None = None,
        recommended_knowledge_ids: list[str] | None = None,
    ) -> tuple[TrainerSpec, str, list[dict[str, str]]]:
        """Fresh generation with full context building.

        This path is used for initial generation or when starting a new conversation.
        It builds the complete system message with knowledge loading.
        """
        # Pre-load recommended knowledge content (handle missing gracefully)
        recommended_knowledge = ""
        loaded_knowledge_ids = []
        if recommended_knowledge_ids:
            for knowledge_id in recommended_knowledge_ids:
                content = self.knowledge_loader.load_knowledge(knowledge_id, "train")
                if content:
                    # Successfully loaded - add to system context
                    recommended_knowledge += (
                        f"\n\n# Training Knowledge: {knowledge_id}\n\n{content}"
                    )
                    loaded_knowledge_ids.append(knowledge_id)
                # If missing, silently skip (already logged at debug level)

        # Build system message with all context
        system_message = self._render_template(
            "prompt.j2",
            {
                "trainer_name": name,
                "instruction": instruction,
                "model_id": model_id,
                "model_spec": model_spec_yaml,
                "model_profile": self._extract_model_profile(model_spec_yaml),
                "available_components": self._get_training_components(),
                "examples": self._get_trainer_examples(instruction),
                "existing_yaml": existing_yaml,
                "ml_plan_training_config": ml_plan_training_config,
                "recommended_knowledge": recommended_knowledge,
            },
        )

        # User message guides tool usage and lists pre-loaded knowledge
        if existing_yaml:
            user_message = (
                f"Edit the existing trainer specification with these "
                f"changes: {instruction}."
            )
        else:
            user_message = f"Generate the trainer specification for '{name}'."

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
                "and read_knowledge_content if you need training guidance."
            )

        # Get ML tools from BaseAgent
        tools = self._get_ml_tools()

        # Generate with multi-turn tool support
        try:
            (
                trainer_spec,
                trainer_yaml,
                conversation_history,
            ) = await self._generate_with_tools(
                system_message=system_message,
                user_message=user_message,
                tools=tools,
                tool_executor=self._execute_ml_tool,
                validator_func=self._validate_trainer_comprehensive,
                validation_context={
                    "available_components": self._get_training_components(),
                },
                max_iterations=3,
                conversation_history=None,  # Fresh generation - no history yet
                progress_callback=self.progress_callback,
            )

            return trainer_spec, trainer_yaml, conversation_history

        except AgentError as e:
            raise MLTrainError(str(e)) from e

    async def _continue_conversation(
        self,
        feedback: str,
        conversation_history: list[dict[str, str]],
    ) -> tuple[TrainerSpec, str, list[dict[str, str]]]:
        """Continue existing conversation with user feedback.

        This path is used during interactive editing when conversation history exists.
        It simply appends the user's feedback to the existing conversation without
        rebuilding the system message.
        """
        # Get ML tools from BaseAgent
        tools = self._get_ml_tools()

        # Continue conversation with feedback
        try:
            (
                trainer_spec,
                trainer_yaml,
                updated_history,
            ) = await self._generate_with_tools(
                system_message="",  # Not used - already in conversation_history
                user_message=feedback,
                tools=tools,
                tool_executor=self._execute_ml_tool,
                validator_func=self._validate_trainer_comprehensive,
                validation_context={
                    "available_components": self._get_training_components(),
                },
                max_iterations=3,
                conversation_history=conversation_history,
                progress_callback=self.progress_callback,
            )

            return trainer_spec, trainer_yaml, updated_history

        except AgentError as e:
            raise MLTrainError(str(e)) from e

    def _validate_trainer_comprehensive(
        self, trainer_yaml: str, context: dict[str, Any]
    ) -> dict[str, Any]:
        """Comprehensive validation of generated trainer with detailed error reporting.

        Args:
            trainer_yaml: Generated YAML trainer string
            context: Generation context for validation

        Returns:
            Dictionary with validation results:
            {"valid": bool, "object": TrainerSpec, "error": str}
        """
        try:
            # Parse YAML
            trainer_dict = self._validate_yaml_syntax(trainer_yaml)

            # Check required top-level fields for trainer
            required_fields = ["model_ref", "optimizer"]
            missing_fields = [
                field for field in required_fields if field not in trainer_dict
            ]
            if missing_fields:
                return {
                    "valid": False,
                    "error": f"Missing required trainer fields: {missing_fields}",
                }

            # Validate trainer structure using dedicated validator
            validate_trainer_dict(trainer_dict)

            # Validate optimizer and loss types against available components
            component_errors = self._validate_trainer_components(trainer_dict, context)
            if component_errors:
                return {
                    "valid": False,
                    "error": f"Component validation errors: {component_errors}",
                }

            # Parse into TrainerSpec object
            try:
                trainer_spec = TrainerSpec.from_yaml(trainer_yaml)
            except Exception as e:
                return {
                    "valid": False,
                    "error": f"Failed to parse into TrainerSpec: {str(e)}",
                }

            return {"valid": True, "object": trainer_spec, "error": None}

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

    def _validate_trainer_components(
        self, trainer_dict: dict, context: dict[str, Any]
    ) -> list[str]:
        """Validate optimizer and loss function components."""
        errors = []

        # Validate optimizer type against new pytorch prefix architecture
        optimizer_type = trainer_dict.get("optimizer", {}).get("type", "")
        available_optimizers = context.get("available_components", {}).get(
            "optimizers", []
        )

        if optimizer_type and optimizer_type not in available_optimizers:
            errors.append(f"Unknown optimizer type: {optimizer_type}")

        # Validate loss function type against new pytorch prefix architecture
        loss_type = trainer_dict.get("loss", {}).get("type", "")
        available_losses = context.get("available_components", {}).get(
            "loss_functions", []
        )

        if loss_type and loss_type not in available_losses:
            errors.append(f"Unknown loss function type: {loss_type}")

        return errors

    def _get_training_components(self) -> dict[str, Any]:
        """Get available training components from the new architecture."""
        try:
            return {
                "loss_functions": list(CORE_LOSSES.keys()),
                "optimizers": list(CORE_OPTIMIZERS.keys()),
            }
        except Exception as e:
            raise RuntimeError(f"Failed to load training components: {e}") from e

    def _get_trainer_examples(self, user_context: str) -> list[dict[str, Any]]:
        """Get relevant trainer examples."""
        examples = self.example_repository.retrieve_relevant_trainer_examples(
            user_context, max_examples=1
        )
        return [
            {
                "schema": ex.schema,
                "name": ex.name,
                "model_profile": ex.model_profile,
                "data_profile": ex.data_profile,
            }
            for ex in examples
        ]

    def _extract_model_profile(self, model_spec: str) -> dict[str, Any]:
        """Extract model profile information from model specification."""
        try:
            model_dict = yaml.safe_load(model_spec)
            if not isinstance(model_dict, dict):
                return {"error": "Invalid model specification"}

            # Extract basic model information
            profile = {
                "inputs": model_dict.get("inputs", {}),
                "outputs": model_dict.get("outputs", {}),
            }

            # Analyze model architecture for task type inference
            graph = model_dict.get("graph", [])

            # Look for activation patterns to infer task type
            has_sigmoid = any(node.get("type") == "torch.nn.Sigmoid" for node in graph)
            has_softmax = any(node.get("type") == "torch.nn.Softmax" for node in graph)

            # Find the final layer to determine output size
            final_layer = None
            for node in graph:
                if node.get("type") == "torch.nn.Linear":
                    final_layer = node

            if final_layer:
                out_features = final_layer.get("params", {}).get("out_features", 1)
                profile["final_layer_output_size"] = out_features

                # Infer task type and recommend loss
                if has_sigmoid and out_features == 1:
                    profile["inferred_task_type"] = "binary_classification"
                    profile["recommended_loss"] = "torch.nn.BCELoss"
                elif has_softmax or out_features > 1:
                    profile["inferred_task_type"] = "multiclass_classification"
                    profile["recommended_loss"] = "torch.nn.CrossEntropyLoss"
                else:
                    profile["inferred_task_type"] = "regression"
                    profile["recommended_loss"] = "torch.nn.MSELoss"

            # Extract input shape information for batch size recommendations
            for _input_name, input_spec in profile["inputs"].items():
                shape = input_spec.get("shape", [])
                if len(shape) >= 2:
                    num_features = shape[1] if shape[1] is not None else 0
                    profile["num_features"] = num_features

                    # Recommend batch size based on feature count
                    if num_features < 10:
                        profile["recommended_batch_size"] = 16
                    elif num_features < 50:
                        profile["recommended_batch_size"] = 32
                    else:
                        profile["recommended_batch_size"] = 64

            return profile

        except Exception as e:
            return {"error": f"Failed to extract model profile: {str(e)}"}
