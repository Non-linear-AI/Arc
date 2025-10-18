"""Arc ML trainer agent."""

from __future__ import annotations

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


class MLTrainerError(AgentError):
    """Raised when trainer generation fails."""


class MLTrainerAgent(BaseAgent):
    """Specialized agent for generating Arc trainer specifications using LLM."""

    def __init__(
        self,
        services: ServiceContainer,
        api_key: str,
        base_url: str | None = None,
        model: str | None = None,
    ):
        """Initialize trainer generator agent.

        Args:
            services: Service container for database access
            api_key: API key for LLM interactions
            base_url: Optional base URL
            model: Optional model name
        """
        super().__init__(services, api_key, base_url, model)
        self.example_repository = ExampleRepository()

    def get_template_directory(self) -> Path:
        """Get the template directory for trainer generation.

        Returns:
            Path to the trainer generator template directory
        """
        return Path(__file__).parent / "templates"

    async def generate_trainer(
        self,
        name: str,
        user_context: str,
        model_id: str,
        model_spec_yaml: str,
        existing_yaml: str | None = None,
        editing_instructions: str | None = None,
        ml_plan_training_config: str | None = None,
    ) -> tuple[TrainerSpec, str]:
        """Generate Arc trainer specification based on model and context.

        Args:
            name: Trainer name for the specification
            user_context: User description of desired training setup
            model_id: ID of the registered model (e.g., "diabetes-logistic-v1")
            model_spec_yaml: Model specification YAML content
            existing_yaml: Optional existing YAML to edit
            editing_instructions: Optional instructions for editing existing YAML
            ml_plan_training_config: Optional training configuration from ML plan

        Returns:
            Tuple of (parsed TrainerSpec, raw YAML string)

        Raises:
            TrainerGeneratorError: If generation fails
        """
        # Build simple context for LLM
        context = {
            "trainer_name": name,
            "user_intent": user_context,
            "model_id": model_id,
            "model_spec": model_spec_yaml,
            "model_profile": self._extract_model_profile(model_spec_yaml),
            "available_components": self._get_training_components(),
            "examples": self._get_trainer_examples(user_context),
            "is_editing": existing_yaml is not None,
            "existing_yaml": existing_yaml,
            "editing_instructions": editing_instructions,
            "ml_plan_training_config": ml_plan_training_config,
        }

        # Generate trainer specification with single attempt
        try:
            trainer_spec, trainer_yaml = await self._generate_with_validation_loop(
                context, self._validate_trainer_comprehensive, 1
            )

            return trainer_spec, trainer_yaml

        except AgentError as e:
            raise MLTrainerError(str(e)) from e

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
