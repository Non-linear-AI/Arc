"""Arc evaluator specification generation agent."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

from arc.core.agents.shared.base_agent import AgentError, BaseAgent
from arc.core.agents.shared.example_repository import ExampleRepository
from arc.database.services import ServiceContainer
from arc.graph.evaluator import (
    EvaluatorSpec,
    validate_evaluator_dict,
)


class EvaluatorGeneratorError(AgentError):
    """Raised when evaluator generation fails."""


class EvaluatorGeneratorAgent(BaseAgent):
    """Specialized agent for generating Arc evaluator specifications using LLM."""

    AVAILABLE_METRICS = {
        "classification": [
            "accuracy",
            "precision",
            "recall",
            "f1_score",
            "auc",
        ],
        "regression": [
            "mse",
            "mae",
            "rmse",
            "r2_score",
        ],
    }

    def __init__(
        self,
        services: ServiceContainer,
        api_key: str,
        base_url: str | None = None,
        model: str | None = None,
    ):
        """Initialize evaluator generator agent.

        Args:
            services: Service container for database access
            api_key: API key for LLM interactions
            base_url: Optional base URL
            model: Optional model name
        """
        super().__init__(services, api_key, base_url, model)
        self.example_repository = ExampleRepository()

    def get_template_directory(self) -> Path:
        """Get the template directory for evaluator generation.

        Returns:
            Path to the evaluator generator template directory
        """
        return Path(__file__).parent / "templates"

    async def generate_evaluator(
        self,
        name: str,
        user_context: str,
        trainer_ref: str,
        trainer_spec_yaml: str,
        dataset: str,
        target_column: str,
        target_column_exists: bool = True,
        existing_yaml: str | None = None,
        editing_instructions: str | None = None,
    ) -> tuple[EvaluatorSpec, str]:
        """Generate Arc evaluator specification based on trainer and context.

        Args:
            name: Evaluator name for the specification
            user_context: User description of desired evaluation setup
            trainer_ref: Reference to the trainer (e.g., "diabetes_trainer")
            trainer_spec_yaml: Trainer specification YAML content
            dataset: Test dataset table name
            target_column: Target column name in the dataset
            target_column_exists: Whether target column exists in dataset
            existing_yaml: Optional existing YAML to edit
            editing_instructions: Optional instructions for editing existing YAML

        Returns:
            Tuple of (parsed EvaluatorSpec, raw YAML string)

        Raises:
            EvaluatorGeneratorError: If generation fails
        """
        # Build context for LLM
        context = {
            "evaluator_name": name,
            "user_intent": user_context,
            "trainer_ref": trainer_ref,
            "trainer_spec": trainer_spec_yaml,
            "dataset": dataset,
            "target_column": target_column,
            "target_column_exists": target_column_exists,
            "trainer_profile": self._extract_trainer_profile(trainer_spec_yaml),
            "available_metrics": self._get_available_metrics(),
            "examples": self._get_evaluator_examples(user_context),
            "is_editing": existing_yaml is not None,
            "existing_yaml": existing_yaml,
            "editing_instructions": editing_instructions,
        }

        # Generate evaluator specification with single attempt
        try:
            evaluator_spec, evaluator_yaml = await self._generate_with_validation_loop(
                context, self._validate_evaluator_comprehensive, 1
            )

            return evaluator_spec, evaluator_yaml

        except AgentError as e:
            raise EvaluatorGeneratorError(str(e)) from e

    def _validate_evaluator_comprehensive(
        self, evaluator_yaml: str, context: dict[str, Any]
    ) -> dict[str, Any]:
        """Validate generated evaluator with detailed error reporting.

        Args:
            evaluator_yaml: Generated YAML evaluator string
            context: Generation context for validation

        Returns:
            Dictionary with validation results:
            {"valid": bool, "object": EvaluatorSpec, "error": str}
        """
        try:
            # Parse YAML
            evaluator_dict = self._validate_yaml_syntax(evaluator_yaml)

            # Check required top-level fields for evaluator
            required_fields = ["name", "trainer_ref", "dataset", "target_column"]
            missing_fields = [
                field for field in required_fields if field not in evaluator_dict
            ]
            if missing_fields:
                return {
                    "valid": False,
                    "error": f"Missing required evaluator fields: {missing_fields}",
                }

            # Validate evaluator structure using dedicated validator
            validate_evaluator_dict(evaluator_dict)

            # Validate metrics against available metrics
            metric_errors = self._validate_evaluator_metrics(evaluator_dict, context)
            if metric_errors:
                return {
                    "valid": False,
                    "error": f"Metric validation errors: {metric_errors}",
                }

            # Parse into EvaluatorSpec object
            try:
                evaluator_spec = EvaluatorSpec.from_yaml(evaluator_yaml)
            except Exception as e:
                return {
                    "valid": False,
                    "error": f"Failed to parse into EvaluatorSpec: {str(e)}",
                }

            return {"valid": True, "object": evaluator_spec, "error": None}

        except Exception as e:
            return {
                "valid": False,
                "error": f"Validation exception: {str(e)}",
            }

    def _validate_evaluator_metrics(
        self, evaluator_dict: dict, context: dict[str, Any]
    ) -> list[str]:
        """Validate metrics against available metrics."""
        errors = []

        metrics = evaluator_dict.get("metrics", [])
        if not metrics:
            # Metrics are optional, will use defaults
            return errors

        # Get all available metrics from both classification and regression
        all_available_metrics = []
        for metric_list in context.get("available_metrics", {}).values():
            all_available_metrics.extend(metric_list)

        # Check each metric
        for metric in metrics:
            if metric not in all_available_metrics:
                errors.append(
                    f"Unknown metric '{metric}'. Available: {all_available_metrics}"
                )

        return errors

    def _get_available_metrics(self) -> dict[str, list[str]]:
        """Get available evaluation metrics."""
        return self.AVAILABLE_METRICS

    def _get_evaluator_examples(self, _user_context: str) -> list[dict[str, Any]]:
        """Get relevant evaluator examples."""
        # For now, return empty list. Can add example repository support later.
        return []

    def _extract_trainer_profile(self, trainer_spec: str) -> dict[str, Any]:
        """Extract trainer profile information from trainer specification."""
        try:
            trainer_dict = yaml.safe_load(trainer_spec)
            if not isinstance(trainer_dict, dict):
                return {"error": "Invalid trainer specification"}

            # Extract basic trainer information
            profile = {
                "model_ref": trainer_dict.get("model_ref", ""),
                "optimizer": trainer_dict.get("optimizer", {}),
            }

            # Infer task type from optimizer or other hints
            # This is a simple heuristic - could be improved
            epochs = trainer_dict.get(
                "epochs", trainer_dict.get("config", {}).get("epochs", 10)
            )
            batch_size = trainer_dict.get(
                "batch_size", trainer_dict.get("config", {}).get("batch_size", 32)
            )

            profile["epochs"] = epochs
            profile["batch_size"] = batch_size

            # Suggest default metrics based on common patterns
            # This is heuristic - the LLM will make the final decision
            profile["suggested_metrics"] = [
                "accuracy",
                "precision",
                "recall",
                "f1_score",
            ]

            return profile

        except Exception as e:
            return {"error": f"Failed to extract trainer profile: {str(e)}"}
