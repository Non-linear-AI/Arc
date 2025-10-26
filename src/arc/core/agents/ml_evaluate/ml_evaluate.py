"""Arc ML evaluate agent."""

from __future__ import annotations

import logging
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

logger = logging.getLogger(__name__)


class MLEvaluateError(AgentError):
    """Raised when evaluator generation fails."""


class MLEvaluateAgent(BaseAgent):
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
        # Note: knowledge_loader is now initialized in BaseAgent

    def get_template_directory(self) -> Path:
        """Get the template directory for evaluator generation.

        Returns:
            Path to the evaluator generator template directory
        """
        return Path(__file__).parent / "templates"

    async def generate_evaluator(
        self,
        name: str,
        instruction: str,
        trainer_ref: str,
        trainer_spec_yaml: str,
        dataset: str,
        target_column: str,
        target_column_exists: bool = True,
        existing_yaml: str | None = None,
        ml_plan_evaluation: str | None = None,
        recommended_knowledge_ids: list[str] | None = None,
        conversation_history: list[dict[str, str]] | None = None,
    ) -> tuple[EvaluatorSpec, str, list[dict[str, str]]]:
        """Generate Arc evaluator specification based on trainer and instruction.

        Args:
            name: Evaluator name for the specification
            instruction: User's instruction for evaluation setup.
                For generation: requirements for evaluation configuration.
                For editing: changes to make to existing YAML.
            trainer_ref: Reference to the trainer (e.g., "diabetes_trainer")
            trainer_spec_yaml: Trainer specification YAML content
            dataset: Test dataset table name
            target_column: Target column name in the dataset
            target_column_exists: Whether target column exists in dataset
            existing_yaml: Optional existing YAML to edit (editing mode)
            ml_plan_evaluation: Optional evaluation guidance from ML plan
            recommended_knowledge_ids: Optional list of knowledge IDs
                recommended by ML Plan
            conversation_history: Optional conversation history for editing workflow

        Returns:
            Tuple of (parsed EvaluatorSpec, raw YAML string, conversation_history)

        Raises:
            EvaluatorGeneratorError: If generation fails
        """
        # Route to appropriate generation path
        if conversation_history is None:
            # Fresh generation - build full context
            return await self._generate_fresh(
                name=name,
                instruction=instruction,
                trainer_ref=trainer_ref,
                trainer_spec_yaml=trainer_spec_yaml,
                dataset=dataset,
                target_column=target_column,
                target_column_exists=target_column_exists,
                existing_yaml=existing_yaml,
                ml_plan_evaluation=ml_plan_evaluation,
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
        trainer_ref: str,
        trainer_spec_yaml: str,
        dataset: str,
        target_column: str,
        target_column_exists: bool = True,
        existing_yaml: str | None = None,
        ml_plan_evaluation: str | None = None,
        recommended_knowledge_ids: list[str] | None = None,
    ) -> tuple[EvaluatorSpec, str, list[dict[str, str]]]:
        """Fresh generation with full context building.

        This path is used for initial generation or when starting a new conversation.
        It builds the complete system message with knowledge loading.
        """
        # Don't pre-load knowledge - let agent discover what's available using tools
        # This prevents errors from non-existent knowledge IDs and gives agent flexibility
        recommended_knowledge_guidance = ""
        if recommended_knowledge_ids:
            recommended_knowledge_guidance = (
                f"\n\nRecommended knowledge IDs from ML Plan: "
                f"{', '.join(recommended_knowledge_ids)}\n"
                f"Note: Use list_available_knowledge first to see what exists, "
                f"then read relevant documents."
            )

        # Build system message with all context
        system_message = self._render_template(
            "prompt.j2",
            {
                "evaluator_name": name,
                "instruction": instruction,
                "trainer_ref": trainer_ref,
                "trainer_spec": trainer_spec_yaml,
                "dataset": dataset,
                "target_column": target_column,
                "target_column_exists": target_column_exists,
                "trainer_profile": self._extract_trainer_profile(trainer_spec_yaml),
                "model_outputs": self._extract_model_outputs(trainer_spec_yaml),
                "available_metrics": self._get_available_metrics(),
                "examples": self._get_evaluator_examples(instruction),
                "existing_yaml": existing_yaml,
                "ml_plan_evaluation": ml_plan_evaluation,
                "recommended_knowledge": recommended_knowledge_guidance,
            },
        )

        # User message guides tool usage
        if existing_yaml:
            user_message = (
                f"Edit the existing evaluator specification with these "
                f"changes: {instruction}. "
                "If you need evaluation guidance, first use list_available_knowledge "
                "to see what's available, then read_knowledge_content for relevant documents."
            )
        else:
            user_message = (
                f"Generate the evaluator specification for '{name}'. "
                "If you need evaluation guidance, first use list_available_knowledge "
                "to see what's available, then read_knowledge_content for relevant documents."
            )

        # Get ML tools from BaseAgent
        tools = self._get_ml_tools()

        # Generate with multi-turn tool support
        try:
            (
                evaluator_spec,
                evaluator_yaml,
                conversation_history,
            ) = await self._generate_with_tools(
                system_message=system_message,
                user_message=user_message,
                tools=tools,
                tool_executor=self._execute_ml_tool,
                validator_func=self._validate_evaluator_comprehensive,
                validation_context={
                    "available_metrics": self._get_available_metrics(),
                },
                max_iterations=3,
                conversation_history=None,  # Fresh start
            )

            return evaluator_spec, evaluator_yaml, conversation_history

        except AgentError as e:
            raise MLEvaluateError(str(e)) from e

    async def _continue_conversation(
        self,
        feedback: str,
        conversation_history: list[dict[str, str]],
    ) -> tuple[EvaluatorSpec, str, list[dict[str, str]]]:
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
                evaluator_spec,
                evaluator_yaml,
                updated_history,
            ) = await self._generate_with_tools(
                system_message="",  # Not used - already in conversation_history
                user_message=feedback,
                tools=tools,
                tool_executor=self._execute_ml_tool,
                validator_func=self._validate_evaluator_comprehensive,
                validation_context={
                    "available_metrics": self._get_available_metrics(),
                },
                max_iterations=3,
                conversation_history=conversation_history,
            )

            return evaluator_spec, evaluator_yaml, updated_history

        except AgentError as e:
            raise MLEvaluateError(str(e)) from e

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

    def _extract_model_outputs(self, trainer_spec: str) -> list[str] | None:
        """Extract model output names from the trainer's model specification.

        Args:
            trainer_spec: Trainer specification YAML

        Returns:
            List of model output names, or None if model has single output or error
        """
        try:
            trainer_dict = yaml.safe_load(trainer_spec)
            if not isinstance(trainer_dict, dict):
                logger.debug("Trainer spec is not a dict")
                return None

            # Get model_ref from trainer
            model_ref = trainer_dict.get("model_ref")
            if not model_ref:
                logger.debug("No model_ref in trainer spec")
                return None

            logger.debug(f"Looking up model: {model_ref}")

            # Look up model spec in database
            model = self.services.models.get_model_by_id(model_ref)
            if not model:
                logger.debug(f"Model {model_ref} not found in database")
                return None

            # Parse model spec to get outputs
            model_spec_dict = yaml.safe_load(model.spec)
            if not isinstance(model_spec_dict, dict):
                logger.debug("Model spec is not a dict")
                return None

            outputs = model_spec_dict.get("outputs", {})
            if not isinstance(outputs, dict):
                logger.debug(f"Outputs is not a dict: {type(outputs)}")
                return None

            logger.debug(f"Found {len(outputs)} outputs: {list(outputs.keys())}")

            # If model has multiple outputs, return the list
            if len(outputs) > 1:
                logger.info(
                    f"Model has multiple outputs: {list(outputs.keys())} - "
                    "will include in prompt"
                )
                return list(outputs.keys())

            # Single output - no need to specify
            logger.debug("Model has single output - no need to specify output_name")
            return None

        except Exception as e:
            # Log error but don't fail - not critical
            logger.debug(f"Could not extract model outputs: {e}")
            return None
