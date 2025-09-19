"""Predictor specification generator agent."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from ....database.services import ServiceContainer
from ....graph import PredictorSpec, validate_predictor_dict
from ...agent import ArcAgent
from ..shared.base_agent import AgentError, BaseAgent
from ..shared.example_repository import ExampleRepository

logger = logging.getLogger(__name__)


class PredictorGeneratorError(AgentError):
    """Raised when predictor generation fails."""


class PredictorGeneratorAgent(BaseAgent):
    """Specialized agent for generating Arc predictor specifications using LLM."""

    def __init__(self, services: ServiceContainer, agent: ArcAgent):
        """Initialize predictor generator agent.

        Args:
            services: Service container for database access
            agent: Arc agent for LLM interactions
        """
        super().__init__(services, agent)
        self.example_repository = ExampleRepository()

    def get_template_directory(self) -> Path:
        """Get the template directory for predictor generation.

        Returns:
            Path to the predictor generator template directory
        """
        return Path(__file__).parent / "templates"

    async def generate_predictor(
        self,
        model_id: str,
        user_context: str,
        model_version: int | None = None,
        model_outputs: dict[str, str] | None = None,
        prediction_requirements: str | None = None,
        max_iterations: int = 3,
    ) -> tuple[PredictorSpec, str]:
        """Generate Arc predictor specification based on model and user context.

        Args:
            model_id: ID of the model to create predictor for
            user_context: User description of prediction requirements
            model_version: Version of the model (None for latest)
            model_outputs: Available model outputs {output_name: description}
            prediction_requirements: Specific prediction output requirements
            max_iterations: Maximum number of generation attempts

        Returns:
            Tuple of (parsed PredictorSpec, raw YAML string)

        Raises:
            PredictorGeneratorError: If generation fails after max iterations
        """
        logger.info(
            f"Generating predictor for model {model_id}, version {model_version}"
        )

        # Build simple context for LLM
        context = {
            "model_id": model_id,
            "model_version": model_version or "latest",
            "user_context": user_context,
            "outputs_info": self._format_model_outputs(model_outputs),
            "requirements_info": prediction_requirements
            or "No specific requirements provided",
            "predictor_examples": self._get_predictor_examples(user_context),
        }

        # Use the base agent validation loop
        try:
            predictor_spec, predictor_yaml = await self._generate_with_validation_loop(
                context, self._validate_predictor_comprehensive, max_iterations
            )

            return predictor_spec, predictor_yaml

        except Exception as e:
            logger.error(f"Predictor generation failed: {e}")
            raise PredictorGeneratorError(f"Failed to generate predictor: {e}") from e

    def _format_model_outputs(self, model_outputs: dict[str, str] | None) -> str:
        """Format model outputs for prompt."""
        if model_outputs:
            outputs_list = []
            for output_name, description in model_outputs.items():
                outputs_list.append(f"  - {output_name}: {description}")
            return "Available model outputs:\n" + "\n".join(outputs_list)
        else:
            return "Model outputs: Not specified (will use all model outputs)"

    def _get_predictor_examples(self, user_context: str) -> list[dict[str, Any]]:
        """Get relevant predictor examples."""
        examples = self.example_repository.retrieve_relevant_predictor_examples(
            user_context, max_examples=1
        )
        return [{"schema": ex.schema, "name": ex.name} for ex in examples]

    def _validate_predictor_comprehensive(
        self, yaml_content: str, _context: dict[str, Any]
    ) -> dict[str, Any]:
        """Comprehensive validation of generated predictor with detailed reporting."""
        try:
            # Parse YAML
            import yaml

            data = yaml.safe_load(yaml_content)

            if not isinstance(data, dict):
                return {
                    "valid": False,
                    "object": None,
                    "error": "YAML must contain a dictionary",
                }

            # Check required top-level fields for predictor
            required_fields = ["name", "model_id"]
            missing_fields = [field for field in required_fields if field not in data]

            if missing_fields:
                return {
                    "valid": False,
                    "object": None,
                    "error": f"Missing required fields: {', '.join(missing_fields)}",
                }

            # Validate and create PredictorSpec
            predictor_spec = validate_predictor_dict(data)

            logger.debug(f"Generated predictor: {predictor_spec.name}")
            return {"valid": True, "object": predictor_spec, "error": None}

        except Exception as e:
            logger.error(f"Failed to validate predictor YAML: {e}")
            logger.debug(f"Invalid YAML content: {yaml_content}")
            return {"valid": False, "object": None, "error": str(e)}

    async def generate_predictor_from_model_spec(
        self,
        model_id: str,
        model_spec_dict: dict[str, Any],
        user_context: str,
        model_version: int | None = None,
    ) -> tuple[PredictorSpec, str]:
        """Generate predictor specification from a model specification.

        Args:
            model_id: ID of the model
            model_spec_dict: Model specification dictionary
            user_context: User description of prediction requirements
            model_version: Version of the model

        Returns:
            Tuple of (PredictorSpec, YAML string)
        """
        # Extract model outputs with descriptions
        model_outputs = {}
        if "outputs" in model_spec_dict:
            for output_name, output_ref in model_spec_dict["outputs"].items():
                model_outputs[output_name] = f"Model output from {output_ref}"

        return await self.generate_predictor(
            model_id=model_id,
            user_context=user_context,
            model_version=model_version,
            model_outputs=model_outputs,
        )
