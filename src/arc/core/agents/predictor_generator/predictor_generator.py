"""Predictor specification generator agent."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from arc.core.agents.shared.base_agent import AgentError, BaseAgent
from arc.core.agents.shared.example_repository import ExampleRepository
from arc.database.services import ServiceContainer

logger = logging.getLogger(__name__)


class PredictorGeneratorError(AgentError):
    """Raised when predictor generation fails."""


class PredictorGeneratorAgent(BaseAgent):
    """Specialized agent for generating Arc predictor specifications using LLM."""

    def __init__(
        self,
        services: ServiceContainer,
        api_key: str,
        base_url: str | None = None,
        model: str | None = None,
    ):
        """Initialize predictor generator agent.

        Args:
            services: Service container for database access
            api_key: API key for LLM interactions
            base_url: Optional base URL
            model: Optional model name
        """
        super().__init__(services, api_key, base_url, model)
        self.example_repository = ExampleRepository()

    def get_template_directory(self) -> Path:
        """Get the template directory for predictor generation.

        Returns:
            Path to the predictor generator template directory
        """
        return Path(__file__).parent / "templates"

    async def generate_predictor(
        self,
        user_context: str,
        model_spec_path: str,
        trainer_spec_path: str | None = None,
    ) -> str:
        """Generate Arc predictor specification based on model and trainer specs.

        Args:
            user_context: User description of prediction requirements
            model_spec_path: Path to model YAML specification file
            trainer_spec_path: Path to trainer YAML specification file (optional)

        Returns:
            Generated predictor YAML string

        Raises:
            PredictorGeneratorError: If generation fails
        """
        logger.info(f"Generating predictor from model spec: {model_spec_path}")

        # Read model spec from file
        try:
            model_spec = Path(model_spec_path).read_text(encoding="utf-8")
        except OSError as e:
            raise PredictorGeneratorError(
                f"Failed to read model spec file {model_spec_path}: {e}"
            ) from e

        # Read trainer spec from file if provided
        trainer_spec = None
        if trainer_spec_path:
            try:
                trainer_spec = Path(trainer_spec_path).read_text(encoding="utf-8")
            except OSError as e:
                raise PredictorGeneratorError(
                    f"Failed to read trainer spec file {trainer_spec_path}: {e}"
                ) from e

        # Build context for LLM with specs and user requirements
        context = {
            "user_context": user_context,
            "model_spec": model_spec,
            "trainer_spec": trainer_spec,
            "model_profile": self._extract_model_profile(model_spec),
            "trainer_profile": self._extract_trainer_profile(trainer_spec)
            if trainer_spec
            else None,
            "predictor_examples": self._get_predictor_examples(user_context),
        }

        # Generate predictor specification with single attempt
        try:
            _validated_data, predictor_yaml = await self._generate_with_validation_loop(
                context, self._validate_predictor_comprehensive, 1
            )

            return predictor_yaml

        except Exception as e:
            logger.error(f"Predictor generation failed: {e}")
            raise PredictorGeneratorError(f"Failed to generate predictor: {e}") from e

    def _extract_model_profile(self, model_spec: str) -> dict[str, Any]:
        """Extract relevant information from model specification.

        Args:
            model_spec: Model YAML specification

        Returns:
            Dictionary with model profile information
        """
        try:
            import yaml

            model_data = yaml.safe_load(model_spec)
            if not model_data:
                return {}

            profile = {}

            # Extract inputs information
            if "inputs" in model_data:
                inputs_info = []
                for input_name, input_spec in model_data["inputs"].items():
                    inputs_info.append(
                        {
                            "name": input_name,
                            "dtype": input_spec.get("dtype"),
                            "shape": input_spec.get("shape"),
                            "columns": input_spec.get("columns"),
                        }
                    )
                profile["inputs"] = inputs_info

            # Extract outputs information
            if "outputs" in model_data:
                profile["outputs"] = list(model_data["outputs"].keys())
                profile["output_mappings"] = model_data["outputs"]

            return profile

        except Exception as e:
            logger.warning(f"Failed to extract model profile: {e}")
            return {}

    def _extract_trainer_profile(self, trainer_spec: str) -> dict[str, Any]:
        """Extract relevant information from trainer specification.

        Args:
            trainer_spec: Trainer YAML specification

        Returns:
            Dictionary with trainer profile information
        """
        try:
            import yaml

            trainer_data = yaml.safe_load(trainer_spec)
            if not trainer_data:
                return {}

            profile = {}

            # Extract optimizer information
            if "optimizer" in trainer_data:
                profile["optimizer"] = trainer_data["optimizer"]

            # Extract loss function information
            if "loss" in trainer_data:
                profile["loss"] = trainer_data["loss"]

            # Extract training configuration
            if "config" in trainer_data:
                profile["training_config"] = trainer_data["config"]

            return profile

        except Exception as e:
            logger.warning(f"Failed to extract trainer profile: {e}")
            return {}

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
            required_fields = ["name"]
            missing_fields = [field for field in required_fields if field not in data]

            if missing_fields:
                return {
                    "valid": False,
                    "object": None,
                    "error": f"Missing required fields: {', '.join(missing_fields)}",
                }

            # Basic YAML structure validation without creating PredictorSpec
            name = data.get("name")
            if not isinstance(name, str) or not name.strip():
                return {
                    "valid": False,
                    "object": None,
                    "error": "name must be a non-empty string",
                }

            logger.debug(f"Generated predictor YAML: {name}")
            return {"valid": True, "object": data, "error": None}

        except Exception as e:
            logger.error(f"Failed to validate predictor YAML: {e}")
            logger.debug(f"Invalid YAML content: {yaml_content}")
            return {"valid": False, "object": None, "error": str(e)}
