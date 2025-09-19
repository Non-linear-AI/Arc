"""Repository for Arc-Graph schema examples with context mapping."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

# Note: Example repository no longer needs validation as examples are pre-validated


class ModelExample:
    """Represents a model specification example."""

    def __init__(
        self,
        name: str,
        user_intent: str,
        schema: str,
        data_profile: dict[str, Any],
        explanation: str | None = None,
    ):
        """Initialize model example.

        Args:
            name: Descriptive name for the example
            user_intent: The user's intent/goal that led to this model
            schema: The model YAML specification content
            data_profile: Sample data profile that matches this model
            explanation: Optional explanation of the model architecture
        """
        self.name = name
        self.user_intent = user_intent
        self.schema = schema
        self.data_profile = data_profile
        self.explanation = explanation


class TrainerExample:
    """Represents a trainer specification example."""

    def __init__(
        self,
        name: str,
        user_intent: str,
        schema: str,
        model_profile: dict[str, Any],
        data_profile: dict[str, Any],
        explanation: str | None = None,
    ):
        """Initialize trainer example.

        Args:
            name: Descriptive name for the example
            user_intent: The user's intent/goal for training
            schema: The trainer YAML specification content
            model_profile: Model specification profile that this trainer trains
            data_profile: Data profile that matches this trainer's data requirements
            explanation: Optional explanation of the training configuration
        """
        self.name = name
        self.user_intent = user_intent
        self.schema = schema
        self.model_profile = model_profile
        self.data_profile = data_profile
        self.explanation = explanation


class PredictorExample:
    """Represents a predictor specification example."""

    def __init__(
        self,
        name: str,
        user_intent: str,
        schema: str,
        model_profile: dict[str, Any],
        explanation: str | None = None,
    ):
        """Initialize predictor example.

        Args:
            name: Descriptive name for the example
            user_intent: The user's intent/goal for prediction
            schema: The predictor YAML specification content
            model_profile: Model specification profile that this predictor uses
            explanation: Optional explanation of the prediction configuration
        """
        self.name = name
        self.user_intent = user_intent
        self.schema = schema
        self.model_profile = model_profile
        self.explanation = explanation


# Backward compatibility alias
SchemaExample = ModelExample


class ExampleRepository:
    """Repository for managing model and trainer examples."""

    def __init__(self):
        """Initialize the example repository."""
        self.model_examples: list[ModelExample] = []
        self.trainer_examples: list[TrainerExample] = []
        self.predictor_examples: list[PredictorExample] = []
        self._load_builtin_examples()

    @property
    def examples(self) -> list[ModelExample]:
        """Backward compatibility property for model examples."""
        return self.model_examples

    def _load_builtin_examples(self):
        """Load built-in model and trainer examples."""
        examples_dir = Path(__file__).parent / "examples"

        # Load model example
        model_path = examples_dir / "diabetes_binary_classification.model.yaml"
        if model_path.exists():
            try:
                model_content = model_path.read_text(encoding="utf-8")
                self.model_examples.append(
                    ModelExample(
                        name="Diabetes Binary Classification Model",
                        user_intent="Binary classification model for diabetes "
                        "prediction using health metrics",
                        schema=model_content,
                        data_profile={},  # Empty - not used in prompt anyway
                        explanation="Simple binary classification model with direct "
                        "column references, torch.nn.Linear + torch.nn.Sigmoid layers",
                    )
                )
            except OSError as e:
                logging.error(f"Failed to read model example: {e}")
        else:
            logging.warning(f"Model example not found at {model_path}")

        # Load trainer example
        trainer_path = examples_dir / "diabetes_binary_classification.trainer.yaml"
        if trainer_path.exists():
            try:
                trainer_content = trainer_path.read_text(encoding="utf-8")
                # Create model and data profiles for the trainer example
                model_profile = {
                    "inputs": {
                        "patient_data": {
                            "dtype": "float32",
                            "shape": [None, 8],
                            "columns": [
                                "pregnancies",
                                "glucose",
                                "blood_pressure",
                                "skin_thickness",
                                "insulin",
                                "bmi",
                                "diabetes_pedigree",
                                "age",
                            ],
                        }
                    },
                    "outputs": {
                        "logits": "classifier.output",
                        "prediction": "sigmoid.output",
                    },
                    "architecture": "binary_classification",
                    "num_features": 8,
                    "output_type": "sigmoid",
                }

                data_profile = {
                    "target_column": "outcome",
                    "feature_columns": [
                        "pregnancies",
                        "glucose",
                        "blood_pressure",
                        "skin_thickness",
                        "insulin",
                        "bmi",
                        "diabetes_pedigree",
                        "age",
                    ],
                    "task_type": "binary_classification",
                    "num_features": 8,
                    "target_type": "binary",
                }

                self.trainer_examples.append(
                    TrainerExample(
                        name="Diabetes Binary Classification Trainer",
                        user_intent="Training configuration for binary classification "
                        "with Adam optimizer and BCELoss",
                        schema=trainer_content,
                        model_profile=model_profile,
                        data_profile=data_profile,
                        explanation="Standard training setup with torch.optim.Adam "
                        "optimizer and torch.nn.BCELoss for binary classification",
                    )
                )
            except OSError as e:
                logging.error(f"Failed to read trainer example: {e}")
        else:
            logging.warning(f"Trainer example not found at {trainer_path}")

        # Load predictor example
        predictor_path = examples_dir / "diabetes_binary_classification.predictor.yaml"
        if predictor_path.exists():
            try:
                predictor_content = predictor_path.read_text(encoding="utf-8")
                # Use the same model profile as trainer
                model_profile = {
                    "inputs": {
                        "patient_data": {
                            "dtype": "float32",
                            "shape": [None, 8],
                            "columns": [
                                "pregnancies",
                                "glucose",
                                "blood_pressure",
                                "skin_thickness",
                                "insulin",
                                "bmi",
                                "diabetes_pedigree",
                                "age",
                            ],
                        }
                    },
                    "outputs": {
                        "logits": "classifier.output",
                        "prediction": "sigmoid.output",
                    },
                    "architecture": "binary_classification",
                    "num_features": 8,
                    "output_type": "sigmoid",
                }

                self.predictor_examples.append(
                    PredictorExample(
                        name="Diabetes Binary Classification Predictor",
                        user_intent="Prediction service for diabetes classification "
                        "with custom output mapping",
                        schema=predictor_content,
                        model_profile=model_profile,
                        explanation="Predictor that maps model outputs to prediction, "
                        "confidence, and raw_logits for flexible inference",
                    )
                )
            except OSError as e:
                logging.error(f"Failed to read predictor example: {e}")
        else:
            logging.warning(f"Predictor example not found at {predictor_path}")

    def retrieve_relevant_examples(
        self, _user_context: str, _data_profile: dict[str, Any], max_examples: int = 1
    ) -> list[ModelExample]:
        """Retrieve relevant model examples for the given context.

        Args:
            user_context: User description of their ML objective
            data_profile: Analysis of their data table
            max_examples: Maximum number of examples to return

        Returns:
            List of relevant model examples
        """
        # For now, just return available model examples
        return self.model_examples[:max_examples]

    def retrieve_relevant_model_examples(
        self, _user_context: str, _data_profile: dict[str, Any], max_examples: int = 1
    ) -> list[ModelExample]:
        """Retrieve relevant model examples for the given context.

        Args:
            user_context: User description of their ML objective
            data_profile: Analysis of their data table
            max_examples: Maximum number of examples to return

        Returns:
            List of relevant model examples
        """
        # For now, just return available model examples
        return self.model_examples[:max_examples]

    def retrieve_relevant_trainer_examples(
        self, _user_context: str, max_examples: int = 1
    ) -> list[TrainerExample]:
        """Retrieve relevant trainer examples for the given context.

        Args:
            user_context: User description of their training objective
            max_examples: Maximum number of examples to return

        Returns:
            List of relevant trainer examples
        """
        # For now, just return available trainer examples
        return self.trainer_examples[:max_examples]

    def retrieve_relevant_predictor_examples(
        self, _user_context: str, max_examples: int = 1
    ) -> list[PredictorExample]:
        """Retrieve relevant predictor examples for the given context.

        Args:
            user_context: User description of their prediction objective
            max_examples: Maximum number of examples to return

        Returns:
            List of relevant predictor examples
        """
        # For now, just return available predictor examples
        return self.predictor_examples[:max_examples]

    def get_examples_by_type(self, _use_case_type: str) -> list[SchemaExample]:
        """Get examples by use case type.

        Args:
            use_case_type: Type of use case (not used in current implementation)

        Returns:
            List of examples
        """
        return self.examples
