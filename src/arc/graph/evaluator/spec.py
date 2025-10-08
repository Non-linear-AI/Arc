"""Evaluator specification for Arc-Graph."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


class EvaluatorValidationError(Exception):
    """Raised when evaluator specification validation fails."""


@dataclass
class EvaluatorSpec:
    """Evaluator specification for model assessment.

    Defines how to evaluate a trained model on a test dataset.
    """

    name: str
    trainer_ref: str  # Reference to trainer (which knows the model)
    dataset: str  # Test dataset table name
    target_column: str  # Target column name in the dataset
    metrics: list[str] | None = None  # Metric names (infer if None)
    version: int | None = None  # Optional: specific training version (None = latest)

    @classmethod
    def from_yaml(cls, yaml_str: str) -> EvaluatorSpec:
        """Parse EvaluatorSpec from YAML string.

        Args:
            yaml_str: YAML string containing evaluator specification

        Returns:
            EvaluatorSpec: Parsed and validated evaluator specification

        Raises:
            ValueError: If YAML is invalid
            EvaluatorValidationError: If specification is invalid
        """
        data = yaml.safe_load(yaml_str)
        if not isinstance(data, dict):
            raise ValueError("Top-level YAML must be a mapping")

        return validate_evaluator_dict(data)

    def to_yaml(self) -> str:
        """Convert EvaluatorSpec to YAML string.

        Returns:
            YAML string representation of the evaluator specification
        """
        data = {
            "name": self.name,
            "trainer_ref": self.trainer_ref,
            "dataset": self.dataset,
            "target_column": self.target_column,
        }

        if self.metrics is not None:
            data["metrics"] = self.metrics

        if self.version is not None:
            data["version"] = self.version

        return yaml.dump(data, default_flow_style=False, sort_keys=False)


def validate_evaluator_dict(data: dict[str, Any]) -> EvaluatorSpec:
    """Validate and convert evaluator dictionary to EvaluatorSpec.

    Args:
        data: Dictionary containing evaluator specification

    Returns:
        Validated EvaluatorSpec instance

    Raises:
        EvaluatorValidationError: If validation fails
    """
    try:
        # Required fields
        if "name" not in data:
            raise EvaluatorValidationError("Missing required field: name")
        if "trainer_ref" not in data:
            raise EvaluatorValidationError("Missing required field: trainer_ref")
        if "dataset" not in data:
            raise EvaluatorValidationError("Missing required field: dataset")
        if "target_column" not in data:
            raise EvaluatorValidationError("Missing required field: target_column")

        name = data["name"]
        trainer_ref = data["trainer_ref"]
        dataset = data["dataset"]
        target_column = data["target_column"]

        if not isinstance(name, str) or not name.strip():
            raise EvaluatorValidationError("name must be a non-empty string")
        if not isinstance(trainer_ref, str) or not trainer_ref.strip():
            raise EvaluatorValidationError("trainer_ref must be a non-empty string")
        if not isinstance(dataset, str) or not dataset.strip():
            raise EvaluatorValidationError("dataset must be a non-empty string")
        if not isinstance(target_column, str) or not target_column.strip():
            raise EvaluatorValidationError("target_column must be a non-empty string")

        # Optional fields
        metrics = data.get("metrics")
        if metrics is not None:
            if not isinstance(metrics, list):
                raise EvaluatorValidationError("metrics must be a list")
            if not all(isinstance(m, str) for m in metrics):
                raise EvaluatorValidationError("All metrics must be strings")
            if not all(m.strip() for m in metrics):
                raise EvaluatorValidationError("Metric names cannot be empty strings")

        version = data.get("version")
        if version is not None and not isinstance(version, int):
            raise EvaluatorValidationError("version must be an integer")

        return EvaluatorSpec(
            name=name.strip(),
            trainer_ref=trainer_ref.strip(),
            dataset=dataset.strip(),
            target_column=target_column.strip(),
            metrics=[m.strip() for m in metrics] if metrics else None,
            version=version,
        )

    except EvaluatorValidationError:
        raise
    except Exception as e:
        raise EvaluatorValidationError(f"Invalid evaluator specification: {e}") from e


def load_evaluator_from_yaml(file_path: str | Path) -> EvaluatorSpec:
    """Load evaluator specification from YAML file.

    Args:
        file_path: Path to YAML file

    Returns:
        Loaded and validated EvaluatorSpec

    Raises:
        EvaluatorValidationError: If file loading or validation fails
    """
    try:
        with open(file_path, encoding="utf-8") as f:
            data = yaml.safe_load(f)

        if not isinstance(data, dict):
            raise EvaluatorValidationError("YAML file must contain a dictionary")

        return validate_evaluator_dict(data)

    except yaml.YAMLError as e:
        raise EvaluatorValidationError(f"Invalid YAML syntax: {e}") from e
    except FileNotFoundError as e:
        raise EvaluatorValidationError(f"Evaluator file not found: {file_path}") from e
    except Exception as e:
        raise EvaluatorValidationError(
            f"Failed to load evaluator specification: {e}"
        ) from e


def save_evaluator_to_yaml(spec: EvaluatorSpec, file_path: str | Path) -> None:
    """Save evaluator specification to YAML file.

    Args:
        spec: EvaluatorSpec to save
        file_path: Path where to save the YAML file
    """
    data = {
        "name": spec.name,
        "trainer_ref": spec.trainer_ref,
        "dataset": spec.dataset,
        "target_column": spec.target_column,
    }

    if spec.metrics is not None:
        data["metrics"] = spec.metrics

    if spec.version is not None:
        data["version"] = spec.version

    with open(file_path, "w", encoding="utf-8") as f:
        yaml.dump(data, f, default_flow_style=False, sort_keys=False)
