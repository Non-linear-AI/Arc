"""Predictor specification for Arc-Graph."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


class PredictorValidationError(Exception):
    """Raised when predictor specification validation fails."""


@dataclass
class PredictorSpec:
    """Predictor specification for inference configuration.

    Defines which model to use and how to map its outputs for prediction.
    """

    name: str
    model_id: str
    model_version: int | None = None  # None = latest version
    outputs: dict[str, str] | None = None  # predictor_name -> model_output_name mapping

    @classmethod
    def from_yaml(cls, yaml_str: str) -> PredictorSpec:
        """Parse PredictorSpec from YAML string.

        Args:
            yaml_str: YAML string containing predictor specification

        Returns:
            PredictorSpec: Parsed and validated predictor specification

        Raises:
            ValueError: If YAML is invalid
            PredictorValidationError: If specification is invalid
        """
        data = yaml.safe_load(yaml_str)
        if not isinstance(data, dict):
            raise ValueError("Top-level YAML must be a mapping")

        return validate_predictor_dict(data)

    def to_yaml(self) -> str:
        """Convert PredictorSpec to YAML string.

        Returns:
            YAML string representation of the predictor specification
        """
        data = {
            "name": self.name,
            "model_id": self.model_id,
        }

        if self.model_version is not None:
            data["model_version"] = self.model_version

        if self.outputs is not None:
            data["outputs"] = self.outputs

        return yaml.dump(data, default_flow_style=False, sort_keys=False)


def validate_predictor_dict(data: dict[str, Any]) -> PredictorSpec:
    """Validate and convert predictor dictionary to PredictorSpec.

    Args:
        data: Dictionary containing predictor specification

    Returns:
        Validated PredictorSpec instance

    Raises:
        PredictorValidationError: If validation fails
    """
    try:
        # Required fields
        if "name" not in data:
            raise PredictorValidationError("Missing required field: name")
        if "model_id" not in data:
            raise PredictorValidationError("Missing required field: model_id")

        name = data["name"]
        model_id = data["model_id"]

        if not isinstance(name, str) or not name.strip():
            raise PredictorValidationError("name must be a non-empty string")
        if not isinstance(model_id, str) or not model_id.strip():
            raise PredictorValidationError("model_id must be a non-empty string")

        # Optional fields
        model_version = data.get("model_version")
        if model_version is not None and not isinstance(model_version, int):
            raise PredictorValidationError("model_version must be an integer")

        outputs = data.get("outputs")
        if outputs is not None:
            if not isinstance(outputs, dict):
                raise PredictorValidationError(
                    "outputs must be a dictionary mapping predictor names to "
                    "model output names"
                )

            # Validate output mappings
            for pred_name, model_output in outputs.items():
                if not isinstance(pred_name, str) or not pred_name.strip():
                    raise PredictorValidationError(
                        f"Predictor output name must be a non-empty string: {pred_name}"
                    )
                if not isinstance(model_output, str) or not model_output.strip():
                    raise PredictorValidationError(
                        f"Model output reference must be a non-empty string: "
                        f"{model_output}"
                    )

        return PredictorSpec(
            name=name.strip(),
            model_id=model_id.strip(),
            model_version=model_version,
            outputs=outputs,
        )

    except PredictorValidationError:
        raise
    except Exception as e:
        raise PredictorValidationError(f"Invalid predictor specification: {e}") from e


def load_predictor_from_yaml(file_path: str | Path) -> PredictorSpec:
    """Load predictor specification from YAML file.

    Args:
        file_path: Path to YAML file

    Returns:
        Loaded and validated PredictorSpec

    Raises:
        PredictorValidationError: If file loading or validation fails
    """
    try:
        with open(file_path, encoding="utf-8") as f:
            data = yaml.safe_load(f)

        if not isinstance(data, dict):
            raise PredictorValidationError("YAML file must contain a dictionary")

        return validate_predictor_dict(data)

    except yaml.YAMLError as e:
        raise PredictorValidationError(f"Invalid YAML syntax: {e}") from e
    except FileNotFoundError as e:
        raise PredictorValidationError(f"Predictor file not found: {file_path}") from e
    except Exception as e:
        raise PredictorValidationError(
            f"Failed to load predictor specification: {e}"
        ) from e


def save_predictor_to_yaml(spec: PredictorSpec, file_path: str | Path) -> None:
    """Save predictor specification to YAML file.

    Args:
        spec: PredictorSpec to save
        file_path: Path where to save the YAML file
    """
    data = {
        "name": spec.name,
        "model_id": spec.model_id,
    }

    if spec.model_version is not None:
        data["model_version"] = spec.model_version

    if spec.outputs is not None:
        data["outputs"] = spec.outputs

    with open(file_path, "w", encoding="utf-8") as f:
        yaml.dump(data, f, default_flow_style=False, sort_keys=False)
