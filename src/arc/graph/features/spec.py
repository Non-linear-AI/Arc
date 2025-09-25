"""Features specification for Arc-Graph."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

try:
    import yaml
except ImportError as e:
    raise RuntimeError(
        "PyYAML is required for Arc-Graph. "
        "Install with 'uv add pyyaml' or 'pip install pyyaml'."
    ) from e


@dataclass
class ProcessorConfig:
    """Configuration for a feature processor."""

    name: str
    op: str  # Processor operation type
    train_only: bool = False
    inputs: dict[str, str] | None = None
    outputs: dict[str, str] | None = None
    params: dict[str, Any] | None = None


@dataclass
class FeatureSpec:
    """Complete features specification."""

    feature_columns: list[str]
    target_columns: list[str] | None = None
    processors: list[ProcessorConfig] | None = None

    @classmethod
    def from_yaml(cls, yaml_str: str) -> FeatureSpec:
        """Parse FeatureSpec from YAML string.

        Args:
            yaml_str: YAML string containing features specification

        Returns:
            FeatureSpec: Parsed and validated features specification

        Raises:
            ValueError: If YAML is invalid or doesn't contain valid features spec
        """
        from arc.graph.features.validator import validate_features_dict

        data = yaml.safe_load(yaml_str)
        if not isinstance(data, dict):
            raise ValueError("Top-level YAML must be a mapping")

        # Validate the features structure
        validate_features_dict(data)

        # Parse feature columns
        feature_columns = data["feature_columns"]

        # Parse target columns (optional)
        target_columns = data.get("target_columns")

        # Parse processors (optional)
        processors = None
        if "processors" in data and data["processors"]:
            processors = []
            for proc_data in data["processors"]:
                processors.append(
                    ProcessorConfig(
                        name=proc_data["name"],
                        op=proc_data["op"],
                        train_only=proc_data.get("train_only", False),
                        inputs=proc_data.get("inputs"),
                        outputs=proc_data.get("outputs"),
                        params=proc_data.get("params"),
                    )
                )

        return cls(
            feature_columns=feature_columns,
            target_columns=target_columns,
            processors=processors,
        )

    @classmethod
    def from_yaml_file(cls, path: str) -> FeatureSpec:
        """Parse FeatureSpec from YAML file.

        Args:
            path: Path to YAML file containing features specification

        Returns:
            FeatureSpec: Parsed and validated features specification

        Raises:
            FileNotFoundError: If file doesn't exist
            ValueError: If YAML is invalid or doesn't contain valid features spec
        """
        with open(path, encoding="utf-8") as f:
            return cls.from_yaml(f.read())

    def to_yaml(self) -> str:
        """Convert FeatureSpec to YAML string.

        Returns:
            YAML string representation of the features specification
        """
        return yaml.dump(asdict(self), default_flow_style=False)

    def to_yaml_file(self, path: str) -> None:
        """Save FeatureSpec to YAML file.

        Args:
            path: Path to save the YAML file
        """
        with open(path, "w", encoding="utf-8") as f:
            f.write(self.to_yaml())

    def get_feature_columns(self) -> list[str]:
        """Get list of feature column names.

        Returns:
            List of feature column names
        """
        return self.feature_columns.copy()

    def get_target_columns(self) -> list[str]:
        """Get list of target column names.

        Returns:
            List of target column names, empty list if none specified
        """
        return self.target_columns.copy() if self.target_columns else []

    def get_processor_names(self) -> list[str]:
        """Get list of processor names.

        Returns:
            List of processor names, empty list if no processors
        """
        return [proc.name for proc in self.processors] if self.processors else []

    def get_processor_types(self) -> dict[str, str]:
        """Get mapping of processor names to operation types.

        Returns:
            Dictionary mapping processor names to their operation types
        """
        if not self.processors:
            return {}
        return {proc.name: proc.op for proc in self.processors}
