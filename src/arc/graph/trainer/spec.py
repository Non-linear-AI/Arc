"""Trainer specification for Arc-Graph."""

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
class OptimizerConfig:
    """Configuration for optimizer."""

    type: str  # Direct PyTorch optimizer name with pytorch prefix
    lr: float = 0.001
    params: dict[str, Any] | None = None


@dataclass
class LossConfig:
    """Configuration for loss function."""

    type: str  # Direct PyTorch loss name with pytorch prefix
    inputs: dict[str, str] | None = None  # Map loss inputs to model outputs/targets
    params: dict[str, Any] | None = None


@dataclass
class TrainingConfig:
    """Configuration for model training."""

    # Core training parameters
    epochs: int = 10
    batch_size: int = 32
    validation_split: float = 0.2
    shuffle: bool = True

    # Hardware and performance
    device: str = "auto"  # "auto", "cpu", "cuda", "mps"
    num_workers: int = 0
    pin_memory: bool = False

    # Checkpointing and saving
    checkpoint_every: int = 5  # epochs
    save_best_only: bool = True
    save_dir: str | None = None

    # Early stopping
    early_stopping_patience: int | None = None
    early_stopping_min_delta: float = 0.001
    early_stopping_monitor: str = "val_loss"
    early_stopping_mode: str = "min"  # "min" or "max"

    # Logging and monitoring
    log_every: int = 10  # batches
    verbose: bool = True

    # Advanced training options
    gradient_clip_val: float | None = None
    gradient_clip_norm: float | None = None
    accumulate_grad_batches: int = 1

    # Reproducibility
    seed: int | None = None


@dataclass
class TrainerSpec:
    """Complete trainer specification."""

    model_ref: str  # Reference to model ID (e.g., "diabetes-logistic-v1")
    optimizer: OptimizerConfig

    # Flattened config properties for direct access
    epochs: int = 10
    batch_size: int = 32
    learning_rate: float = 0.001
    validation_split: float = 0.2
    early_stopping_patience: int | None = None
    device: str = "auto"

    @classmethod
    def from_yaml(cls, yaml_str: str) -> TrainerSpec:
        """Parse TrainerSpec from YAML string.

        Args:
            yaml_str: YAML string containing trainer specification

        Returns:
            TrainerSpec: Parsed and validated trainer specification

        Raises:
            ValueError: If YAML is invalid or doesn't contain valid trainer spec
        """
        from arc.graph.trainer.validator import validate_trainer_dict

        data = yaml.safe_load(yaml_str)
        if not isinstance(data, dict):
            raise ValueError("Top-level YAML must be a mapping")

        # Validate the trainer structure
        validate_trainer_dict(data)

        # Parse model reference (required)
        model_ref = data.get("model_ref")
        if not model_ref:
            raise ValueError("trainer.model_ref is required")

        # Parse optimizer
        optimizer_data = data["optimizer"]
        optimizer = OptimizerConfig(
            type=optimizer_data["type"],
            lr=optimizer_data.get("lr", 0.001),
            params=optimizer_data.get("params"),
        )

        # Parse config if present
        config_data = data.get("config", {})

        return cls(
            model_ref=str(model_ref),
            optimizer=optimizer,
            epochs=config_data.get("epochs", 10),
            batch_size=config_data.get("batch_size", 32),
            learning_rate=config_data.get("learning_rate", 0.001),
            validation_split=config_data.get("validation_split", 0.2),
            early_stopping_patience=config_data.get("early_stopping_patience"),
            device=config_data.get("device", "auto"),
        )

    @classmethod
    def from_yaml_file(cls, path: str) -> TrainerSpec:
        """Parse TrainerSpec from YAML file.

        Args:
            path: Path to YAML file containing trainer specification

        Returns:
            TrainerSpec: Parsed and validated trainer specification

        Raises:
            FileNotFoundError: If file doesn't exist
            ValueError: If YAML is invalid or doesn't contain valid trainer spec
        """
        with open(path, encoding="utf-8") as f:
            return cls.from_yaml(f.read())

    def to_yaml(self) -> str:
        """Convert TrainerSpec to YAML string.

        Returns:
            YAML string representation of the trainer specification
        """
        data = {
            "model_ref": self.model_ref,
            "optimizer": asdict(self.optimizer),
            "config": {
                "epochs": self.epochs,
                "batch_size": self.batch_size,
                "learning_rate": self.learning_rate,
                "validation_split": self.validation_split,
                "device": self.device,
            },
        }

        # Only include early_stopping_patience if it's set
        if self.early_stopping_patience is not None:
            data["config"]["early_stopping_patience"] = self.early_stopping_patience

        return yaml.dump(data, default_flow_style=False, sort_keys=False)

    def to_yaml_file(self, path: str) -> None:
        """Save TrainerSpec to YAML file.

        Args:
            path: Path to save the YAML file
        """
        with open(path, "w", encoding="utf-8") as f:
            f.write(self.to_yaml())

    def get_training_config(self) -> TrainingConfig:
        """Get training configuration, creating default if not specified.

        Returns:
            TrainingConfig instance
        """
        return TrainingConfig(
            epochs=self.epochs,
            batch_size=self.batch_size,
            validation_split=self.validation_split,
            device=self.device,
            early_stopping_patience=self.early_stopping_patience,
        )
