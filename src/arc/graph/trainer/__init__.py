"""Trainer specification and components for Arc-Graph."""

from arc.graph.trainer.components import (
    CORE_LOSSES,
    CORE_OPTIMIZERS,
    get_loss_class,
    get_optimizer_class,
)
from arc.graph.trainer.spec import (
    LossConfig,
    OptimizerConfig,
    TrainerSpec,
    TrainingConfig,
)
from arc.graph.trainer.validator import TrainerValidationError, validate_trainer_dict


def load_trainer_from_yaml(file_path: str) -> TrainerSpec:
    """Load trainer specification from YAML file.

    Args:
        file_path: Path to YAML file

    Returns:
        Loaded and validated TrainerSpec
    """
    return TrainerSpec.from_yaml_file(file_path)


__all__ = [
    "OptimizerConfig",
    "LossConfig",
    "TrainingConfig",
    "TrainerSpec",
    "get_optimizer_class",
    "get_loss_class",
    "CORE_OPTIMIZERS",
    "CORE_LOSSES",
    "validate_trainer_dict",
    "TrainerValidationError",
    "load_trainer_from_yaml",
]
