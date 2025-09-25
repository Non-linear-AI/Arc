"""Arc-Graph parsing and validation package."""

# New separated architecture imports
from arc.graph.features import (
    CORE_PROCESSORS,
    FeatureSpec,
    FeaturesValidationError,
    ProcessorConfig,
    get_processor_class,
    validate_features_dict,
)
from arc.graph.model import (
    CORE_LAYERS,
    GraphNode,
    ModelInput,
    ModelSpec,
    ModelValidationError,
    get_layer_class,
    validate_model_dict,
)
from arc.graph.predictor import (
    PredictorSpec,
    PredictorValidationError,
    validate_predictor_dict,
)
from arc.graph.trainer import (
    CORE_LOSSES,
    CORE_OPTIMIZERS,
    LossConfig,
    OptimizerConfig,
    TrainerSpec,
    TrainerValidationError,
    TrainingConfig,
    get_loss_class,
    get_optimizer_class,
    validate_trainer_dict,
)

__all__ = [
    # Model architecture
    "ModelSpec",
    "ModelInput",
    "GraphNode",
    "validate_model_dict",
    "ModelValidationError",
    "get_layer_class",
    "CORE_LAYERS",
    # Trainer architecture
    "TrainerSpec",
    "OptimizerConfig",
    "LossConfig",
    "TrainingConfig",
    "validate_trainer_dict",
    "TrainerValidationError",
    "get_optimizer_class",
    "get_loss_class",
    "CORE_OPTIMIZERS",
    "CORE_LOSSES",
    # Features architecture
    "FeatureSpec",
    "ProcessorConfig",
    "validate_features_dict",
    "FeaturesValidationError",
    "get_processor_class",
    "CORE_PROCESSORS",
    # Predictor architecture
    "PredictorSpec",
    "validate_predictor_dict",
    "PredictorValidationError",
]
