"""Arc-Graph parsing and validation package."""

# New separated architecture imports
from arc.graph.evaluator import (
    EvaluatorSpec,
    EvaluatorValidationError,
    validate_evaluator_dict,
)
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
    ArcGraphModel,
    GraphNode,
    ModelInput,
    ModelSpec,
    ModelValidationError,
    ModuleDefinition,
    build_model_from_file,
    build_model_from_spec,
    build_model_from_yaml,
    get_layer_class,
    validate_model_dict,
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
    "ModuleDefinition",
    "ArcGraphModel",
    "build_model_from_spec",
    "build_model_from_yaml",
    "build_model_from_file",
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
    # Evaluator architecture
    "EvaluatorSpec",
    "validate_evaluator_dict",
    "EvaluatorValidationError",
]
