"""Arc ML package for PyTorch model building and training."""

from .artifacts import ModelArtifact, ModelArtifactManager
from .builder import ArcModel, ModelBuilder
from .layers import get_layer_class
from .metrics import MetricsTracker, create_metrics_for_task
from .predictor import ArcPredictor, PredictionError
from .trainer import ArcTrainer, TrainingConfig, TrainingResult
from .training_service import TrainingJobConfig, TrainingService
from .utils import auto_detect_input_size, validate_tensor_shape

__all__ = [
    "ModelBuilder",
    "ArcModel",
    "get_layer_class",
    "auto_detect_input_size",
    "validate_tensor_shape",
    "ArcTrainer",
    "TrainingConfig",
    "TrainingResult",
    "TrainingService",
    "TrainingJobConfig",
    "MetricsTracker",
    "create_metrics_for_task",
    "ModelArtifact",
    "ModelArtifactManager",
    "ArcPredictor",
    "PredictionError",
]
