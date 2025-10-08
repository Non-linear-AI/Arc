"""Arc ML package for PyTorch model building and training."""

from arc.ml.artifacts import ModelArtifact, ModelArtifactManager
from arc.ml.builder import ArcModel, ModelBuilder
from arc.ml.layers import get_layer_class
from arc.ml.metrics import MetricsTracker, create_metrics_for_task
from arc.ml.predictor import ArcPredictor, PredictionError
from arc.ml.tensorboard import TensorBoardManager
from arc.ml.trainer import ArcTrainer, TrainingConfig, TrainingResult
from arc.ml.training_service import TrainingJobConfig, TrainingService
from arc.ml.utils import auto_detect_input_size, validate_tensor_shape

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
    "TensorBoardManager",
]
