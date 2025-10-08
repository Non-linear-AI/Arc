"""Model evaluation engine for Arc-Graph models."""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from datetime import datetime
from typing import Any

import torch
import torch.nn as nn

from arc.database.services.ml_data_service import MLDataService
from arc.database.services.trainer_service import TrainerService
from arc.graph import EvaluatorSpec, ModelSpec, TrainerSpec
from arc.ml.artifacts import ModelArtifactManager
from arc.ml.builder import ModelBuilder
from arc.ml.metrics import Metric, MetricsTracker, create_metrics_for_task

logger = logging.getLogger(__name__)


class EvaluationError(Exception):
    """Raised when evaluation fails."""


@dataclass
class EvaluationResult:
    """Result from model evaluation."""

    evaluator_name: str
    trainer_ref: str
    model_ref: str
    version: int
    dataset: str
    num_samples: int

    # Core metrics
    metrics: dict[str, float]  # {metric_name: value}

    # Metadata
    evaluation_time: float  # seconds
    timestamp: str

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "evaluator_name": self.evaluator_name,
            "trainer_ref": self.trainer_ref,
            "model_ref": self.model_ref,
            "version": self.version,
            "dataset": self.dataset,
            "num_samples": self.num_samples,
            "metrics": self.metrics,
            "evaluation_time": self.evaluation_time,
            "timestamp": self.timestamp,
        }


class ArcEvaluator:
    """Model evaluation engine for Arc-Graph models.

    Evaluates trained models on test datasets and computes metrics.
    """

    def __init__(
        self,
        model: nn.Module,
        model_spec: ModelSpec,
        trainer_spec: TrainerSpec,
        evaluator_spec: EvaluatorSpec,
        artifact_version: int,
        device: str = "cpu",
    ):
        """Initialize evaluator.

        Args:
            model: Trained PyTorch model
            model_spec: Model specification
            trainer_spec: Trainer specification
            evaluator_spec: Evaluator specification
            artifact_version: Which training run/version
            device: Device for inference
        """
        self.model = model
        self.model_spec = model_spec
        self.trainer_spec = trainer_spec
        self.evaluator_spec = evaluator_spec
        self.artifact_version = artifact_version
        self.device = torch.device(device)

        # Move model to device and set to evaluation mode
        self.model.to(self.device)
        self.model.eval()

        logger.info(f"ArcEvaluator initialized for trainer {trainer_spec.model_ref}")
        logger.info(f"Evaluation dataset: {evaluator_spec.dataset}")

    def evaluate(
        self,
        ml_data_service: MLDataService,
    ) -> EvaluationResult:
        """Run model evaluation on test dataset.

        Workflow:
        1. Load test data from dataset table
        2. Run predictions
        3. Compute metrics
        4. Return results

        Args:
            ml_data_service: Service for accessing ML data

        Returns:
            EvaluationResult with computed metrics

        Raises:
            EvaluationError: If evaluation fails
        """
        start_time = time.time()

        try:
            # 1. Load test dataset (features + targets)
            logger.info(f"Loading evaluation dataset: {self.evaluator_spec.dataset}")
            features, targets = self._load_test_data(ml_data_service)

            # 2. Run predictions
            logger.info(f"Running predictions on {len(targets)} samples")
            predictions = self._run_predictions(features)

            # 3. Compute metrics
            logger.info("Computing evaluation metrics")
            metrics_dict = self._compute_metrics(predictions, targets)

            evaluation_time = time.time() - start_time

            # 4. Create result
            result = EvaluationResult(
                evaluator_name=self.evaluator_spec.name,
                trainer_ref=self.evaluator_spec.trainer_ref,
                model_ref=self.trainer_spec.model_ref,
                version=self.artifact_version,
                dataset=self.evaluator_spec.dataset,
                num_samples=len(targets),
                metrics=metrics_dict,
                evaluation_time=evaluation_time,
                timestamp=datetime.now().isoformat(),
            )

            logger.info(f"Evaluation completed in {evaluation_time:.2f}s")
            for metric_name, value in metrics_dict.items():
                logger.info(f"  {metric_name}: {value:.4f}")

            return result

        except Exception as e:
            logger.error(f"Evaluation failed: {e}")
            raise EvaluationError(f"Evaluation failed: {e}") from e

    def _load_test_data(
        self, ml_data_service: MLDataService
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Load test dataset features and targets.

        Args:
            ml_data_service: Service for accessing ML data

        Returns:
            Tuple of (features, targets) tensors

        Raises:
            EvaluationError: If data loading fails
        """
        try:
            # Get feature columns from model spec
            if not self.model_spec.inputs:
                raise EvaluationError("No inputs defined in model spec")

            # Get the first input (assuming single input for simplicity)
            first_input = next(iter(self.model_spec.inputs.values()))
            if not hasattr(first_input, "columns") or not first_input.columns:
                raise EvaluationError(
                    "No feature columns found in model input specification"
                )

            feature_columns = first_input.columns

            # Get target column from evaluator spec
            target_column = self.evaluator_spec.target_column

            logger.debug(f"Feature columns: {feature_columns}")
            logger.debug(f"Target column: {target_column}")

            # Load features and targets as tensors
            features, targets = ml_data_service.get_features_as_tensors(
                dataset_name=self.evaluator_spec.dataset,
                feature_columns=feature_columns,
                target_columns=[target_column],
            )

            # targets is returned as 2D tensor, squeeze to 1D if single target
            if targets.dim() == 2 and targets.size(1) == 1:
                targets = targets.squeeze(1)

            return features, targets

        except Exception as e:
            logger.error(f"Failed to load test data: {e}")
            raise EvaluationError(f"Failed to load test data: {e}") from e

    def _run_predictions(self, features: torch.Tensor) -> torch.Tensor:
        """Run predictions on features.

        Args:
            features: Input features tensor

        Returns:
            Predictions tensor

        Raises:
            EvaluationError: If prediction fails
        """
        try:
            features = features.to(self.device)

            with torch.no_grad():
                output = self.model(features)

            # Handle different output types
            if isinstance(output, dict):
                # Multi-output model: use the first output
                # (for MVP, we assume single output for evaluation)
                output_key = list(self.model_spec.outputs.keys())[0]
                predictions = output[output_key]
            else:
                predictions = output

            # Move predictions back to CPU
            predictions = predictions.cpu()

            return predictions

        except Exception as e:
            logger.error(f"Prediction failed: {e}")
            raise EvaluationError(f"Prediction failed: {e}") from e

    def _compute_metrics(
        self, predictions: torch.Tensor, targets: torch.Tensor
    ) -> dict[str, float]:
        """Compute specified metrics, or infer from model loss function.

        Args:
            predictions: Model predictions
            targets: Ground truth targets

        Returns:
            Dictionary of metric names to values

        Raises:
            EvaluationError: If metric computation fails
        """
        try:
            # Determine task type from model loss
            task_type = self._infer_task_type()

            # Use specified metrics or defaults based on task type
            if self.evaluator_spec.metrics:
                metric_names = self.evaluator_spec.metrics
                tracker = MetricsTracker()
                for metric_name in metric_names:
                    metric = self._create_metric(metric_name, task_type)
                    tracker.add_metric(metric)
            else:
                # Use default metrics for task type
                tracker = create_metrics_for_task(task_type)

            # Compute all metrics
            results = tracker.compute_metrics(predictions, targets)
            return {name: result.value for name, result in results.items()}

        except Exception as e:
            logger.error(f"Metric computation failed: {e}")
            raise EvaluationError(f"Metric computation failed: {e}") from e

    def _infer_task_type(self) -> str:
        """Infer task type from model's loss function.

        Returns:
            Task type: 'classification' or 'regression'
        """
        if not self.model_spec.loss:
            # Default to classification if no loss specified
            return "classification"

        loss_type = self.model_spec.loss.get("type", "").lower()

        # Check for classification losses
        classification_losses = [
            "cross_entropy",
            "bce",
            "binary_cross_entropy",
            "nll",
            "negative_log_likelihood",
        ]
        if any(cls_loss in loss_type for cls_loss in classification_losses):
            return "classification"

        # Check for regression losses
        regression_losses = ["mse", "mae", "l1", "l2", "smooth_l1"]
        if any(reg_loss in loss_type for reg_loss in regression_losses):
            return "regression"

        # Default to classification
        return "classification"

    def _create_metric(self, metric_name: str, _task_type: str) -> Metric:
        """Create a metric instance from name.

        Args:
            metric_name: Name of the metric
            task_type: Task type for context

        Returns:
            Metric instance

        Raises:
            EvaluationError: If metric name is unknown
        """
        from arc.ml.metrics import (
            AUC,
            Accuracy,
            F1Score,
            MeanAbsoluteError,
            MeanSquaredError,
            Precision,
            R2Score,
            Recall,
            RootMeanSquaredError,
        )

        metric_map = {
            "accuracy": Accuracy,
            "precision": Precision,
            "recall": Recall,
            "f1_score": F1Score,
            "f1": F1Score,
            "auc": AUC,
            "mse": MeanSquaredError,
            "mae": MeanAbsoluteError,
            "rmse": RootMeanSquaredError,
            "r2_score": R2Score,
            "r2": R2Score,
        }

        metric_name_lower = metric_name.lower()
        if metric_name_lower not in metric_map:
            raise EvaluationError(
                f"Unknown metric: {metric_name}. Available: {list(metric_map.keys())}"
            )

        return metric_map[metric_name_lower]()

    @classmethod
    def load_from_trainer(
        cls,
        artifact_manager: ModelArtifactManager,
        trainer_service: TrainerService,
        evaluator_spec: EvaluatorSpec,
        device: str = "cpu",
    ) -> ArcEvaluator:
        """Load evaluator from trainer reference.

        Args:
            artifact_manager: Manager for loading model artifacts
            trainer_service: Service for loading trainer specs
            evaluator_spec: Evaluator specification
            device: Device for inference

        Returns:
            Loaded evaluator ready for evaluation

        Raises:
            EvaluationError: If loading fails
        """
        try:
            logger.info(f"Loading evaluator for trainer: {evaluator_spec.trainer_ref}")

            # 1. Load trainer spec
            trainer = trainer_service.get_trainer_by_name(evaluator_spec.trainer_ref)
            if not trainer:
                raise EvaluationError(
                    f"Trainer '{evaluator_spec.trainer_ref}' not found"
                )

            trainer_spec = TrainerSpec.from_yaml(trainer.spec)
            model_ref = trainer_spec.model_ref

            logger.info(f"Trainer references model: {model_ref}")

            # 2. Load trained model artifact (latest version or specified)
            version = evaluator_spec.version  # None = latest
            state_dict, artifact = artifact_manager.load_model_state_dict(
                model_id=model_ref,
                version=version,
                device=device,
            )

            logger.info(f"Loaded model artifact version: {artifact.version}")

            # 3. Reconstruct model from spec
            if not artifact.model_spec:
                raise EvaluationError(
                    f"No model specification found in artifact for model {model_ref}"
                )

            # Convert model_spec dict to ModelSpec
            model_dict = artifact.model_spec
            if "model" in model_dict:
                model_dict = model_dict["model"]

            from arc.graph.model import GraphNode, ModelInput, validate_model_dict

            # Validate the model structure
            validate_model_dict(model_dict)

            # Parse inputs
            inputs = {}
            for input_name, input_spec in model_dict["inputs"].items():
                inputs[input_name] = ModelInput(
                    dtype=input_spec["dtype"],
                    shape=input_spec["shape"],
                    columns=input_spec.get("columns"),
                )

            # Parse graph nodes
            graph = []
            for node_data in model_dict["graph"]:
                graph.append(
                    GraphNode(
                        name=node_data["name"],
                        type=node_data["type"],
                        params=node_data.get("params", {}),
                        inputs=node_data.get("inputs", {}),
                    )
                )

            # Create ModelSpec
            model_spec = ModelSpec(
                inputs=inputs,
                graph=graph,
                outputs=model_dict["outputs"],
                loss=model_dict.get("loss"),
            )

            # 4. Build model from spec
            builder = ModelBuilder()
            model = builder.build_model(model_spec)

            # 5. Load trained weights
            model.load_state_dict(state_dict)

            logger.info("Model loaded successfully")

            return cls(
                model=model,
                model_spec=model_spec,
                trainer_spec=trainer_spec,
                evaluator_spec=evaluator_spec,
                artifact_version=artifact.version,
                device=device,
            )

        except Exception as e:
            logger.error(f"Failed to load evaluator: {e}")
            raise EvaluationError(f"Failed to load evaluator: {e}") from e
