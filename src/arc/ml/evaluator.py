"""Model evaluation engine for Arc-Graph models."""

from __future__ import annotations

import logging
import re
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
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
        output_table: str | None = None,
        tensorboard_log_dir: str | Path | None = None,
    ) -> EvaluationResult:
        """Run model evaluation on test dataset.

        Workflow:
        1. Load test data from dataset table
        2. Run predictions
        3. Compute metrics
        4. Log visualizations to TensorBoard (if enabled)
        5. Optionally save predictions to output table
        6. Return results

        Args:
            ml_data_service: Service for accessing ML data
            output_table: Optional table to save predictions (features + targets)
            tensorboard_log_dir: Optional directory for TensorBoard logging

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

            # 4. Log visualizations to TensorBoard if enabled
            if tensorboard_log_dir:
                self._log_to_tensorboard(
                    predictions, targets, metrics_dict, tensorboard_log_dir
                )

            # 5. Save predictions to output table if specified
            if output_table:
                logger.info(f"Saving predictions to table: {output_table}")
                self._save_predictions(
                    ml_data_service, features, targets, predictions, output_table
                )

            evaluation_time = time.time() - start_time

            # 6. Create result
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
                # Multi-output model: use output specified in evaluator spec
                output_keys = list(self.model_spec.outputs.keys())

                # Use output specified in evaluator spec if provided
                if self.evaluator_spec.output_name:
                    if self.evaluator_spec.output_name in output:
                        output_key = self.evaluator_spec.output_name
                        logger.info(f"Using specified output: '{output_key}'")
                    else:
                        raise EvaluationError(
                            f"Specified output '{self.evaluator_spec.output_name}' "
                            f"not found. Available outputs: {output_keys}"
                        )
                else:
                    # Auto-detect: prefer probability outputs
                    probability_output = None
                    for key in output_keys:
                        if key.lower() in [
                            "probabilities",
                            "probs",
                            "probability",
                            "prob",
                        ]:
                            probability_output = key
                            break

                    if probability_output:
                        output_key = probability_output
                        logger.info(f"Auto-detected probability output: '{output_key}'")
                    else:
                        # Fall back to first output
                        output_key = output_keys[0]
                        logger.warning(
                            f"No probability output found. Using first output: "
                            f"'{output_key}'. Available outputs: {output_keys}. "
                            "Specify output_name in evaluator spec to override."
                        )

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
            # Convert tensor values to Python floats for JSON serialization
            return {
                name: (
                    float(result.value)
                    if hasattr(result.value, "item")
                    else result.value
                )
                for name, result in results.items()
            }

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

    def _log_to_tensorboard(
        self,
        predictions: torch.Tensor,
        targets: torch.Tensor,
        metrics_dict: dict[str, float],
        log_dir: str | Path,
    ) -> None:
        """Log evaluation metrics and visualizations to TensorBoard.

        Uses direct SummaryWriter calls for simplicity and consistency
        with trainer logging.

        Args:
            predictions: Model predictions
            targets: Ground truth targets
            metrics_dict: Computed metrics
            log_dir: Directory for TensorBoard logs

        Raises:
            EvaluationError: If TensorBoard logging fails (non-fatal, logged as warning)
        """
        try:
            from torch.utils.tensorboard import SummaryWriter

            # Create log directory
            log_path = Path(log_dir)
            log_path.mkdir(parents=True, exist_ok=True)

            # Create TensorBoard writer
            writer = SummaryWriter(log_dir=str(log_path))

            # Log scalar metrics directly
            for metric_name, value in metrics_dict.items():
                writer.add_scalar(f"evaluation/{metric_name}", value, 1)

            # Determine task type for appropriate visualizations
            task_type = self._infer_task_type()
            logger.info(f"🔍 Task type detected: {task_type}")
            logger.debug(
                f"Predictions shape: {predictions.shape}, "
                f"Targets shape: {targets.shape}"
            )

            if task_type == "classification":
                logger.info(
                    "📊 Logging classification visualizations (including PR curve)..."
                )
                # Log classification visualizations
                self._log_classification_visualizations(
                    writer, predictions, targets, step=1
                )
            elif task_type == "regression":
                logger.warning("📊 Logging regression visualizations...")
                # Log regression visualizations
                self._log_regression_visualizations(
                    writer, predictions, targets, step=1
                )

            # Close writer
            writer.flush()
            writer.close()

        except Exception as e:
            # Don't fail the evaluation, just log a warning with full traceback
            import traceback

            logger.error(
                f"❌ Failed to log to TensorBoard: {e}\n{traceback.format_exc()}"
            )

    def _log_classification_visualizations(
        self,
        writer: Any,  # SummaryWriter
        predictions: torch.Tensor,
        targets: torch.Tensor,
        step: int = 1,
    ) -> None:
        """Log classification-specific visualizations using direct SummaryWriter calls.

        Args:
            writer: TensorBoard SummaryWriter instance
            predictions: Model predictions (logits or probabilities)
            targets: Ground truth labels
            step: Global step for logging
        """
        try:
            import numpy as np

            # Determine if binary or multi-class
            num_classes = predictions.shape[-1] if len(predictions.shape) > 1 else 2

            # Binary classification: single sigmoid output or 2-class softmax
            if num_classes <= 2 or len(predictions.shape) == 1:
                # Binary classification
                # Get probabilities for positive class (as tensors for add_pr_curve)
                if len(predictions.shape) > 1 and predictions.shape[-1] == 2:
                    # If predictions are [N, 2], take second column (softmax output)
                    y_scores = predictions[:, 1]
                elif len(predictions.shape) > 1 and predictions.shape[-1] == 1:
                    # If predictions are [N, 1], squeeze to [N] (sigmoid output)
                    y_scores = predictions.squeeze(-1)
                else:
                    # If predictions are [N], treat as scores
                    y_scores = predictions

                # Ensure targets are binary (0/1) as integer tensor on CPU
                y_true = targets.flatten().int().cpu()

                # Ensure scores are float tensor on CPU
                y_scores = y_scores.float().cpu()

                # Validate scores are in [0, 1] range
                if y_scores.min() < -0.01 or y_scores.max() > 1.01:
                    logger.warning(
                        f"⚠️  Warning: Predictions outside [0, 1] range: "
                        f"[{y_scores.min():.4f}, {y_scores.max():.4f}]. "
                        "Expected probabilities. Check model spec outputs."
                    )

                # Validate we have both classes
                unique_labels = torch.unique(y_true)

                # Calculate class distribution
                n_positive = (y_true == 1).sum().item()
                n_negative = (y_true == 0).sum().item()

                # Log basic statistics
                logger.info(
                    f"Evaluation data: {n_positive} positive, "
                    f"{n_negative} negative samples"
                )

                if len(unique_labels) >= 2:
                    # Log PR curve directly - this appears in PR CURVES tab
                    try:
                        writer.add_pr_curve(
                            "evaluation/pr_curve",
                            labels=y_true,
                            predictions=y_scores,
                            global_step=step,
                            num_thresholds=1000,  # Use 1000 thresholds for smooth curve
                        )
                        writer.flush()  # Force write immediately
                        logger.info("✓ Successfully logged PR curve to TensorBoard")
                    except Exception as e:
                        logger.error(f"✗ Failed to log PR curve: {e}")
                        import traceback

                        logger.error(traceback.format_exc())
                else:
                    logger.warning(
                        "Skipping PR curve: only one class present in targets: "
                        f"{unique_labels.tolist()}"
                    )

                # Note: ROC curve not logged (TensorBoard doesn't support it natively)

                # Log confusion matrix
                try:
                    y_pred = (y_scores > 0.5).int()
                    self._log_confusion_matrix(
                        writer, y_true, y_pred, ["Class 0", "Class 1"], step
                    )
                except Exception as e:
                    logger.error(f"Failed to log confusion matrix: {e}")

            else:
                # Multi-class classification
                predictions_np = predictions.cpu().numpy()
                targets_np = targets.cpu().numpy()

                # Get predicted classes
                y_pred = np.argmax(predictions_np, axis=1)
                y_true = targets_np.flatten().astype(int)

                # Generate class names
                unique_classes = sorted(set(y_true))
                class_names = [f"Class {i}" for i in unique_classes]

                # Log confusion matrix
                self._log_confusion_matrix(
                    writer,
                    torch.from_numpy(y_true),
                    torch.from_numpy(y_pred),
                    class_names,
                    step,
                )

                # Log per-class metrics
                self._log_class_metrics(writer, y_true, y_pred, class_names, step)

        except Exception as e:
            import traceback

            logger.error(
                f"Failed to log classification visualizations: {e}\n"
                f"{traceback.format_exc()}"
            )

    def _log_confusion_matrix(
        self,
        writer: Any,
        y_true: torch.Tensor,
        y_pred: torch.Tensor,
        class_names: list[str],
        step: int,
    ) -> None:
        """Compute and log confusion matrix as text."""
        try:
            import numpy as np

            # Convert to numpy
            y_true_np = y_true.cpu().numpy().flatten()
            y_pred_np = y_pred.cpu().numpy().flatten()

            # Compute confusion matrix
            classes = np.unique(np.concatenate([y_true_np, y_pred_np]))
            n_classes = len(classes)
            cm = np.zeros((n_classes, n_classes), dtype=int)

            for i, true_class in enumerate(classes):
                for j, pred_class in enumerate(classes):
                    cm[i, j] = np.sum(
                        (y_true_np == true_class) & (y_pred_np == pred_class)
                    )

            # Format as text table
            text = "Confusion Matrix:\n\n"
            text += "Predicted →\n"
            header = " | ".join(f"{name:^10}" for name in class_names[:n_classes])
            text += f"True ↓ | {header}\n"
            text += "-" * (12 + 13 * n_classes) + "\n"

            for i, true_name in enumerate(class_names[:n_classes]):
                row = f"{true_name:^10} | "
                row += " | ".join(f"{cm[i, j]:^10d}" for j in range(n_classes))
                text += row + "\n"

            # Log as text
            writer.add_text("evaluation/confusion_matrix", text, step)
            logger.debug("Logged confusion matrix")

        except Exception as e:
            logger.warning(f"Failed to log confusion matrix: {e}")

    def _log_class_metrics(
        self,
        writer: Any,
        y_true: Any,  # np.ndarray
        y_pred: Any,  # np.ndarray
        class_names: list[str],
        step: int,
    ) -> None:
        """Compute and log per-class metrics."""
        try:
            import numpy as np

            classes = np.unique(np.concatenate([y_true, y_pred]))

            for i, class_id in enumerate(classes):
                if i < len(class_names):
                    class_name = class_names[i]
                else:
                    class_name = f"Class_{class_id}"

                # Compute metrics
                tp = np.sum((y_true == class_id) & (y_pred == class_id))
                fp = np.sum((y_true != class_id) & (y_pred == class_id))
                fn = np.sum((y_true == class_id) & (y_pred != class_id))

                precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
                recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
                f1 = (
                    2 * precision * recall / (precision + recall)
                    if (precision + recall) > 0
                    else 0.0
                )

                # Log metrics
                writer.add_scalar(
                    f"evaluation/class_{class_name}/precision", precision, step
                )
                writer.add_scalar(f"evaluation/class_{class_name}/recall", recall, step)
                writer.add_scalar(f"evaluation/class_{class_name}/f1_score", f1, step)

            logger.debug("Logged per-class metrics")

        except Exception as e:
            logger.warning(f"Failed to log class metrics: {e}")

    def _log_regression_visualizations(
        self,
        writer: Any,
        predictions: torch.Tensor,
        targets: torch.Tensor,
        step: int,
    ) -> None:
        """Log regression-specific visualizations."""
        try:
            # Log histograms
            writer.add_histogram("evaluation/true_values", targets, step)
            writer.add_histogram("evaluation/predictions", predictions, step)

            # Compute and log residuals
            residuals = predictions - targets
            writer.add_histogram("evaluation/residuals", residuals, step)

            # Log summary statistics
            writer.add_scalar(
                "evaluation/mean_absolute_error",
                torch.abs(residuals).mean().item(),
                step,
            )
            writer.add_scalar(
                "evaluation/mean_squared_error",
                (residuals**2).mean().item(),
                step,
            )

            logger.debug("Logged regression visualizations")

        except Exception as e:
            logger.warning(f"Failed to log regression visualizations: {e}")

    def _save_predictions(
        self,
        ml_data_service: MLDataService,
        features: torch.Tensor,
        targets: torch.Tensor,
        predictions: torch.Tensor,
        output_table: str,
    ) -> None:
        """Save predictions along with features and targets to output table.

        Args:
            ml_data_service: Service for saving data
            features: Feature tensor
            targets: Target tensor
            predictions: Prediction tensor
            output_table: Name of output table

        Raises:
            EvaluationError: If saving fails
        """
        try:
            import pandas as pd

            # Get feature column names from model spec
            first_input = next(iter(self.model_spec.inputs.values()))
            feature_columns = first_input.columns

            # Convert tensors to numpy arrays
            features_np = features.cpu().numpy()
            targets_np = targets.cpu().numpy().flatten()
            predictions_np = predictions.cpu().numpy().flatten()

            # Build dataframe
            data_dict = {}
            for idx, col_name in enumerate(feature_columns):
                data_dict[col_name] = features_np[:, idx]

            # Add target and prediction columns
            target_col = self.evaluator_spec.target_column
            data_dict[target_col] = targets_np
            data_dict["prediction"] = predictions_np

            df = pd.DataFrame(data_dict)  # noqa: F841 - used by DuckDB SQL

            # Drop existing table and create new one
            db_manager = ml_data_service.db_manager
            db_manager.user_execute(f'DROP TABLE IF EXISTS "{output_table}"')

            # Use DuckDB's SQL with CREATE TABLE AS directly from INSERT
            # First create the table structure from the dataframe
            conn = db_manager._get_user_db()._connection
            conn.execute(f'CREATE TABLE "{output_table}" AS SELECT * FROM df')

            logger.info(
                f"Saved {len(predictions_np)} predictions to table '{output_table}'"
            )

        except Exception as e:
            logger.error(f"Failed to save predictions: {e}")
            raise EvaluationError(f"Failed to save predictions: {e}") from e

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
            trainer = trainer_service.get_trainer_by_id(evaluator_spec.trainer_ref)
            if not trainer:
                raise EvaluationError(
                    f"Trainer '{evaluator_spec.trainer_ref}' not found"
                )

            trainer_spec = TrainerSpec.from_yaml(trainer.spec)
            model_ref = trainer_spec.model_ref

            logger.info(f"Trainer references model: {model_ref}")

            # 2. Find which model version to load based on training_runs
            # Artifacts are saved per trainer_id for isolation
            from arc.database.services import TrainingTrackingService

            tracking_service = TrainingTrackingService(trainer_service.db_manager)
            runs = tracking_service.list_runs(trainer_id=evaluator_spec.trainer_ref)

            if not runs:
                trainer_ref = evaluator_spec.trainer_ref
                raise EvaluationError(
                    f"Trainer '{trainer_ref}' has never been trained. "
                    f"Train first: /ml train --trainer-id {trainer_ref}"
                )

            # Get the most recent successful run
            from arc.database.models.training import TrainingStatus

            completed_runs = [
                r
                for r in runs
                if r.status == TrainingStatus.COMPLETED and r.artifact_path
            ]
            if not completed_runs:
                trainer_ref = evaluator_spec.trainer_ref
                raise EvaluationError(
                    f"Trainer '{trainer_ref}' has no successful training runs. "
                    f"Train first: /ml train --trainer-id {trainer_ref}"
                )

            # Use the most recent completed run (list_runs sorts by created_at DESC)
            latest_run = completed_runs[0]

            # Parse version from artifact_path
            # Format: artifacts/trainer-id/v{version}/
            version_match = re.search(r"/v(\d+)/?$", latest_run.artifact_path)
            if not version_match:
                # Try to load latest version
                version = evaluator_spec.version
            else:
                version = int(version_match.group(1))
                logger.info(f"Using model v{version} from run {latest_run.run_id}")

            # Load the model artifact (keyed by trainer_id, not model_ref)
            try:
                state_dict, artifact = artifact_manager.load_model_state_dict(
                    model_id=evaluator_spec.trainer_ref,
                    version=version,
                    device=device,
                )
            except FileNotFoundError as e:
                raise EvaluationError(
                    f"Artifacts not found for trainer '{evaluator_spec.trainer_ref}'. "
                    f"The training run completed but artifacts are missing. "
                    f"Artifact path: {latest_run.artifact_path}"
                ) from e

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
