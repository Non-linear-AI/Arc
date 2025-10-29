"""ML visualization utilities for TensorBoard and other visualization backends.

This module provides reusable components for visualizing ML metrics including:
- Classification metrics (PR curves, ROC curves, confusion matrices)
- Regression metrics (scatter plots, residual plots)
- Distribution visualizations

These components can be used in both training and evaluation runs.
Uses TensorBoard's native APIs where available to minimize dependencies.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import numpy as np
import torch

if TYPE_CHECKING:
    from torch.utils.tensorboard import SummaryWriter

logger = logging.getLogger(__name__)


class TensorBoardVisualizer:
    """Visualizer for logging ML metrics to TensorBoard.

    Provides methods for creating visualizations that can be logged
    to TensorBoard during training or evaluation runs.

    Args:
        writer: TensorBoard SummaryWriter instance
        enabled: Whether visualization is enabled (default: True)

    Example:
        >>> from torch.utils.tensorboard import SummaryWriter
        >>> writer = SummaryWriter(log_dir="logs/run_123")
        >>> viz = TensorBoardVisualizer(writer)
        >>> viz.log_confusion_matrix(y_true, y_pred, step=1)
    """

    def __init__(self, writer: SummaryWriter | None = None, enabled: bool = True):
        self.writer = writer
        self.enabled = enabled and writer is not None

    def log_pr_curve(
        self,
        y_true: np.ndarray | torch.Tensor,
        y_scores: np.ndarray | torch.Tensor,
        tag: str = "pr_curve",
        step: int = 0,
    ) -> None:
        """Log Precision-Recall curve to TensorBoard.

        Uses TensorBoard's native PR curve functionality which creates
        an interactive visualization in the PR CURVES tab.

        Args:
            y_true: True binary labels (0 or 1)
            y_scores: Predicted probabilities for positive class
            tag: Tag name for the curve in TensorBoard
            step: Global step value for logging
        """
        if not self.enabled or self.writer is None:
            return

        try:
            # Convert to tensors if needed (required for add_pr_curve)
            if isinstance(y_true, np.ndarray):
                y_true = torch.from_numpy(y_true).int()
            if isinstance(y_scores, np.ndarray):
                y_scores = torch.from_numpy(y_scores).float()

            # Flatten and ensure correct types
            y_true = y_true.flatten().int()
            y_scores = y_scores.flatten().float()

            # Validate data
            if len(torch.unique(y_true)) < 2:
                logger.warning("Cannot compute PR curve: only one class present")
                return

            # Use TensorBoard's native PR curve (shows in PR CURVES tab)
            self.writer.add_pr_curve(
                tag,
                labels=y_true,
                predictions=y_scores,
                global_step=step,
                num_thresholds=127,  # TensorBoard default
            )

            logger.debug(f"TensorBoard: Logged PR curve '{tag}' at step {step}")

        except Exception as e:
            logger.warning(f"Failed to log PR curve: {e}")

    def log_roc_curve(
        self,
        y_true: np.ndarray | torch.Tensor,
        y_scores: np.ndarray | torch.Tensor,
        tag: str = "roc_curve",
        step: int = 0,
    ) -> None:
        """Log ROC AUC score to TensorBoard.

        Computes and logs the ROC AUC as a scalar metric.
        Note: TensorBoard doesn't have a native ROC curve widget,
        so we log the AUC value. The PR curve (which is related)
        is available in the PR CURVES tab.

        Args:
            y_true: True binary labels (0 or 1)
            y_scores: Predicted probabilities for positive class
            tag: Tag name for the metric in TensorBoard
            step: Global step value for logging
        """
        if not self.enabled or self.writer is None:
            return

        try:
            # Convert to numpy if needed
            if isinstance(y_true, torch.Tensor):
                y_true = y_true.cpu().numpy()
            if isinstance(y_scores, torch.Tensor):
                y_scores = y_scores.cpu().numpy()

            # Flatten arrays
            y_true = y_true.flatten()
            y_scores = y_scores.flatten()

            # Sort by scores ascending for ROC
            sorted_indices = np.argsort(y_scores)
            y_true_sorted = y_true[sorted_indices]

            # Calculate TPR and FPR at each threshold
            n_pos = np.sum(y_true == 1)
            n_neg = np.sum(y_true == 0)

            if n_pos == 0 or n_neg == 0:
                logger.warning("Cannot compute ROC AUC: only one class present")
                return

            # Compute cumulative TPs and FPs (traversing from low to high scores)
            tp = np.cumsum(y_true_sorted[::-1])
            fp = np.cumsum(1 - y_true_sorted[::-1])

            # Compute TPR and FPR
            tpr = tp / n_pos
            fpr = fp / n_neg

            # Add starting point (0, 0) and ending point (1, 1)
            fpr = np.concatenate([[0.0], fpr, [1.0]])
            tpr = np.concatenate([[0.0], tpr, [1.0]])

            # Compute AUC using trapezoidal rule
            auc = np.trapz(tpr, fpr)

            # Log AUC as scalar
            self.writer.add_scalar(f"{tag}/auc", auc, step)

            logger.debug(
                f"TensorBoard: Logged ROC AUC={auc:.3f} for '{tag}' at step {step}"
            )

        except Exception as e:
            logger.warning(f"Failed to log ROC AUC: {e}")

    def log_confusion_matrix(
        self,
        y_true: np.ndarray | torch.Tensor,
        y_pred: np.ndarray | torch.Tensor,
        class_names: list[str] | None = None,
        tag: str = "confusion_matrix",
        step: int = 0,
        normalize: bool = False,
    ) -> None:
        """Log confusion matrix as text to TensorBoard (lightweight version).

        Args:
            y_true: True class labels
            y_pred: Predicted class labels
            class_names: Optional class names for axis labels
            tag: Tag name for the matrix in TensorBoard
            step: Global step value for logging
            normalize: Whether to normalize the matrix (default: False)
        """
        if not self.enabled or self.writer is None:
            return

        try:
            # Convert to numpy if needed
            if isinstance(y_true, torch.Tensor):
                y_true = y_true.cpu().numpy()
            if isinstance(y_pred, torch.Tensor):
                y_pred = y_pred.cpu().numpy()

            # Flatten arrays
            y_true = y_true.flatten()
            y_pred = y_pred.flatten()

            # Compute confusion matrix manually
            classes = np.unique(np.concatenate([y_true, y_pred]))
            n_classes = len(classes)
            cm = np.zeros((n_classes, n_classes), dtype=int)

            for i, true_class in enumerate(classes):
                for j, pred_class in enumerate(classes):
                    cm[i, j] = np.sum((y_true == true_class) & (y_pred == pred_class))

            if normalize:
                cm = cm.astype("float") / cm.sum(axis=1)[:, np.newaxis]

            # Format as text table
            if class_names is None:
                class_names = [f"Class {int(i)}" for i in classes]

            # Create text representation
            text = "Confusion Matrix:\n\n"
            text += "Predicted →\n"
            header = " | ".join(f"{name:^10}" for name in class_names)
            text += f"True ↓ | {header}\n"
            text += "-" * (12 + 13 * n_classes) + "\n"

            for i, true_name in enumerate(class_names):
                row = f"{true_name:^10} | "
                if normalize:
                    # Format as percentage with 1 decimal place
                    row += " | ".join(f"{cm[i, j]:^10.1%}" for j in range(n_classes))
                else:
                    # Format as integer
                    row += " | ".join(f"{cm[i, j]:^10d}" for j in range(n_classes))
                text += row + "\n"

            # Log as text
            self.writer.add_text(tag, text, step)

            logger.debug(f"TensorBoard: Logged confusion matrix '{tag}' at step {step}")

        except Exception as e:
            logger.warning(f"Failed to log confusion matrix: {e}")

    def log_prediction_distribution(
        self,
        y_true: np.ndarray | torch.Tensor,
        y_pred: np.ndarray | torch.Tensor,
        tag: str = "prediction_distribution",
        step: int = 0,
    ) -> None:
        """Log prediction distribution metrics (lightweight version).

        Logs histograms and statistics for regression tasks.

        Args:
            y_true: True values
            y_pred: Predicted values
            tag: Tag name in TensorBoard
            step: Global step value for logging
        """
        if not self.enabled or self.writer is None:
            return

        try:
            # Convert to tensors if needed
            if isinstance(y_true, np.ndarray):
                y_true_t = torch.from_numpy(y_true)
            else:
                y_true_t = y_true

            if isinstance(y_pred, np.ndarray):
                y_pred_t = torch.from_numpy(y_pred)
            else:
                y_pred_t = y_pred

            # Log histograms
            self.writer.add_histogram(f"{tag}/true_values", y_true_t, step)
            self.writer.add_histogram(f"{tag}/predictions", y_pred_t, step)

            # Compute and log residuals
            residuals = y_pred_t - y_true_t
            self.writer.add_histogram(f"{tag}/residuals", residuals, step)

            # Log summary statistics
            self.writer.add_scalar(
                f"{tag}/mean_absolute_error",
                torch.abs(residuals).mean().item(),
                step,
            )
            self.writer.add_scalar(
                f"{tag}/mean_squared_error",
                (residuals**2).mean().item(),
                step,
            )

            logger.debug(
                f"TensorBoard: Logged prediction distribution '{tag}' at step {step}"
            )

        except Exception as e:
            logger.warning(f"Failed to log prediction distribution: {e}")

    def log_metric_summary(
        self,
        metrics: dict[str, float],
        tag_prefix: str = "evaluation",
        step: int = 0,
    ) -> None:
        """Log multiple metrics as scalars to TensorBoard.

        Args:
            metrics: Dictionary of metric name -> value
            tag_prefix: Prefix for metric tags (e.g., "evaluation", "validation")
            step: Global step value for logging
        """
        if not self.enabled or self.writer is None:
            return

        try:
            for metric_name, value in metrics.items():
                tag = f"{tag_prefix}/{metric_name}"
                self.writer.add_scalar(tag, value, step)

            logger.debug(
                f"TensorBoard: Logged {len(metrics)} metrics with prefix '{tag_prefix}'"
            )

        except Exception as e:
            logger.warning(f"Failed to log metric summary: {e}")

    def log_class_performance(
        self,
        y_true: np.ndarray | torch.Tensor,
        y_pred: np.ndarray | torch.Tensor,
        class_names: list[str] | None = None,
        tag: str = "class_performance",
        step: int = 0,
    ) -> None:
        """Log per-class performance metrics (lightweight version).

        Computes precision, recall, and F1 manually and logs as scalars.

        Args:
            y_true: True class labels
            y_pred: Predicted class labels
            class_names: Optional class names for labeling
            tag: Tag name in TensorBoard
            step: Global step value for logging
        """
        if not self.enabled or self.writer is None:
            return

        try:
            # Convert to numpy if needed
            if isinstance(y_true, torch.Tensor):
                y_true = y_true.cpu().numpy()
            if isinstance(y_pred, torch.Tensor):
                y_pred = y_pred.cpu().numpy()

            # Flatten arrays
            y_true = y_true.flatten()
            y_pred = y_pred.flatten()

            # Get unique classes
            classes = np.unique(np.concatenate([y_true, y_pred]))

            if class_names is None:
                class_names = [f"Class_{i}" for i in classes]

            # Compute per-class metrics manually
            for i, class_id in enumerate(classes):
                if i < len(class_names):
                    class_name = class_names[i]
                else:
                    class_name = f"Class_{class_id}"

                # True positives, false positives, false negatives
                tp = np.sum((y_true == class_id) & (y_pred == class_id))
                fp = np.sum((y_true != class_id) & (y_pred == class_id))
                fn = np.sum((y_true == class_id) & (y_pred != class_id))

                # Compute metrics
                precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
                recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
                f1 = (
                    2 * precision * recall / (precision + recall)
                    if (precision + recall) > 0
                    else 0.0
                )

                # Log metrics for this class
                self.writer.add_scalar(f"{tag}/{class_name}/precision", precision, step)
                self.writer.add_scalar(f"{tag}/{class_name}/recall", recall, step)
                self.writer.add_scalar(f"{tag}/{class_name}/f1_score", f1, step)

            logger.debug(
                f"TensorBoard: Logged class performance '{tag}' at step {step}"
            )

        except Exception as e:
            logger.warning(f"Failed to log class performance: {e}")

    def close(self):
        """Close the TensorBoard writer."""
        if self.writer is not None:
            try:
                self.writer.close()
            except Exception as e:
                logger.warning(f"Error closing TensorBoard writer: {e}")

    def __enter__(self):
        """Context manager entry."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.close()
