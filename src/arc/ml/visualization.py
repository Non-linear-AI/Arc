"""ML visualization utilities for TensorBoard and other visualization backends.

This module provides reusable components for visualizing ML metrics including:
- Classification metrics (PR curves, ROC curves, confusion matrices)
- Regression metrics (scatter plots, residual plots)
- Distribution visualizations

These components can be used in both training and evaluation runs.
"""

from __future__ import annotations

import io
import logging
from typing import TYPE_CHECKING

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image

# Use non-interactive backend for server environments
matplotlib.use("Agg")

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

        Args:
            y_true: True binary labels (0 or 1)
            y_scores: Predicted probabilities for positive class
            tag: Tag name for the curve in TensorBoard
            step: Global step value for logging
        """
        if not self.enabled or self.writer is None:
            return

        try:
            from sklearn.metrics import precision_recall_curve

            # Convert to numpy if needed
            if isinstance(y_true, torch.Tensor):
                y_true = y_true.cpu().numpy()
            if isinstance(y_scores, torch.Tensor):
                y_scores = y_scores.cpu().numpy()

            # Flatten arrays
            y_true = y_true.flatten()
            y_scores = y_scores.flatten()

            # Compute precision-recall curve
            precision, recall, thresholds = precision_recall_curve(y_true, y_scores)

            # Create figure
            fig, ax = plt.subplots(figsize=(8, 6))
            ax.plot(recall, precision, linewidth=2, label="PR Curve")
            ax.set_xlabel("Recall", fontsize=12)
            ax.set_ylabel("Precision", fontsize=12)
            ax.set_title("Precision-Recall Curve", fontsize=14)
            ax.grid(True, alpha=0.3)
            ax.legend()
            ax.set_xlim([0.0, 1.0])
            ax.set_ylim([0.0, 1.05])

            # Convert to image and log
            img = self._fig_to_image(fig)
            self.writer.add_image(tag, img, step, dataformats="HWC")

            plt.close(fig)
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
        """Log ROC (Receiver Operating Characteristic) curve to TensorBoard.

        Args:
            y_true: True binary labels (0 or 1)
            y_scores: Predicted probabilities for positive class
            tag: Tag name for the curve in TensorBoard
            step: Global step value for logging
        """
        if not self.enabled or self.writer is None:
            return

        try:
            from sklearn.metrics import auc, roc_curve

            # Convert to numpy if needed
            if isinstance(y_true, torch.Tensor):
                y_true = y_true.cpu().numpy()
            if isinstance(y_scores, torch.Tensor):
                y_scores = y_scores.cpu().numpy()

            # Flatten arrays
            y_true = y_true.flatten()
            y_scores = y_scores.flatten()

            # Compute ROC curve
            fpr, tpr, thresholds = roc_curve(y_true, y_scores)
            roc_auc = auc(fpr, tpr)

            # Create figure
            fig, ax = plt.subplots(figsize=(8, 6))
            ax.plot(
                fpr,
                tpr,
                linewidth=2,
                label=f"ROC Curve (AUC = {roc_auc:.3f})",
            )
            ax.plot([0, 1], [0, 1], "k--", linewidth=1, label="Random Classifier")
            ax.set_xlabel("False Positive Rate", fontsize=12)
            ax.set_ylabel("True Positive Rate", fontsize=12)
            ax.set_title("ROC Curve", fontsize=14)
            ax.grid(True, alpha=0.3)
            ax.legend(loc="lower right")
            ax.set_xlim([0.0, 1.0])
            ax.set_ylim([0.0, 1.05])

            # Convert to image and log
            img = self._fig_to_image(fig)
            self.writer.add_image(tag, img, step, dataformats="HWC")

            # Also log AUC as scalar
            self.writer.add_scalar(f"{tag}_auc", roc_auc, step)

            plt.close(fig)
            logger.debug(f"TensorBoard: Logged ROC curve '{tag}' at step {step}")

        except Exception as e:
            logger.warning(f"Failed to log ROC curve: {e}")

    def log_confusion_matrix(
        self,
        y_true: np.ndarray | torch.Tensor,
        y_pred: np.ndarray | torch.Tensor,
        class_names: list[str] | None = None,
        tag: str = "confusion_matrix",
        step: int = 0,
        normalize: bool = False,
    ) -> None:
        """Log confusion matrix to TensorBoard.

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
            from sklearn.metrics import confusion_matrix

            # Convert to numpy if needed
            if isinstance(y_true, torch.Tensor):
                y_true = y_true.cpu().numpy()
            if isinstance(y_pred, torch.Tensor):
                y_pred = y_pred.cpu().numpy()

            # Flatten arrays
            y_true = y_true.flatten()
            y_pred = y_pred.flatten()

            # Compute confusion matrix
            cm = confusion_matrix(y_true, y_pred)

            if normalize:
                cm = cm.astype("float") / cm.sum(axis=1)[:, np.newaxis]

            # Create figure
            fig, ax = plt.subplots(figsize=(10, 8))

            # Plot confusion matrix
            im = ax.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
            ax.figure.colorbar(im, ax=ax)

            # Set ticks and labels
            if class_names:
                ax.set(
                    xticks=np.arange(cm.shape[1]),
                    yticks=np.arange(cm.shape[0]),
                    xticklabels=class_names,
                    yticklabels=class_names,
                    ylabel="True label",
                    xlabel="Predicted label",
                )
            else:
                ax.set(
                    ylabel="True label",
                    xlabel="Predicted label",
                )

            # Rotate x labels
            plt.setp(
                ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor"
            )

            # Add text annotations
            fmt = ".2f" if normalize else "d"
            thresh = cm.max() / 2.0
            for i in range(cm.shape[0]):
                for j in range(cm.shape[1]):
                    ax.text(
                        j,
                        i,
                        format(cm[i, j], fmt),
                        ha="center",
                        va="center",
                        color="white" if cm[i, j] > thresh else "black",
                        fontsize=10,
                    )

            ax.set_title(
                "Normalized Confusion Matrix" if normalize else "Confusion Matrix",
                fontsize=14,
            )
            fig.tight_layout()

            # Convert to image and log
            img = self._fig_to_image(fig)
            self.writer.add_image(tag, img, step, dataformats="HWC")

            plt.close(fig)
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
        """Log distribution of predictions vs ground truth.

        Useful for regression tasks to visualize prediction quality.

        Args:
            y_true: True values
            y_pred: Predicted values
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

            # Create figure with subplots
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

            # Scatter plot: predicted vs actual
            ax1.scatter(y_true, y_pred, alpha=0.5, s=20)
            ax1.plot(
                [y_true.min(), y_true.max()],
                [y_true.min(), y_true.max()],
                "r--",
                lw=2,
            )
            ax1.set_xlabel("True Values", fontsize=12)
            ax1.set_ylabel("Predicted Values", fontsize=12)
            ax1.set_title("Predictions vs Actual", fontsize=14)
            ax1.grid(True, alpha=0.3)

            # Residual plot
            residuals = y_pred - y_true
            ax2.scatter(y_pred, residuals, alpha=0.5, s=20)
            ax2.axhline(y=0, color="r", linestyle="--", lw=2)
            ax2.set_xlabel("Predicted Values", fontsize=12)
            ax2.set_ylabel("Residuals", fontsize=12)
            ax2.set_title("Residual Plot", fontsize=14)
            ax2.grid(True, alpha=0.3)

            fig.tight_layout()

            # Convert to image and log
            img = self._fig_to_image(fig)
            self.writer.add_image(tag, img, step, dataformats="HWC")

            plt.close(fig)
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
        """Log per-class performance metrics (precision, recall, f1).

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
            from sklearn.metrics import classification_report

            # Convert to numpy if needed
            if isinstance(y_true, torch.Tensor):
                y_true = y_true.cpu().numpy()
            if isinstance(y_pred, torch.Tensor):
                y_pred = y_pred.cpu().numpy()

            # Flatten arrays
            y_true = y_true.flatten()
            y_pred = y_pred.flatten()

            # Get classification report
            report = classification_report(
                y_true,
                y_pred,
                target_names=class_names,
                output_dict=True,
                zero_division=0,
            )

            # Create bar chart for per-class metrics
            classes = class_names or [f"Class {i}" for i in sorted(set(y_true))]

            metrics_to_plot = ["precision", "recall", "f1-score"]
            x = np.arange(len(classes))
            width = 0.25

            fig, ax = plt.subplots(figsize=(12, 6))

            for idx, metric in enumerate(metrics_to_plot):
                values = [
                    report.get(cls, {}).get(metric, 0)
                    for cls in class_names or classes
                ]
                ax.bar(x + idx * width, values, width, label=metric.capitalize())

            ax.set_xlabel("Class", fontsize=12)
            ax.set_ylabel("Score", fontsize=12)
            ax.set_title("Per-Class Performance Metrics", fontsize=14)
            ax.set_xticks(x + width)
            ax.set_xticklabels(classes, rotation=45, ha="right")
            ax.legend()
            ax.grid(True, alpha=0.3, axis="y")
            ax.set_ylim([0, 1.05])

            fig.tight_layout()

            # Convert to image and log
            img = self._fig_to_image(fig)
            self.writer.add_image(tag, img, step, dataformats="HWC")

            plt.close(fig)
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

    def _fig_to_image(self, fig) -> np.ndarray:
        """Convert matplotlib figure to numpy array (RGB image).

        Args:
            fig: Matplotlib figure

        Returns:
            Numpy array of shape (H, W, 3) with RGB values
        """
        # Save figure to buffer
        buf = io.BytesIO()
        fig.savefig(buf, format="png", dpi=100, bbox_inches="tight")
        buf.seek(0)

        # Convert to PIL image and then numpy array
        img = Image.open(buf)
        img_array = np.array(img)

        # Ensure RGB format (drop alpha channel if present)
        if img_array.shape[2] == 4:
            img_array = img_array[:, :, :3]

        buf.close()
        return img_array

    def __enter__(self):
        """Context manager entry."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.close()
