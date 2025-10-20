"""Evaluation metrics for Arc Graph model training and assessment."""

from __future__ import annotations

import math
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any

import torch
import torch.nn.functional as F


@dataclass
class MetricResult:
    """Result from a metric computation."""

    name: str
    value: float
    higher_is_better: bool = True
    metadata: dict[str, Any] = field(default_factory=dict)

    def __str__(self) -> str:
        return f"{self.name}: {self.value:.4f}"


class Metric(ABC):
    """Abstract base class for metrics."""

    def __init__(self, name: str, higher_is_better: bool = True):
        self.name = name
        self.higher_is_better = higher_is_better

    @abstractmethod
    def compute(self, predictions: torch.Tensor, targets: torch.Tensor) -> MetricResult:
        """Compute metric value.

        Args:
            predictions: Model predictions
            targets: True target values

        Returns:
            Metric result
        """
        ...

    @abstractmethod
    def reset(self) -> None:
        """Reset metric state for new computation."""
        ...


class Accuracy(Metric):
    """Classification accuracy metric."""

    def __init__(self, name: str = "accuracy"):
        super().__init__(name, higher_is_better=True)

    def compute(self, predictions: torch.Tensor, targets: torch.Tensor) -> MetricResult:
        """Compute accuracy.

        Args:
            predictions: Logits or probabilities [batch_size, num_classes]
            targets: Class indices [batch_size]

        Returns:
            Accuracy result
        """
        if predictions.dim() > 1 and predictions.size(1) > 1:
            # Multi-class: get predicted class
            predicted = torch.argmax(predictions, dim=1)
        else:
            # Binary: apply threshold
            predicted = (predictions > 0.5).long().squeeze()

        correct = (predicted == targets).float()
        accuracy = correct.mean().item()

        return MetricResult(
            name=self.name,
            value=accuracy,
            higher_is_better=self.higher_is_better,
            metadata={
                "correct_predictions": correct.sum().item(),
                "total_predictions": len(targets),
            },
        )

    def reset(self) -> None:
        # Stateless metric, nothing to reset
        pass


class Precision(Metric):
    """Precision metric for classification."""

    def __init__(self, name: str = "precision", average: str = "macro"):
        super().__init__(name, higher_is_better=True)
        self.average = average  # 'macro', 'micro', 'weighted'

    def compute(self, predictions: torch.Tensor, targets: torch.Tensor) -> MetricResult:
        """Compute precision."""
        if predictions.dim() > 1 and predictions.size(1) > 1:
            predicted = torch.argmax(predictions, dim=1)
        else:
            predicted = (predictions > 0.5).long().squeeze()

        num_classes = max(targets.max().item(), predicted.max().item()) + 1

        if num_classes == 2 and self.average != "micro":
            # Binary classification - compute for positive class (class 1)
            tp = ((predicted == 1) & (targets == 1)).float().sum()
            fp = ((predicted == 1) & (targets == 0)).float().sum()
            precision = tp / (tp + fp + 1e-8)
        elif self.average == "micro":
            # Micro-average precision
            tp = ((predicted == targets) & (targets == 1)).float().sum()
            fp = ((predicted != targets) & (predicted == 1)).float().sum()
            precision = tp / (tp + fp + 1e-8)
        else:
            # Macro or weighted average
            precisions = []
            weights = []

            for class_idx in range(num_classes):
                tp = ((predicted == class_idx) & (targets == class_idx)).float().sum()
                fp = ((predicted == class_idx) & (targets != class_idx)).float().sum()

                class_precision = tp / (tp + fp + 1e-8)
                precisions.append(class_precision.item())

                if self.average == "weighted":
                    class_weight = (targets == class_idx).float().sum().item()
                    weights.append(class_weight)

            if self.average == "weighted" and sum(weights) > 0:
                precision = sum(
                    p * w for p, w in zip(precisions, weights, strict=False)
                ) / sum(weights)
            else:
                precision = sum(precisions) / len(precisions)

        return MetricResult(
            name=self.name,
            value=precision,
            higher_is_better=self.higher_is_better,
            metadata={"average": self.average, "num_classes": num_classes},
        )

    def reset(self) -> None:
        pass


class Recall(Metric):
    """Recall metric for classification."""

    def __init__(self, name: str = "recall", average: str = "macro"):
        super().__init__(name, higher_is_better=True)
        self.average = average

    def compute(self, predictions: torch.Tensor, targets: torch.Tensor) -> MetricResult:
        """Compute recall."""
        if predictions.dim() > 1 and predictions.size(1) > 1:
            predicted = torch.argmax(predictions, dim=1)
        else:
            predicted = (predictions > 0.5).long().squeeze()

        num_classes = max(targets.max().item(), predicted.max().item()) + 1

        if num_classes == 2 and self.average != "micro":
            # Binary classification - compute for positive class (class 1)
            tp = ((predicted == 1) & (targets == 1)).float().sum()
            fn = ((predicted == 0) & (targets == 1)).float().sum()
            recall = tp / (tp + fn + 1e-8)
        elif self.average == "micro":
            # Micro-average recall
            tp = ((predicted == targets) & (targets == 1)).float().sum()
            fn = ((predicted != targets) & (targets == 1)).float().sum()
            recall = tp / (tp + fn + 1e-8)
        else:
            # Macro or weighted average
            recalls = []
            weights = []

            for class_idx in range(num_classes):
                tp = ((predicted == class_idx) & (targets == class_idx)).float().sum()
                fn = ((predicted != class_idx) & (targets == class_idx)).float().sum()

                class_recall = tp / (tp + fn + 1e-8)
                recalls.append(class_recall.item())

                if self.average == "weighted":
                    class_weight = (targets == class_idx).float().sum().item()
                    weights.append(class_weight)

            if self.average == "weighted" and sum(weights) > 0:
                recall = sum(
                    r * w for r, w in zip(recalls, weights, strict=False)
                ) / sum(weights)
            else:
                recall = sum(recalls) / len(recalls)

        return MetricResult(
            name=self.name,
            value=recall,
            higher_is_better=self.higher_is_better,
            metadata={"average": self.average, "num_classes": num_classes},
        )

    def reset(self) -> None:
        pass


class F1Score(Metric):
    """F1 score metric for classification."""

    def __init__(self, name: str = "f1_score", average: str = "macro"):
        super().__init__(name, higher_is_better=True)
        self.precision = Precision(f"{name}_precision", average)
        self.recall = Recall(f"{name}_recall", average)

    def compute(self, predictions: torch.Tensor, targets: torch.Tensor) -> MetricResult:
        """Compute F1 score."""
        precision_result = self.precision.compute(predictions, targets)
        recall_result = self.recall.compute(predictions, targets)

        precision_val = precision_result.value
        recall_val = recall_result.value

        f1 = 2 * (precision_val * recall_val) / (precision_val + recall_val + 1e-8)

        return MetricResult(
            name=self.name,
            value=f1,
            higher_is_better=self.higher_is_better,
            metadata={
                "precision": precision_val,
                "recall": recall_val,
                "average": self.precision.average,
            },
        )

    def reset(self) -> None:
        self.precision.reset()
        self.recall.reset()


class MeanSquaredError(Metric):
    """Mean Squared Error for regression."""

    def __init__(self, name: str = "mse"):
        super().__init__(name, higher_is_better=False)

    def compute(self, predictions: torch.Tensor, targets: torch.Tensor) -> MetricResult:
        """Compute MSE."""
        mse = F.mse_loss(predictions, targets).item()

        return MetricResult(
            name=self.name,
            value=mse,
            higher_is_better=self.higher_is_better,
        )

    def reset(self) -> None:
        pass


class MeanAbsoluteError(Metric):
    """Mean Absolute Error for regression."""

    def __init__(self, name: str = "mae"):
        super().__init__(name, higher_is_better=False)

    def compute(self, predictions: torch.Tensor, targets: torch.Tensor) -> MetricResult:
        """Compute MAE."""
        mae = F.l1_loss(predictions, targets).item()

        return MetricResult(
            name=self.name,
            value=mae,
            higher_is_better=self.higher_is_better,
        )

    def reset(self) -> None:
        pass


class RootMeanSquaredError(Metric):
    """Root Mean Squared Error for regression."""

    def __init__(self, name: str = "rmse"):
        super().__init__(name, higher_is_better=False)

    def compute(self, predictions: torch.Tensor, targets: torch.Tensor) -> MetricResult:
        """Compute RMSE."""
        mse = F.mse_loss(predictions, targets).item()
        rmse = math.sqrt(mse)

        return MetricResult(
            name=self.name,
            value=rmse,
            higher_is_better=self.higher_is_better,
        )

    def reset(self) -> None:
        pass


class R2Score(Metric):
    """R-squared score for regression."""

    def __init__(self, name: str = "r2_score"):
        super().__init__(name, higher_is_better=True)

    def compute(self, predictions: torch.Tensor, targets: torch.Tensor) -> MetricResult:
        """Compute R-squared score."""
        target_mean = targets.mean()
        ss_tot = ((targets - target_mean) ** 2).sum()
        ss_res = ((targets - predictions) ** 2).sum()

        r2 = 1 - (ss_res / (ss_tot + 1e-8))

        return MetricResult(
            name=self.name,
            value=r2.item(),
            higher_is_better=self.higher_is_better,
            metadata={
                "ss_tot": ss_tot.item(),
                "ss_res": ss_res.item(),
            },
        )

    def reset(self) -> None:
        pass


class AUC(Metric):
    """Area Under the ROC Curve for binary classification."""

    def __init__(self, name: str = "auc"):
        super().__init__(name, higher_is_better=True)

    def compute(self, predictions: torch.Tensor, targets: torch.Tensor) -> MetricResult:
        """Compute AUC-ROC.

        Note: This is a simplified implementation. For production use,
        consider using scikit-learn's roc_auc_score.
        """
        # Convert to probabilities if needed
        if predictions.dim() > 1:
            if predictions.shape[-1] == 2:
                # Two-class softmax output
                probs = F.softmax(predictions, dim=1)[:, 1]
            elif predictions.shape[-1] == 1:
                # Single sigmoid output
                probs = predictions.squeeze(-1)
                probs = torch.sigmoid(probs)
            else:
                raise ValueError(
                    f"Expected 1 or 2 output columns for binary classification, "
                    f"got {predictions.shape[-1]}"
                )
        else:
            # Already 1D predictions
            probs = torch.sigmoid(predictions)

        # Sort by probability
        sorted_indices = torch.argsort(probs, descending=True)
        sorted_targets = targets[sorted_indices]

        # Compute AUC using Mann-Whitney U statistic
        n_pos = sorted_targets.sum().item()
        n_neg = len(sorted_targets) - n_pos

        if n_pos == 0 or n_neg == 0:
            auc = 0.5  # No discrimination possible
        else:
            # Calculate AUC using the trapezoid rule approximation
            # Create a simple ROC curve and calculate area under it
            thresholds = torch.unique(probs, sorted=True)
            thresholds = torch.flip(thresholds, [0])  # descending order
            thresholds = torch.cat([thresholds, torch.tensor([0.0])])

            tpr_list = []
            fpr_list = []

            for threshold in thresholds:
                predicted_pos = (probs >= threshold).float()

                tp = ((predicted_pos == 1) & (targets == 1)).sum().float()
                fp = ((predicted_pos == 1) & (targets == 0)).sum().float()
                tn = ((predicted_pos == 0) & (targets == 0)).sum().float()
                fn = ((predicted_pos == 0) & (targets == 1)).sum().float()

                tpr = tp / (tp + fn + 1e-8) if (tp + fn) > 0 else 0.0
                fpr = fp / (fp + tn + 1e-8) if (fp + tn) > 0 else 0.0

                tpr_list.append(tpr)
                fpr_list.append(fpr)

            # Calculate AUC using trapezoid rule
            fpr_tensor = torch.tensor(fpr_list)
            tpr_tensor = torch.tensor(tpr_list)

            # Sort by FPR for proper trapezoid calculation
            sorted_indices = torch.argsort(fpr_tensor)
            fpr_sorted = fpr_tensor[sorted_indices]
            tpr_sorted = tpr_tensor[sorted_indices]

            auc = torch.trapz(tpr_sorted, fpr_sorted).item()

        return MetricResult(
            name=self.name,
            value=auc,
            higher_is_better=self.higher_is_better,
            metadata={
                "n_positive": n_pos,
                "n_negative": n_neg,
            },
        )

    def reset(self) -> None:
        pass


class MetricsTracker:
    """Tracks multiple metrics during training."""

    def __init__(self):
        self.metrics: dict[str, Metric] = {}
        self.current_metrics: dict[str, float] = {}

    def add_metric(self, metric: Metric) -> None:
        """Add a metric to track."""
        self.metrics[metric.name] = metric

    def add_classification_metrics(self, average: str = "macro") -> None:
        """Add standard classification metrics."""
        self.add_metric(Accuracy())
        self.add_metric(Precision(average=average))
        self.add_metric(Recall(average=average))
        self.add_metric(F1Score(average=average))

    def add_regression_metrics(self) -> None:
        """Add standard regression metrics."""
        self.add_metric(MeanSquaredError())
        self.add_metric(MeanAbsoluteError())
        self.add_metric(RootMeanSquaredError())
        self.add_metric(R2Score())

    def compute_metrics(
        self, predictions: torch.Tensor, targets: torch.Tensor
    ) -> dict[str, MetricResult]:
        """Compute all tracked metrics."""
        results = {}

        for name, metric in self.metrics.items():
            try:
                result = metric.compute(predictions, targets)
                results[name] = result
                self.current_metrics[name] = result.value
            except Exception as e:
                # Skip metrics that fail to compute
                print(f"Warning: Failed to compute metric {name}: {e}")

        return results

    def get_current_metrics(self) -> dict[str, float]:
        """Get current metric values."""
        return self.current_metrics.copy()

    def reset_metrics(self) -> None:
        """Reset all metrics."""
        for metric in self.metrics.values():
            metric.reset()
        self.current_metrics.clear()

    def get_best_metric_name(self) -> str | None:
        """Get name of the primary metric for model selection."""
        # Return the first metric, or a commonly used one
        if "f1_score" in self.metrics:
            return "f1_score"
        elif "accuracy" in self.metrics:
            return "accuracy"
        elif "r2_score" in self.metrics:
            return "r2_score"
        elif "mse" in self.metrics:
            return "mse"
        elif self.metrics:
            return next(iter(self.metrics.keys()))
        return None


def create_metrics_for_task(task_type: str, **kwargs) -> MetricsTracker:
    """Create appropriate metrics for a given task type.

    Args:
        task_type: Type of ML task ('classification', 'regression',
                   'binary_classification')
        **kwargs: Additional arguments for metric configuration

    Returns:
        Configured metrics tracker
    """
    tracker = MetricsTracker()

    if task_type == "classification":
        average = kwargs.get("average", "macro")
        tracker.add_classification_metrics(average=average)

        # Add AUC for binary classification
        if kwargs.get("num_classes", 2) == 2:
            tracker.add_metric(AUC())

    elif task_type == "binary_classification":
        tracker.add_classification_metrics(average="binary")
        tracker.add_metric(AUC())

    elif task_type == "regression":
        tracker.add_regression_metrics()

    else:
        raise ValueError(f"Unknown task type: {task_type}")

    return tracker
