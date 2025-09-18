"""Tests for the Arc Graph metrics system."""

import pytest
import torch

from arc.ml.metrics import (
    AUC,
    Accuracy,
    F1Score,
    MeanAbsoluteError,
    MeanSquaredError,
    MetricsTracker,
    Precision,
    R2Score,
    Recall,
    RootMeanSquaredError,
    create_metrics_for_task,
)


class TestAccuracy:
    """Test accuracy metric."""

    def test_binary_classification_accuracy(self):
        """Test accuracy for binary classification."""
        metric = Accuracy()

        # Perfect predictions
        predictions = torch.tensor([0.1, 0.9, 0.2, 0.8])
        targets = torch.tensor([0, 1, 0, 1])

        result = metric.compute(predictions, targets)

        assert result.name == "accuracy"
        assert result.value == 1.0  # 100% accuracy
        assert result.higher_is_better is True
        assert result.metadata["correct_predictions"] == 4
        assert result.metadata["total_predictions"] == 4

    def test_multiclass_accuracy(self):
        """Test accuracy for multiclass classification."""
        metric = Accuracy()

        # Logits for 3-class classification
        predictions = torch.tensor(
            [
                [2.0, 1.0, 0.0],  # Class 0
                [0.0, 2.0, 1.0],  # Class 1
                [1.0, 0.0, 2.0],  # Class 2
                [2.0, 1.0, 0.0],  # Class 0
            ]
        )
        targets = torch.tensor([0, 1, 2, 1])  # One wrong prediction

        result = metric.compute(predictions, targets)

        assert result.value == 0.75  # 3/4 correct
        assert result.metadata["correct_predictions"] == 3
        assert result.metadata["total_predictions"] == 4

    def test_zero_accuracy(self):
        """Test completely wrong predictions."""
        metric = Accuracy()

        predictions = torch.tensor([0.9, 0.1, 0.9, 0.1])
        targets = torch.tensor([0, 1, 0, 1])

        result = metric.compute(predictions, targets)

        assert result.value == 0.0


class TestPrecision:
    """Test precision metric."""

    def test_binary_precision(self):
        """Test precision for binary classification."""
        metric = Precision()

        predictions = torch.tensor([0, 1, 1, 0, 1])
        targets = torch.tensor([0, 1, 0, 0, 1])

        result = metric.compute(predictions, targets)

        # TP=2, FP=1, so precision = 2/3
        assert abs(result.value - 2 / 3) < 1e-6
        assert result.higher_is_better is True

    def test_perfect_precision(self):
        """Test perfect precision."""
        metric = Precision()

        predictions = torch.tensor([0, 1, 1, 0])
        targets = torch.tensor([0, 1, 1, 0])

        result = metric.compute(predictions, targets)

        assert result.value == 1.0


class TestRecall:
    """Test recall metric."""

    def test_binary_recall(self):
        """Test recall for binary classification."""
        metric = Recall()

        predictions = torch.tensor([0, 1, 0, 0, 1])
        targets = torch.tensor([0, 1, 1, 0, 1])

        result = metric.compute(predictions, targets)

        # TP=2, FN=1, so recall = 2/3
        assert abs(result.value - 2 / 3) < 1e-6
        assert result.higher_is_better is True

    def test_perfect_recall(self):
        """Test perfect recall."""
        metric = Recall()

        predictions = torch.tensor([1, 1, 1, 0])
        targets = torch.tensor([1, 1, 1, 0])

        result = metric.compute(predictions, targets)

        assert result.value == 1.0


class TestF1Score:
    """Test F1 score metric."""

    def test_f1_score(self):
        """Test F1 score calculation."""
        metric = F1Score(average="macro")

        predictions = torch.tensor([0, 1, 1, 0, 1])
        targets = torch.tensor([0, 1, 0, 0, 1])

        result = metric.compute(predictions, targets)

        # For binary classification: TP=2, FP=1, FN=0
        # Precision = 2/3, Recall = 2/2 = 1.0, F1 = 2 * (2/3 * 1) / (2/3 + 1) = 0.8
        assert abs(result.value - 0.8) < 1e-6
        assert result.higher_is_better is True
        assert abs(result.metadata["precision"] - 2 / 3) < 1e-6
        assert abs(result.metadata["recall"] - 1.0) < 1e-6

    def test_perfect_f1(self):
        """Test perfect F1 score."""
        metric = F1Score()

        predictions = torch.tensor([0, 1, 1, 0])
        targets = torch.tensor([0, 1, 1, 0])

        result = metric.compute(predictions, targets)

        assert result.value == 1.0


class TestMeanSquaredError:
    """Test MSE metric."""

    def test_mse_calculation(self):
        """Test MSE calculation."""
        metric = MeanSquaredError()

        predictions = torch.tensor([1.0, 2.0, 3.0, 4.0])
        targets = torch.tensor([1.1, 2.1, 2.9, 4.1])

        result = metric.compute(predictions, targets)

        expected_mse = (0.1**2 + 0.1**2 + 0.1**2 + 0.1**2) / 4
        assert abs(result.value - expected_mse) < 1e-6
        assert result.higher_is_better is False

    def test_perfect_mse(self):
        """Test perfect MSE (zero error)."""
        metric = MeanSquaredError()

        predictions = torch.tensor([1.0, 2.0, 3.0])
        targets = torch.tensor([1.0, 2.0, 3.0])

        result = metric.compute(predictions, targets)

        assert result.value == 0.0


class TestMeanAbsoluteError:
    """Test MAE metric."""

    def test_mae_calculation(self):
        """Test MAE calculation."""
        metric = MeanAbsoluteError()

        predictions = torch.tensor([1.0, 2.0, 3.0, 4.0])
        targets = torch.tensor([1.1, 2.1, 2.9, 4.1])

        result = metric.compute(predictions, targets)

        expected_mae = (0.1 + 0.1 + 0.1 + 0.1) / 4
        assert abs(result.value - expected_mae) < 1e-6
        assert result.higher_is_better is False


class TestRootMeanSquaredError:
    """Test RMSE metric."""

    def test_rmse_calculation(self):
        """Test RMSE calculation."""
        metric = RootMeanSquaredError()

        predictions = torch.tensor([1.0, 2.0, 3.0])
        targets = torch.tensor([2.0, 3.0, 4.0])

        result = metric.compute(predictions, targets)

        expected_rmse = ((1.0 + 1.0 + 1.0) / 3) ** 0.5
        assert abs(result.value - expected_rmse) < 1e-6
        assert result.higher_is_better is False


class TestR2Score:
    """Test R-squared score metric."""

    def test_r2_calculation(self):
        """Test R-squared calculation."""
        metric = R2Score()

        predictions = torch.tensor([2.0, 4.0, 6.0, 8.0])
        targets = torch.tensor([2.0, 4.0, 6.0, 8.0])

        result = metric.compute(predictions, targets)

        assert result.value == 1.0  # Perfect fit
        assert result.higher_is_better is True

    def test_r2_imperfect_fit(self):
        """Test R-squared with imperfect fit."""
        metric = R2Score()

        predictions = torch.tensor([1.5, 3.5, 5.5, 7.5])
        targets = torch.tensor([2.0, 4.0, 6.0, 8.0])

        result = metric.compute(predictions, targets)

        assert 0 <= result.value <= 1


class TestAUC:
    """Test AUC metric."""

    def test_auc_perfect(self):
        """Test perfect AUC."""
        metric = AUC()

        # Perfect separation
        predictions = torch.tensor([0.1, 0.2, 0.8, 0.9])
        targets = torch.tensor([0, 0, 1, 1])

        result = metric.compute(predictions, targets)

        assert result.value == 1.0
        assert result.higher_is_better is True

    def test_auc_random(self):
        """Test random AUC (should be around 0.5)."""
        metric = AUC()

        predictions = torch.tensor([0.5, 0.5, 0.5, 0.5])
        targets = torch.tensor([0, 1, 0, 1])

        result = metric.compute(predictions, targets)

        assert 0.0 <= result.value <= 1.0


class TestMetricsTracker:
    """Test metrics tracker functionality."""

    def test_empty_tracker(self):
        """Test empty metrics tracker."""
        tracker = MetricsTracker()

        assert len(tracker.metrics) == 0
        assert len(tracker.get_current_metrics()) == 0

    def test_add_metric(self):
        """Test adding metrics to tracker."""
        tracker = MetricsTracker()
        tracker.add_metric(Accuracy())
        tracker.add_metric(Precision())

        assert len(tracker.metrics) == 2
        assert "accuracy" in tracker.metrics
        assert "precision" in tracker.metrics

    def test_add_classification_metrics(self):
        """Test adding standard classification metrics."""
        tracker = MetricsTracker()
        tracker.add_classification_metrics()

        expected_metrics = ["accuracy", "precision", "recall", "f1_score"]
        for metric_name in expected_metrics:
            assert metric_name in tracker.metrics

    def test_add_regression_metrics(self):
        """Test adding standard regression metrics."""
        tracker = MetricsTracker()
        tracker.add_regression_metrics()

        expected_metrics = ["mse", "mae", "rmse", "r2_score"]
        for metric_name in expected_metrics:
            assert metric_name in tracker.metrics

    def test_compute_metrics(self):
        """Test computing all tracked metrics."""
        tracker = MetricsTracker()
        tracker.add_metric(Accuracy())
        tracker.add_metric(Precision())

        predictions = torch.tensor([0, 1, 1, 0])
        targets = torch.tensor([0, 1, 1, 0])

        results = tracker.compute_metrics(predictions, targets)

        assert len(results) == 2
        assert "accuracy" in results
        assert "precision" in results
        assert results["accuracy"].value == 1.0
        assert results["precision"].value == 1.0

    def test_get_current_metrics(self):
        """Test getting current metric values."""
        tracker = MetricsTracker()
        tracker.add_metric(Accuracy())

        predictions = torch.tensor([0, 1, 1, 0])
        targets = torch.tensor([0, 1, 1, 0])

        tracker.compute_metrics(predictions, targets)
        current = tracker.get_current_metrics()

        assert "accuracy" in current
        assert current["accuracy"] == 1.0

    def test_reset_metrics(self):
        """Test resetting metrics."""
        tracker = MetricsTracker()
        tracker.add_metric(Accuracy())

        predictions = torch.tensor([0, 1, 1, 0])
        targets = torch.tensor([0, 1, 1, 0])

        tracker.compute_metrics(predictions, targets)
        assert len(tracker.get_current_metrics()) > 0

        tracker.reset_metrics()
        assert len(tracker.get_current_metrics()) == 0

    def test_get_best_metric_name(self):
        """Test getting best metric name for model selection."""
        tracker = MetricsTracker()

        # No metrics
        assert tracker.get_best_metric_name() is None

        # Add F1 score
        tracker.add_metric(F1Score())
        assert tracker.get_best_metric_name() == "f1_score"

        # Add accuracy (should still prefer F1)
        tracker.add_metric(Accuracy())
        assert tracker.get_best_metric_name() == "f1_score"

    def test_failed_metric_computation(self):
        """Test handling of failed metric computation."""
        tracker = MetricsTracker()
        tracker.add_metric(Accuracy())

        # Invalid tensor shapes
        predictions = torch.tensor([1, 2, 3])
        targets = torch.tensor([1, 2])  # Different size

        # Should not crash, just skip failed metrics
        results = tracker.compute_metrics(predictions, targets)
        assert len(results) == 0  # No successful computations


class TestCreateMetricsForTask:
    """Test metrics creation utility."""

    def test_classification_metrics(self):
        """Test creating classification metrics."""
        tracker = create_metrics_for_task("classification")

        expected_metrics = ["accuracy", "precision", "recall", "f1_score"]
        for metric_name in expected_metrics:
            assert metric_name in tracker.metrics

    def test_binary_classification_metrics(self):
        """Test creating binary classification metrics."""
        tracker = create_metrics_for_task("binary_classification")

        expected_metrics = ["accuracy", "precision", "recall", "f1_score", "auc"]
        for metric_name in expected_metrics:
            assert metric_name in tracker.metrics

    def test_regression_metrics(self):
        """Test creating regression metrics."""
        tracker = create_metrics_for_task("regression")

        expected_metrics = ["mse", "mae", "rmse", "r2_score"]
        for metric_name in expected_metrics:
            assert metric_name in tracker.metrics

    def test_classification_with_num_classes(self):
        """Test classification with specific number of classes."""
        tracker = create_metrics_for_task("classification", num_classes=2)

        # Should include AUC for binary classification
        assert "auc" in tracker.metrics

        tracker = create_metrics_for_task("classification", num_classes=3)

        # Should not include AUC for multiclass
        assert "auc" not in tracker.metrics

    def test_invalid_task_type(self):
        """Test invalid task type raises error."""
        with pytest.raises(ValueError, match="Unknown task type"):
            create_metrics_for_task("invalid_task")

    def test_custom_average_parameter(self):
        """Test custom average parameter for classification."""
        tracker = create_metrics_for_task("classification", average="weighted")

        # Check that the average parameter is passed correctly
        precision_metric = tracker.metrics["precision"]
        assert precision_metric.average == "weighted"
