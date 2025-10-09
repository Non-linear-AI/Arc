"""Tests for ArcEvaluator."""

from unittest.mock import MagicMock, patch

import pytest
import torch
import torch.nn as nn

from arc.graph import EvaluatorSpec, ModelSpec, TrainerSpec
from arc.ml.evaluator import ArcEvaluator, EvaluationError, EvaluationResult
from arc.ml.metrics import Accuracy


class SimpleModel(nn.Module):
    """Simple test model."""

    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(10, 1)

    def forward(self, x):
        return torch.sigmoid(self.linear(x))


class TestEvaluationResult:
    """Tests for EvaluationResult."""

    def test_evaluation_result_creation(self):
        """Test creating an evaluation result."""
        result = EvaluationResult(
            evaluator_name="test_eval",
            trainer_ref="test_trainer",
            model_ref="test_model",
            version=1,
            dataset="test_data",
            num_samples=100,
            metrics={"accuracy": 0.85, "precision": 0.82},
            evaluation_time=2.5,
            timestamp="2024-01-01T12:00:00",
        )

        assert result.evaluator_name == "test_eval"
        assert result.trainer_ref == "test_trainer"
        assert result.model_ref == "test_model"
        assert result.version == 1
        assert result.dataset == "test_data"
        assert result.num_samples == 100
        assert result.metrics == {"accuracy": 0.85, "precision": 0.82}
        assert result.evaluation_time == 2.5
        assert result.timestamp == "2024-01-01T12:00:00"

    def test_evaluation_result_to_dict(self):
        """Test converting evaluation result to dict."""
        result = EvaluationResult(
            evaluator_name="test_eval",
            trainer_ref="test_trainer",
            model_ref="test_model",
            version=1,
            dataset="test_data",
            num_samples=100,
            metrics={"accuracy": 0.85},
            evaluation_time=2.5,
            timestamp="2024-01-01T12:00:00",
        )

        result_dict = result.to_dict()

        assert result_dict["evaluator_name"] == "test_eval"
        assert result_dict["trainer_ref"] == "test_trainer"
        assert result_dict["model_ref"] == "test_model"
        assert result_dict["version"] == 1
        assert result_dict["dataset"] == "test_data"
        assert result_dict["num_samples"] == 100
        assert result_dict["metrics"] == {"accuracy": 0.85}
        assert result_dict["evaluation_time"] == 2.5
        assert result_dict["timestamp"] == "2024-01-01T12:00:00"


class TestArcEvaluator:
    """Tests for ArcEvaluator."""

    @pytest.fixture
    def simple_model(self):
        """Create a simple test model."""
        return SimpleModel()

    @pytest.fixture
    def model_spec(self):
        """Create a simple model spec."""
        from arc.graph.model import GraphNode, ModelInput

        return ModelSpec(
            inputs={
                "features": ModelInput(
                    dtype="float32",
                    shape=[10],
                    columns=[
                        "f1",
                        "f2",
                        "f3",
                        "f4",
                        "f5",
                        "f6",
                        "f7",
                        "f8",
                        "f9",
                        "f10",
                    ],
                )
            },
            graph=[
                GraphNode(
                    name="linear",
                    type="torch.nn.Linear",
                    params={"in_features": 10, "out_features": 1},
                    inputs={"input": "features"},
                ),
                GraphNode(
                    name="sigmoid",
                    type="torch.sigmoid",
                    params={},
                    inputs={"input": "linear.output"},
                ),
            ],
            outputs={"probability": "sigmoid.output"},
            loss={
                "type": "torch.nn.functional.binary_cross_entropy",
                "params": {},
                "inputs": {"input": "probability", "target": "target"},
            },
        )

    @pytest.fixture
    def trainer_spec(self):
        """Create a trainer spec."""
        from arc.graph.trainer import OptimizerConfig

        return TrainerSpec(
            model_ref="test_model",
            optimizer=OptimizerConfig(type="torch.optim.Adam", params={"lr": 0.001}),
            epochs=10,
            batch_size=32,
            validation_split=0.2,
        )

    @pytest.fixture
    def evaluator_spec(self):
        """Create an evaluator spec."""
        return EvaluatorSpec(
            name="test_eval",
            trainer_ref="test_trainer",
            dataset="test_data",
            target_column="target",
            metrics=["accuracy"],
        )

    def test_evaluator_initialization(
        self, simple_model, model_spec, trainer_spec, evaluator_spec
    ):
        """Test creating an evaluator."""
        evaluator = ArcEvaluator(
            model=simple_model,
            model_spec=model_spec,
            trainer_spec=trainer_spec,
            evaluator_spec=evaluator_spec,
            artifact_version=1,
            device="cpu",
        )

        assert evaluator.model is simple_model
        assert evaluator.model_spec is model_spec
        assert evaluator.trainer_spec is trainer_spec
        assert evaluator.evaluator_spec is evaluator_spec
        assert evaluator.artifact_version == 1
        assert evaluator.device == torch.device("cpu")

    def test_evaluator_sets_model_to_eval_mode(
        self, simple_model, model_spec, trainer_spec, evaluator_spec
    ):
        """Test that evaluator sets model to evaluation mode."""
        simple_model.train()  # Set to training mode first
        assert simple_model.training is True

        evaluator = ArcEvaluator(
            model=simple_model,
            model_spec=model_spec,
            trainer_spec=trainer_spec,
            evaluator_spec=evaluator_spec,
            artifact_version=1,
        )

        assert evaluator.model.training is False  # Should be in eval mode

    def test_infer_task_type_classification(
        self, simple_model, model_spec, trainer_spec, evaluator_spec
    ):
        """Test inferring classification task from loss function."""
        evaluator = ArcEvaluator(
            model=simple_model,
            model_spec=model_spec,
            trainer_spec=trainer_spec,
            evaluator_spec=evaluator_spec,
            artifact_version=1,
        )

        task_type = evaluator._infer_task_type()
        assert task_type == "classification"

    def test_infer_task_type_regression(
        self, simple_model, model_spec, trainer_spec, evaluator_spec
    ):
        """Test inferring regression task from loss function."""
        # Modify model spec to use regression loss
        model_spec.loss = {
            "type": "torch.nn.functional.mse_loss",
            "params": {},
            "inputs": {"input": "probability", "target": "target"},
        }

        evaluator = ArcEvaluator(
            model=simple_model,
            model_spec=model_spec,
            trainer_spec=trainer_spec,
            evaluator_spec=evaluator_spec,
            artifact_version=1,
        )

        task_type = evaluator._infer_task_type()
        assert task_type == "regression"

    def test_create_metric_accuracy(
        self, simple_model, model_spec, trainer_spec, evaluator_spec
    ):
        """Test creating accuracy metric."""
        evaluator = ArcEvaluator(
            model=simple_model,
            model_spec=model_spec,
            trainer_spec=trainer_spec,
            evaluator_spec=evaluator_spec,
            artifact_version=1,
        )

        metric = evaluator._create_metric("accuracy", "classification")
        assert isinstance(metric, Accuracy)

    def test_create_metric_unknown(
        self, simple_model, model_spec, trainer_spec, evaluator_spec
    ):
        """Test creating unknown metric raises error."""
        evaluator = ArcEvaluator(
            model=simple_model,
            model_spec=model_spec,
            trainer_spec=trainer_spec,
            evaluator_spec=evaluator_spec,
            artifact_version=1,
        )

        with pytest.raises(EvaluationError, match="Unknown metric"):
            evaluator._create_metric("unknown_metric", "classification")

    def test_run_predictions(
        self, simple_model, model_spec, trainer_spec, evaluator_spec
    ):
        """Test running predictions on features."""
        evaluator = ArcEvaluator(
            model=simple_model,
            model_spec=model_spec,
            trainer_spec=trainer_spec,
            evaluator_spec=evaluator_spec,
            artifact_version=1,
        )

        # Create test features
        features = torch.randn(10, 10)

        # Run predictions
        predictions = evaluator._run_predictions(features)

        # Check predictions shape
        assert predictions.shape == (10, 1)
        # Check predictions are on CPU
        assert predictions.device.type == "cpu"

    def test_compute_metrics_with_specified_metrics(
        self, simple_model, model_spec, trainer_spec, evaluator_spec
    ):
        """Test computing metrics when metrics are specified."""
        evaluator = ArcEvaluator(
            model=simple_model,
            model_spec=model_spec,
            trainer_spec=trainer_spec,
            evaluator_spec=evaluator_spec,
            artifact_version=1,
        )

        # Create test predictions and targets
        predictions = torch.tensor([[0.8], [0.3], [0.9], [0.2]])
        targets = torch.tensor([1, 0, 1, 0])

        # Compute metrics
        metrics = evaluator._compute_metrics(predictions, targets)

        # Should compute accuracy (as specified in evaluator_spec)
        assert "accuracy" in metrics
        assert isinstance(metrics["accuracy"], float)
        assert 0.0 <= metrics["accuracy"] <= 1.0

    def test_compute_metrics_with_default_metrics(
        self, simple_model, model_spec, trainer_spec
    ):
        """Test computing metrics when no metrics are specified (use defaults)."""
        # Create evaluator spec without metrics
        evaluator_spec = EvaluatorSpec(
            name="test_eval",
            trainer_ref="test_trainer",
            dataset="test_data",
            target_column="target",
            metrics=None,  # No metrics specified
        )

        evaluator = ArcEvaluator(
            model=simple_model,
            model_spec=model_spec,
            trainer_spec=trainer_spec,
            evaluator_spec=evaluator_spec,
            artifact_version=1,
        )

        # Create test predictions and targets
        predictions = torch.tensor([[0.8], [0.3], [0.9], [0.2]])
        targets = torch.tensor([1, 0, 1, 0])

        # Compute metrics
        metrics = evaluator._compute_metrics(predictions, targets)

        # Should compute default classification metrics
        assert "accuracy" in metrics
        assert "precision" in metrics
        assert "recall" in metrics
        assert "f1_score" in metrics

    @patch("arc.ml.evaluator.ModelArtifactManager")
    @patch("arc.ml.evaluator.TrainerService")
    def test_load_from_trainer(
        self, _mock_trainer_service_class, _mock_artifact_manager_class
    ):
        """Test loading evaluator from trainer reference."""
        # Setup mocks
        mock_trainer_service = MagicMock()
        mock_artifact_manager = MagicMock()

        # Mock trainer
        mock_trainer = MagicMock()
        mock_trainer.spec = """
model_ref: test_model
optimizer:
  type: torch.optim.Adam
  params:
    lr: 0.001
config:
  epochs: 10
  batch_size: 32
  validation_split: 0.2
  target_column: target
"""
        mock_trainer_service.get_trainer_by_id.return_value = mock_trainer

        # Mock training runs for artifact loading
        mock_training_run = MagicMock()
        mock_training_run.status = MagicMock()
        mock_training_run.status.value = "completed"
        mock_training_run.artifact_path = "artifacts/test_trainer/v1/"

        from arc.database.models.training import TrainingStatus

        mock_training_run.status = TrainingStatus.COMPLETED

        # Mock db_manager for TrainingTrackingService
        mock_db_manager = MagicMock()
        mock_trainer_service.db_manager = mock_db_manager

        # Mock artifact
        mock_artifact = MagicMock()
        mock_artifact.version = 1
        mock_artifact.model_spec = {
            "inputs": {
                "features": {
                    "dtype": "float32",
                    "shape": [10],
                    "columns": [
                        "f1",
                        "f2",
                        "f3",
                        "f4",
                        "f5",
                        "f6",
                        "f7",
                        "f8",
                        "f9",
                        "f10",
                    ],
                }
            },
            "graph": [
                {
                    "name": "linear",
                    "type": "torch.nn.Linear",
                    "params": {"in_features": 10, "out_features": 1},
                    "inputs": {"input": "features"},
                },
                {
                    "name": "output",
                    "type": "torch.nn.Sigmoid",
                    "params": {},
                    "inputs": {"input": "linear.output"},
                },
            ],
            "outputs": {"probability": "output.output"},
            "loss": {
                "type": "torch.nn.functional.binary_cross_entropy",
                "params": {},
                "inputs": {"input": "probability", "target": "target"},
            },
        }

        # Build the model from the spec to get the correct structure
        from arc.graph.model import GraphNode, ModelInput
        from arc.ml.builder import ModelBuilder

        inputs = {
            "features": ModelInput(
                dtype="float32",
                shape=[10],
                columns=["f1", "f2", "f3", "f4", "f5", "f6", "f7", "f8", "f9", "f10"],
            )
        }
        graph = [
            GraphNode(
                name="linear",
                type="torch.nn.Linear",
                params={"in_features": 10, "out_features": 1},
                inputs={"input": "features"},
            ),
            GraphNode(
                name="output",
                type="torch.nn.Sigmoid",
                params={},
                inputs={"input": "linear.output"},
            ),
        ]
        from arc.graph import ModelSpec

        temp_model_spec = ModelSpec(
            inputs=inputs,
            graph=graph,
            outputs={"probability": "output.output"},
            loss={
                "type": "torch.nn.functional.binary_cross_entropy",
                "params": {},
                "inputs": {"input": "probability", "target": "target"},
            },
        )
        builder = ModelBuilder()
        temp_model = builder.build_model(temp_model_spec)
        state_dict = temp_model.state_dict()

        mock_artifact_manager.load_model_state_dict.return_value = (
            state_dict,
            mock_artifact,
        )

        # Create evaluator spec
        evaluator_spec = EvaluatorSpec(
            name="test_eval",
            trainer_ref="test_trainer",
            dataset="test_data",
            target_column="target",
            metrics=["accuracy"],
        )

        # Mock TrainingTrackingService
        with patch(
            "arc.database.services.TrainingTrackingService"
        ) as mock_tracking_service_class:
            mock_tracking_service = MagicMock()
            mock_tracking_service_class.return_value = mock_tracking_service
            mock_tracking_service.list_runs.return_value = [mock_training_run]

            # Load evaluator
            evaluator = ArcEvaluator.load_from_trainer(
                artifact_manager=mock_artifact_manager,
                trainer_service=mock_trainer_service,
                evaluator_spec=evaluator_spec,
                device="cpu",
            )

        # Verify
        assert evaluator.evaluator_spec is evaluator_spec
        assert evaluator.artifact_version == 1
        mock_trainer_service.get_trainer_by_id.assert_called_once_with("test_trainer")
        mock_artifact_manager.load_model_state_dict.assert_called_once()

    @patch("arc.ml.evaluator.ModelArtifactManager")
    @patch("arc.ml.evaluator.TrainerService")
    def test_load_from_trainer_trainer_not_found(
        self, _mock_trainer_service_class, _mock_artifact_manager_class
    ):
        """Test loading evaluator fails when trainer not found."""
        mock_trainer_service = MagicMock()
        mock_artifact_manager = MagicMock()

        # Trainer not found
        mock_trainer_service.get_trainer_by_id.return_value = None

        evaluator_spec = EvaluatorSpec(
            name="test_eval",
            trainer_ref="nonexistent_trainer",
            dataset="test_data",
            target_column="target",
        )

        with pytest.raises(EvaluationError, match="Trainer .* not found"):
            ArcEvaluator.load_from_trainer(
                artifact_manager=mock_artifact_manager,
                trainer_service=mock_trainer_service,
                evaluator_spec=evaluator_spec,
            )


class TestArcEvaluatorIntegration:
    """Integration tests for ArcEvaluator with mocked data service."""

    def test_evaluate_end_to_end(self):
        """Test complete evaluation workflow with mocked data."""
        # Create model
        model = SimpleModel()

        # Create specs
        from arc.graph.model import GraphNode, ModelInput
        from arc.graph.trainer import OptimizerConfig

        model_spec = ModelSpec(
            inputs={
                "features": ModelInput(
                    dtype="float32",
                    shape=[10],
                    columns=[
                        "f1",
                        "f2",
                        "f3",
                        "f4",
                        "f5",
                        "f6",
                        "f7",
                        "f8",
                        "f9",
                        "f10",
                    ],
                )
            },
            graph=[
                GraphNode(
                    name="linear",
                    type="torch.nn.Linear",
                    params={"in_features": 10, "out_features": 1},
                    inputs={"input": "features"},
                ),
                GraphNode(
                    name="sigmoid",
                    type="torch.sigmoid",
                    params={},
                    inputs={"input": "linear.output"},
                ),
            ],
            outputs={"probability": "sigmoid.output"},
            loss={
                "type": "torch.nn.functional.binary_cross_entropy",
                "params": {},
                "inputs": {"input": "probability", "target": "target"},
            },
        )

        trainer_spec = TrainerSpec(
            model_ref="test_model",
            optimizer=OptimizerConfig(type="torch.optim.Adam", params={"lr": 0.001}),
            epochs=10,
            batch_size=32,
            validation_split=0.2,
        )

        evaluator_spec = EvaluatorSpec(
            name="test_eval",
            trainer_ref="test_trainer",
            dataset="test_data",
            target_column="target",
            metrics=["accuracy"],
        )

        # Create evaluator
        evaluator = ArcEvaluator(
            model=model,
            model_spec=model_spec,
            trainer_spec=trainer_spec,
            evaluator_spec=evaluator_spec,
            artifact_version=1,
        )

        # Mock ML data service
        mock_ml_data_service = MagicMock()

        # Create test data
        test_features = torch.randn(100, 10)
        test_targets = torch.randint(0, 2, (100,))

        mock_ml_data_service.get_features_as_tensors.return_value = (
            test_features,
            test_targets,
        )

        # Run evaluation
        result = evaluator.evaluate(mock_ml_data_service)

        # Verify result
        assert isinstance(result, EvaluationResult)
        assert result.evaluator_name == "test_eval"
        assert result.trainer_ref == "test_trainer"
        assert result.model_ref == "test_model"
        assert result.version == 1
        assert result.dataset == "test_data"
        assert result.num_samples == 100
        assert "accuracy" in result.metrics
        assert isinstance(result.evaluation_time, float)
        assert result.evaluation_time > 0
        assert result.timestamp is not None
