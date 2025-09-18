"""End-to-end integration tests for the complete Arc ML workflow."""

from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch

from arc.database.manager import DatabaseManager
from arc.database.services import ServiceContainer
from arc.graph.spec import ArcGraph
from arc.ml.artifacts import ModelArtifactManager
from arc.ml.predictor import ArcPredictor, PredictionError
from arc.ml.training_service import TrainingJobConfig, TrainingService

# Sample Arc-Graph specification for testing
SIMPLE_CLASSIFIER_YAML = """
version: "0.1"
model_name: "simple_test_classifier"
description: "Simple binary classifier for end-to-end testing"

features:
  feature_columns: [feature1, feature2]
  target_columns: [target]
  processors: []

model:
  inputs:
    features: {dtype: float32, shape: [null, 2]}
  graph:
    - name: linear
      type: core.Linear
      params: {in_features: 2, out_features: 1, bias: true}
      inputs: {input: features}
    - name: sigmoid
      type: core.Sigmoid
      inputs: {input: linear.output}
  outputs:
    logits: linear.output
    prediction: sigmoid.output

trainer:
  optimizer: {type: adam}
  loss: {type: binary_cross_entropy_with_logits}
  config:
    epochs: 2
    batch_size: 4
    learning_rate: 0.1
    target_output_key: logits
    reshape_targets: true

predictor:
  returns: [prediction, logits]
"""


class TestEndToEndWorkflow:
    """Test complete create → train → predict workflow."""

    @pytest.fixture
    def temp_artifacts_dir(self):
        """Create temporary directory for artifacts."""
        with tempfile.TemporaryDirectory() as temp_dir:
            yield Path(temp_dir)

    @pytest.fixture
    def database_manager(self):
        """Create in-memory database for testing."""
        manager = DatabaseManager(":memory:", ":memory:")
        yield manager
        manager.close()

    @pytest.fixture
    def services(self, database_manager):
        """Create service container."""
        return ServiceContainer(database_manager)

    @pytest.fixture
    def sample_data(self, database_manager):
        """Create sample training data in database."""
        # Create table
        database_manager.user_execute(
            """
            CREATE TABLE test_data (
                feature1 DOUBLE,
                feature2 DOUBLE,
                target DOUBLE
            )
            """
        )

        # Generate sample data for binary classification
        np.random.seed(42)
        n_samples = 20

        # Generate features
        feature1 = np.random.randn(n_samples)
        feature2 = np.random.randn(n_samples)

        # Generate targets (simple linear decision boundary)
        targets = ((feature1 + feature2) > 0).astype(float)

        # Insert data
        for f1, f2, t in zip(feature1, feature2, targets, strict=False):
            database_manager.user_execute(
                f"INSERT INTO test_data VALUES ({f1}, {f2}, {t})"
            )

        return {
            "table_name": "test_data",
            "feature_columns": ["feature1", "feature2"],
            "target_column": "target",
            "n_samples": n_samples,
        }

    def test_create_train_predict_workflow(
        self, temp_artifacts_dir, services, sample_data
    ):
        """Test the complete workflow: create model → train → predict."""

        # Phase 1: Create model from Arc-Graph specification
        arc_graph = ArcGraph.from_yaml(SIMPLE_CLASSIFIER_YAML)

        # Verify the spec is valid
        assert arc_graph.model_name == "simple_test_classifier"
        assert arc_graph.features.feature_columns == ["feature1", "feature2"]
        assert arc_graph.predictor.returns == ["prediction", "logits"]

        # Phase 2: Train the model
        training_service = TrainingService(
            job_service=services.jobs, artifacts_dir=temp_artifacts_dir
        )

        job_config = TrainingJobConfig(
            model_id="test-model",
            model_version=1,
            model_name="test_model",
            arc_graph=arc_graph,
            train_table=sample_data["table_name"],
            target_column=sample_data["target_column"],
            feature_columns=sample_data["feature_columns"],
            training_config=arc_graph.to_training_config(),
            description="End-to-end test training",
        )

        # Submit training job
        job_id = training_service.submit_training_job(job_config)
        assert job_id is not None

        # Wait for training to complete
        result = training_service.wait_for_job(job_id, timeout=30)
        assert result is not None
        assert result.success is True
        assert result.total_epochs >= 1

        # Verify artifacts were saved
        artifact_manager = ModelArtifactManager(temp_artifacts_dir)
        artifacts = artifact_manager.list_artifacts()
        assert len(artifacts) >= 1

        # Find our model artifact
        test_artifacts = [a for a in artifacts if a.model_id == "test-model"]
        assert len(test_artifacts) == 1

        # Phase 3: Load model for prediction
        predictor = ArcPredictor.load_from_artifact(
            artifact_manager=artifact_manager,
            model_id="test-model",
            device="cpu",
        )

        # Verify predictor is configured correctly
        predictor_info = predictor.to_dict()
        assert predictor_info["model_id"] == "test-model"
        assert predictor_info["output_keys"] == ["prediction", "logits"]
        assert predictor_info["feature_columns"] == ["feature1", "feature2"]

        # Phase 4: Run predictions

        # Test 4A: Predict from database table
        predictions_table = predictor.predict_from_table(
            ml_data_service=services.ml_data,
            table_name=sample_data["table_name"],
            batch_size=8,
        )

        assert "prediction" in predictions_table
        assert "logits" in predictions_table
        assert predictions_table["prediction"].shape[0] == sample_data["n_samples"]
        assert predictions_table["logits"].shape[0] == sample_data["n_samples"]

        # Verify predictions are probabilities (between 0 and 1)
        preds = predictions_table["prediction"]
        assert torch.all(preds >= 0) and torch.all(preds <= 1)

        # Test 4B: Single prediction
        single_features = torch.tensor([0.5, -0.3], dtype=torch.float32)
        single_pred = predictor.predict_single(single_features)

        assert "prediction" in single_pred
        assert "logits" in single_pred
        assert single_pred["prediction"].dim() == 0  # Scalar
        assert single_pred["logits"].dim() == 0  # Scalar

        # Test 4C: DataFrame prediction
        test_df = pd.DataFrame(
            {
                "feature1": [1.0, -1.0, 0.5],
                "feature2": [0.5, 1.5, -0.2],
            }
        )

        df_predictions = predictor.predict_dataframe(test_df, batch_size=2)
        assert "prediction" in df_predictions
        assert "logits" in df_predictions
        assert df_predictions["prediction"].shape[0] == 3

        # Phase 5: Verify prediction consistency
        # The same input should produce the same output
        single_pred_2 = predictor.predict_single(single_features)
        assert torch.allclose(single_pred["prediction"], single_pred_2["prediction"])
        assert torch.allclose(single_pred["logits"], single_pred_2["logits"])

    def test_load_from_checkpoint_workflow(
        self, temp_artifacts_dir, services, sample_data
    ):
        """Test loading predictor from training checkpoint."""

        # Train a model and get checkpoint
        arc_graph = ArcGraph.from_yaml(SIMPLE_CLASSIFIER_YAML)
        training_service = TrainingService(
            job_service=services.jobs, artifacts_dir=temp_artifacts_dir
        )

        job_config = TrainingJobConfig(
            model_id="checkpoint-test",
            model_version=1,
            model_name="checkpoint_test",
            arc_graph=arc_graph,
            train_table=sample_data["table_name"],
            target_column=sample_data["target_column"],
            feature_columns=sample_data["feature_columns"],
            training_config=arc_graph.to_training_config(),
            description="Checkpoint test training",
        )

        job_id = training_service.submit_training_job(job_config)
        result = training_service.wait_for_job(job_id, timeout=30)
        assert result is not None and result.success

        # Find checkpoint file
        checkpoint_dir = temp_artifacts_dir / "checkpoint-test" / "checkpoints"
        checkpoint_files = list(checkpoint_dir.glob("*.pt"))
        assert len(checkpoint_files) > 0

        checkpoint_path = checkpoint_files[0]

        # Load predictor from checkpoint
        predictor = ArcPredictor.load_from_checkpoint(
            checkpoint_path=checkpoint_path, arc_graph=arc_graph, device="cpu"
        )

        # Test prediction works
        single_features = torch.tensor([0.1, 0.2], dtype=torch.float32)
        predictions = predictor.predict_single(single_features)

        assert "prediction" in predictions
        assert "logits" in predictions

    def test_prediction_error_handling(self, temp_artifacts_dir, services, sample_data):
        """Test error handling in prediction workflow."""

        # Train a model first
        arc_graph = ArcGraph.from_yaml(SIMPLE_CLASSIFIER_YAML)
        training_service = TrainingService(
            job_service=services.jobs, artifacts_dir=temp_artifacts_dir
        )

        job_config = TrainingJobConfig(
            model_id="error-test",
            model_version=1,
            model_name="error_test",
            arc_graph=arc_graph,
            train_table=sample_data["table_name"],
            target_column=sample_data["target_column"],
            feature_columns=sample_data["feature_columns"],
            training_config=arc_graph.to_training_config(),
            description="Error handling test",
        )

        job_id = training_service.submit_training_job(job_config)
        result = training_service.wait_for_job(job_id, timeout=30)
        assert result is not None and result.success

        # Load predictor
        artifact_manager = ModelArtifactManager(temp_artifacts_dir)
        predictor = ArcPredictor.load_from_artifact(
            artifact_manager=artifact_manager,
            model_id="error-test",
            device="cpu",
        )

        # Test wrong input shape
        with pytest.raises(PredictionError):
            wrong_features = torch.tensor(
                [1.0, 2.0, 3.0], dtype=torch.float32
            )  # 3 features instead of 2
            predictor.predict_single(wrong_features)

        # Test missing columns in DataFrame
        with pytest.raises(PredictionError):
            bad_df = pd.DataFrame({"wrong_col": [1.0, 2.0]})
            predictor.predict_dataframe(bad_df)

        # Test nonexistent table
        with pytest.raises(PredictionError):
            predictor.predict_from_table(
                ml_data_service=services.ml_data, table_name="nonexistent_table"
            )

    def test_custom_predictor_spec(self, temp_artifacts_dir, services, sample_data):
        """Test with custom predictor specification."""

        # Arc-Graph that only returns logits
        custom_yaml = """
version: "0.1"
model_name: "custom_predictor_test"

features:
  feature_columns: [feature1, feature2]
  target_columns: [target]
  processors: []

model:
  inputs:
    features: {dtype: float32, shape: [null, 2]}
  graph:
    - name: linear
      type: core.Linear
      params: {in_features: 2, out_features: 1, bias: true}
      inputs: {input: features}
    - name: sigmoid
      type: core.Sigmoid
      inputs: {input: linear.output}
  outputs:
    logits: linear.output
    prediction: sigmoid.output

trainer:
  optimizer: {type: adam}
  loss: {type: binary_cross_entropy_with_logits}
  config:
    epochs: 1
    batch_size: 4
    target_output_key: logits
    reshape_targets: true

predictor:
  returns: [logits]  # Only return logits
"""

        arc_graph = ArcGraph.from_yaml(custom_yaml)

        # Train model
        training_service = TrainingService(
            job_service=services.jobs, artifacts_dir=temp_artifacts_dir
        )

        job_config = TrainingJobConfig(
            model_id="custom-pred",
            model_version=1,
            model_name="custom_pred",
            arc_graph=arc_graph,
            train_table=sample_data["table_name"],
            target_column=sample_data["target_column"],
            feature_columns=sample_data["feature_columns"],
            training_config=arc_graph.to_training_config(),
            description="Custom predictor test",
        )

        job_id = training_service.submit_training_job(job_config)
        result = training_service.wait_for_job(job_id, timeout=30)
        assert result is not None and result.success

        # Load predictor
        artifact_manager = ModelArtifactManager(temp_artifacts_dir)
        predictor = ArcPredictor.load_from_artifact(
            artifact_manager=artifact_manager,
            model_id="custom-pred",
            device="cpu",
        )

        # Verify only logits are returned
        assert predictor.output_keys == ["logits"]

        # Test prediction
        single_features = torch.tensor([0.1, 0.2], dtype=torch.float32)
        predictions = predictor.predict_single(single_features)

        assert "logits" in predictions
        assert "prediction" not in predictions  # Should not be returned
