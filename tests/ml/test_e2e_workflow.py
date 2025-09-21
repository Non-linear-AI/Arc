"""End-to-end workflow tests for Arc ML training pipeline."""

import asyncio
import json
import tempfile
import time
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from arc.database.manager import DatabaseManager
from arc.database.services import JobService, MLDataService
from arc.graph.spec import ArcGraph
from arc.ml.data import DataProcessor
from arc.ml.trainer import ArcTrainer
from arc.ml.training_service import TrainingJobConfig, TrainingService


def generate_logistic_regression_data(
    n_samples: int = 100, n_features: int = 4, random_state: int = 42
) -> pd.DataFrame:
    """Generate synthetic data for logistic regression."""
    np.random.seed(random_state)

    # Generate feature data
    X = np.random.normal(0, 1, (n_samples, n_features))

    # Generate target with logistic relationship
    true_weights = np.random.normal(
        0, 1, n_features
    )  # Random coefficients based on n_features
    linear_combination = X @ true_weights

    # Add noise and apply logistic function
    noise = np.random.normal(0, 0.3, n_samples)
    probabilities = 1 / (1 + np.exp(-(linear_combination + noise)))
    y = (probabilities > 0.5).astype(int)

    # Create DataFrame
    feature_names = [f"feature_{i + 1}" for i in range(n_features)]
    data = pd.DataFrame(X, columns=feature_names)
    data["target"] = y
    data["id"] = range(1, n_samples + 1)

    # Reorder columns
    data = data[["id"] + feature_names + ["target"]]

    return data


def create_arc_graph_spec() -> str:
    """Create Arc-Graph specification for logistic regression."""
    return """
version: "0.1"
model_name: "logistic_regression_classifier"
description: "Binary logistic regression classifier with feature normalization"

features:
  feature_columns: ["feature_1", "feature_2", "feature_3", "feature_4"]
  target_columns: ["target"]
  processors:
    - name: "feature_normalizer"
      op: "core.StandardNormalization"
      train_only: false
      inputs:
        features: "tensors.features"
      outputs:
        normalized_features: "tensors.normalized_features"

model:
  inputs:
    features:
      dtype: "float32"
      shape: [null, 4]  # batch_size x n_features

  graph:
    - name: "classifier"
      type: "Linear"
      params:
        in_features: 4
        out_features: 1
        bias: true
      inputs:
        input: "features"

    - name: "sigmoid"
      type: "Sigmoid"
      inputs:
        input: "classifier.output"

  outputs:
    logits: "classifier.output"
    prediction: "sigmoid.output"

trainer:
  optimizer:
    type: "adam"
    config:
      lr: 0.01
      weight_decay: 0.001
      betas: [0.9, 0.999]

  loss:
    type: "binary_cross_entropy_with_logits"
    inputs:
      predictions: "model.logits"  # Use logits for BCEWithLogitsLoss
      targets: "vars.target"

  config:
    epochs: 5  # Short for testing
    batch_size: 32
    learning_rate: 0.01
    validation_split: 0.2
    checkpoint_every: 2

predictor:
  returns: ["prediction", "logits"]
"""


async def setup_database_and_data(data: pd.DataFrame) -> tuple[DatabaseManager, str]:
    """Setup database and load data."""
    # Create in-memory database for testing
    db_manager = DatabaseManager(
        system_db_path=":memory:",
        user_db_path=":memory:",
        shared_connections_for_tests=True,
    )

    table_name = "logistic_data"

    # Create table
    feature_cols = ", ".join(
        [f"{col} DOUBLE" for col in data.columns if col.startswith("feature_")]
    )
    create_sql = f"""
    CREATE TABLE {table_name} (
        id INTEGER PRIMARY KEY,
        {feature_cols},
        target INTEGER
    )
    """

    db_manager.user_execute(create_sql)

    # Insert data
    columns = data.columns.tolist()
    placeholders = ", ".join(["?"] * len(columns))
    insert_sql = (
        f"INSERT INTO {table_name} ({', '.join(columns)}) VALUES ({placeholders})"
    )

    # Get the underlying database and use execute with params
    user_db = db_manager._get_user_db()
    for _, row in data.iterrows():
        user_db.execute(insert_sql, list(row))

    return db_manager, table_name


class TestE2EWorkflow:
    """Test cases for end-to-end ML workflow."""

    def test_data_generation(self):
        """Test synthetic data generation."""
        data = generate_logistic_regression_data(n_samples=50, n_features=3)

        assert len(data) == 50
        assert len(data.columns) == 5  # id + 3 features + target
        assert "target" in data.columns
        assert data["target"].dtype == int
        assert set(data["target"].unique()) <= {0, 1}

        # Check feature columns exist
        for i in range(1, 4):
            assert f"feature_{i}" in data.columns

    @pytest.mark.asyncio
    async def test_database_setup(self):
        """Test database setup and data loading."""
        data = generate_logistic_regression_data(n_samples=30, n_features=2)
        db_manager, table_name = await setup_database_and_data(data)

        # Test MLDataService integration
        ml_service = MLDataService(db_manager)

        # Check dataset exists
        assert ml_service.dataset_exists(table_name)

        # Check dataset info
        info = ml_service.get_dataset_info(table_name)
        assert info.row_count == 30
        assert len(info.columns) == 4  # id + 2 features + target

        # Test data loading as tensors
        features, targets = ml_service.get_features_as_tensors(
            table_name,
            feature_columns=["feature_1", "feature_2"],
            target_columns=["target"],
        )

        assert features.shape == (30, 2)
        assert targets.shape == (30,)

    def test_arc_graph_parsing(self):
        """Test Arc-Graph specification parsing."""
        arc_graph_yaml = create_arc_graph_spec()
        arc_graph = ArcGraph.from_yaml(arc_graph_yaml)

        # Check basic properties
        assert arc_graph.model_name == "logistic_regression_classifier"
        assert arc_graph.version == "0.1"

        # Check features
        assert arc_graph.features.feature_columns == [
            "feature_1",
            "feature_2",
            "feature_3",
            "feature_4",
        ]
        assert arc_graph.features.target_columns == ["target"]

        # Check trainer config
        assert arc_graph.trainer.config is not None
        config = arc_graph.trainer.config
        assert config.epochs == 5
        assert config.batch_size == 32
        assert config.learning_rate == 0.01

        # Test training config extraction
        training_config = arc_graph.to_training_config()
        assert training_config.optimizer == "adam"
        assert training_config.loss_function == "binary_cross_entropy_with_logits"
        assert training_config.epochs == 5

    @pytest.mark.asyncio
    async def test_data_processor_integration(self):
        """Test DataProcessor with MLDataService integration."""
        data = generate_logistic_regression_data(n_samples=40, n_features=2)
        db_manager, table_name = await setup_database_and_data(data)

        ml_service = MLDataService(db_manager)
        processor = DataProcessor(ml_data_service=ml_service)

        # Test data loading
        features, targets = processor.load_from_table(
            table_name,
            feature_columns=["feature_1", "feature_2"],
            target_columns=["target"],
        )

        assert features.shape == (40, 2)
        assert targets.shape == (40,)

        # Test data loader creation
        data_loader = processor.create_dataloader_from_dataset(
            table_name,
            feature_columns=["feature_1", "feature_2"],
            target_columns=["target"],
            batch_size=16,
            shuffle=False,
        )

        # Test one batch
        batch = next(iter(data_loader))
        batch_features, batch_targets = batch
        assert batch_features.shape[0] <= 16  # batch size
        assert batch_features.shape[1] == 2  # features
        assert len(batch_targets) == len(batch_features)

    @pytest.mark.asyncio
    async def test_training_job_submission(self):
        """Test training job submission and status tracking."""
        data = generate_logistic_regression_data(n_samples=20, n_features=2)
        db_manager, table_name = await setup_database_and_data(data)

        arc_graph = ArcGraph.from_yaml(create_arc_graph_spec())

        # Setup services
        job_service = JobService(db_manager)

        with tempfile.TemporaryDirectory() as temp_dir:
            training_service = TrainingService(job_service, artifacts_dir=temp_dir)

            # Create training job config
            config = TrainingJobConfig(
                model_id="test_logistic_model",
                model_version=1,
                model_name="Test Logistic Regression",
                arc_graph=arc_graph,
                train_table=table_name,
                target_column="target",
                feature_columns=["feature_1", "feature_2"],
                description="Unit test training job",
            )

            # Submit job
            job_id = training_service.submit_training_job(config)
            assert job_id is not None
            assert isinstance(job_id, str)

            # Check job is active
            assert job_id in training_service.list_active_jobs()

            # Check initial status
            status = training_service.get_job_status(job_id)
            assert status["job_id"] == job_id
            assert status["is_active"] is True

            # Wait for completion or cancel for speed
            await asyncio.sleep(0.5)  # Let it start
            training_service.cancel_job(job_id)

            # Check final status (cancelled or failed - both acceptable)
            final_status = training_service.get_job_status(job_id)
            assert final_status["status"] in ["cancelled", "failed"]

            # Cleanup
            training_service.shutdown()

    @pytest.mark.asyncio
    async def test_training_config_persisted_in_artifact(self, monkeypatch):
        """Ensure artifacts record the effective training configuration."""
        data = generate_logistic_regression_data(n_samples=24, n_features=4)
        db_manager, table_name = await setup_database_and_data(data)

        arc_graph = ArcGraph.from_yaml(create_arc_graph_spec())
        expected_config = arc_graph.trainer.config
        assert expected_config is not None

        num_features = len(arc_graph.features.feature_columns)

        features_tensor = torch.randn(64, num_features, dtype=torch.float32)
        targets_tensor = torch.randint(0, 2, (64, 1), dtype=torch.float32)

        class DummyModel(nn.Module):
            def __init__(self, in_features: int):
                super().__init__()
                self.linear = nn.Linear(in_features, 1)
                self.input_names = ["features"]
                self.output_mapping = {"prediction": "linear.output"}
                self.execution_order = ["linear"]

            def forward(self, x: torch.Tensor) -> torch.Tensor:
                return self.linear(x)

        def fake_build_model(_self, _arc_graph):
            return DummyModel(num_features)

        def fake_create_dataloader_from_dataset(_self, *_args, **kwargs):
            batch_size = kwargs.get("batch_size", expected_config.batch_size)
            dataset = TensorDataset(features_tensor, targets_tensor)
            return DataLoader(dataset, batch_size=batch_size, shuffle=False)

        monkeypatch.setattr(
            "arc.ml.training_service.ModelBuilder.build_model", fake_build_model
        )
        monkeypatch.setattr(
            "arc.ml.training_service.DataProcessor.create_dataloader_from_dataset",
            fake_create_dataloader_from_dataset,
        )

        job_service = JobService(db_manager)

        with tempfile.TemporaryDirectory() as temp_dir:
            training_service = TrainingService(job_service, artifacts_dir=temp_dir)

            config = TrainingJobConfig(
                model_id="config_test_model",
                model_version=1,
                model_name="Config Test",
                arc_graph=arc_graph,
                train_table=table_name,
                target_column="target",
                feature_columns=["feature_1", "feature_2", "feature_3", "feature_4"],
            )

            job_id = training_service.submit_training_job(config)

            result = training_service.wait_for_job(job_id, timeout=10)
            assert result.success is True

            artifact_path = Path(temp_dir) / config.model_id / "1"
            metadata = json.loads((artifact_path / "metadata.json").read_text())
            history = json.loads((artifact_path / "training_history.json").read_text())

            assert metadata["training_config"]["epochs"] == expected_config.epochs
            assert metadata["training_config"]["learning_rate"] == pytest.approx(
                expected_config.learning_rate
            )
            assert history["config"]["batch_size"] == expected_config.batch_size

            training_service.shutdown()

    @pytest.mark.asyncio
    async def test_job_cancellation_stops_training(self, monkeypatch):
        """Cancellation should halt training and skip artifact creation."""
        data = generate_logistic_regression_data(n_samples=50, n_features=4)
        db_manager, table_name = await setup_database_and_data(data)

        arc_graph_yaml = create_arc_graph_spec().replace("epochs: 5", "epochs: 100")
        arc_graph = ArcGraph.from_yaml(arc_graph_yaml)
        expected_config = arc_graph.trainer.config
        assert expected_config is not None

        num_features = len(arc_graph.features.feature_columns)

        features_tensor = torch.randn(256, num_features, dtype=torch.float32)
        targets_tensor = torch.randint(0, 2, (256, 1), dtype=torch.float32)

        class DummyModel(nn.Module):
            def __init__(self, in_features: int):
                super().__init__()
                self.linear = nn.Linear(in_features, 1)
                self.input_names = ["features"]
                self.output_mapping = {"prediction": "linear.output"}
                self.execution_order = ["linear"]

            def forward(self, x: torch.Tensor) -> torch.Tensor:
                return self.linear(x)

        def fake_build_model(_self, _arc_graph):
            return DummyModel(num_features)

        def fake_create_dataloader_from_dataset(_self, *_args, **kwargs):
            batch_size = kwargs.get("batch_size", expected_config.batch_size)
            dataset = TensorDataset(features_tensor, targets_tensor)
            return DataLoader(dataset, batch_size=batch_size, shuffle=True)

        original_train_epoch = ArcTrainer._train_epoch

        def slow_train_epoch(self, train_loader, callback, stop_event):
            time.sleep(0.05)
            return original_train_epoch(self, train_loader, callback, stop_event)

        monkeypatch.setattr(
            "arc.ml.training_service.ModelBuilder.build_model", fake_build_model
        )
        monkeypatch.setattr(
            "arc.ml.training_service.DataProcessor.create_dataloader_from_dataset",
            fake_create_dataloader_from_dataset,
        )
        monkeypatch.setattr("arc.ml.trainer.ArcTrainer._train_epoch", slow_train_epoch)

        job_service = JobService(db_manager)

        with tempfile.TemporaryDirectory() as temp_dir:
            training_service = TrainingService(job_service, artifacts_dir=temp_dir)

            config = TrainingJobConfig(
                model_id="cancel_test_model",
                model_version=1,
                model_name="Cancel Test",
                arc_graph=arc_graph,
                train_table=table_name,
                target_column="target",
                feature_columns=["feature_1", "feature_2", "feature_3", "feature_4"],
            )

            job_id = training_service.submit_training_job(config)

            await asyncio.sleep(0.2)
            assert training_service.cancel_job(job_id) is True

            result = training_service.wait_for_job(job_id, timeout=10)
            assert result.success is False
            # Note: Due to timing, cancellation might be detected as failure
            # This is acceptable as long as training didn't complete successfully

            status = training_service.get_job_status(job_id)
            # Accept either cancelled or failed status (due to race conditions)
            assert status["status"] in ["cancelled", "failed"]

            artifact_dir = Path(temp_dir) / config.model_id
            assert not (artifact_dir / "metadata.json").exists()
            assert not (artifact_dir / "training_history.json").exists()

            training_service.shutdown()

    @pytest.mark.asyncio
    async def test_complete_e2e_pipeline(self):
        """Test complete end-to-end pipeline with very small dataset."""
        # Generate minimal data for speed
        data = generate_logistic_regression_data(n_samples=10, n_features=2)
        db_manager, table_name = await setup_database_and_data(data)

        # Parse Arc-Graph
        arc_graph_yaml = create_arc_graph_spec()
        arc_graph = ArcGraph.from_yaml(arc_graph_yaml)

        # Override config for speed
        training_config = arc_graph.to_training_config({"epochs": 1, "batch_size": 5})
        assert training_config.epochs == 1

        # Test MLDataService
        ml_service = MLDataService(db_manager)
        info = ml_service.get_dataset_info(table_name)
        assert info.row_count == 10

        # Test DataProcessor
        processor = DataProcessor(ml_data_service=ml_service)
        features, targets = processor.load_from_table(
            table_name,
            feature_columns=["feature_1", "feature_2"],
            target_columns=["target"],
        )
        assert features.shape == (10, 2)

        # Test training service setup (without actual training for speed)
        job_service = JobService(db_manager)

        with tempfile.TemporaryDirectory() as temp_dir:
            training_service = TrainingService(job_service, artifacts_dir=temp_dir)

            config = TrainingJobConfig(
                model_id="e2e_test_model",
                model_version=1,
                model_name="E2E Test Model",
                arc_graph=arc_graph,
                train_table=table_name,
                target_column="target",
                feature_columns=["feature_1", "feature_2"],
            )

            # Just test job creation, not actual training
            job_id = training_service.submit_training_job(config)

            # Quick check and cancel
            assert job_id in training_service.list_active_jobs()
            training_service.cancel_job(job_id)

            # Cleanup
            training_service.shutdown()

    def test_training_config_overrides(self):
        """Test training configuration with overrides."""
        arc_graph = ArcGraph.from_yaml(create_arc_graph_spec())

        # Test default config
        default_config = arc_graph.to_training_config()
        assert default_config.epochs == 5  # From YAML
        assert default_config.batch_size == 32  # From YAML

        # Test with overrides
        override_config = arc_graph.to_training_config(
            {
                "epochs": 20,
                "batch_size": 64,
                "learning_rate": 0.005,
            }
        )
        assert override_config.epochs == 20
        assert override_config.batch_size == 64
        assert override_config.learning_rate == 0.005
        # Original values should remain for non-overridden fields
        assert override_config.optimizer == "adam"
        assert override_config.loss_function == "binary_cross_entropy_with_logits"

    def test_feature_config_extraction(self):
        """Test feature configuration extraction."""
        arc_graph = ArcGraph.from_yaml(create_arc_graph_spec())

        features = arc_graph.features
        assert features.feature_columns == [
            "feature_1",
            "feature_2",
            "feature_3",
            "feature_4",
        ]
        assert features.target_columns == ["target"]
        assert len(features.processors) == 1
        assert features.processors[0].name == "feature_normalizer"
        assert features.processors[0].op == "core.StandardNormalization"

    def test_model_spec_validation(self):
        """Test model specification validation."""
        arc_graph = ArcGraph.from_yaml(create_arc_graph_spec())

        model = arc_graph.model
        assert "features" in model.inputs
        assert model.inputs["features"].dtype == "float32"
        assert model.inputs["features"].shape == [None, 4]

        assert len(model.graph) == 2
        assert model.graph[0].name == "classifier"
        assert model.graph[0].type == "Linear"
        assert model.graph[1].name == "sigmoid"
        assert model.graph[1].type == "Sigmoid"

        assert "logits" in model.outputs
        assert "prediction" in model.outputs
