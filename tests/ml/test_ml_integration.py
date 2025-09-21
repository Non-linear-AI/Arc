"""Simple ML integration tests for core functionality."""

import tempfile

import pytest

from arc.database.manager import DatabaseManager
from arc.database.services import JobService, MLDataService
from arc.graph.spec import ArcGraph
from arc.ml.training_service import TrainingService


@pytest.fixture
def db_manager(tmp_path):
    """Create file-based database manager for thread-safe testing."""
    system_db = tmp_path / "system.db"
    user_db = tmp_path / "user.db"
    return DatabaseManager(str(system_db), str(user_db))


@pytest.fixture
def ml_data_service(db_manager):
    """Create ML data service."""
    return MLDataService(db_manager)


@pytest.fixture
def job_service(db_manager):
    """Create job service."""
    return JobService(db_manager)


@pytest.fixture
def training_service(job_service):
    """Create training service."""
    with tempfile.TemporaryDirectory() as temp_dir:
        yield TrainingService(job_service, artifacts_dir=temp_dir)


@pytest.fixture
def simple_arc_graph():
    """Create a simple Arc Graph for testing."""
    return ArcGraph.from_yaml("""
version: "0.1"
model_name: "simple_test_model"
description: "Simple test model"

features:
  feature_columns: ["x", "y"]
  target_columns: ["label"]

model:
  inputs:
    features:
      dtype: "float32"
      shape: [null, 2]

  graph:
    - name: "linear"
      type: "Linear"
      params:
        in_features: 2
        out_features: 1
        bias: true
      inputs:
        input: "features"

  outputs:
    prediction: "linear.output"

trainer:
  optimizer:
    type: "adam"
    config:
      lr: 0.01

  loss:
    type: "binary_cross_entropy"
    inputs:
      predictions: "model.prediction"
      targets: "vars.label"

  config:
    epochs: 1
    batch_size: 4
    learning_rate: 0.01

predictor:
  returns: ["prediction"]
""")


class TestMLIntegration:
    """Test ML component integration."""

    def test_ml_data_service_creation(self, ml_data_service):
        """Test MLDataService can be created."""
        assert ml_data_service is not None
        assert ml_data_service.list_datasets() == []

    def test_job_service_creation(self, job_service):
        """Test JobService can be created."""
        assert job_service is not None
        assert job_service.list_jobs() == []

    def test_training_service_creation(self, training_service):
        """Test TrainingService can be created."""
        assert training_service is not None
        assert training_service.list_active_jobs() == []

    def test_arc_graph_parsing(self, simple_arc_graph):
        """Test Arc Graph can be parsed."""
        assert simple_arc_graph.model_name == "simple_test_model"
        assert simple_arc_graph.features.feature_columns == ["x", "y"]
        assert simple_arc_graph.features.target_columns == ["label"]

    def test_training_config_extraction(self, simple_arc_graph):
        """Test training config can be extracted from Arc Graph."""
        config = simple_arc_graph.to_training_config()
        assert config.epochs == 1
        assert config.batch_size == 4
        assert config.learning_rate == 0.01
        assert config.optimizer == "adam"

    def test_training_config_overrides(self, simple_arc_graph):
        """Test training config overrides work."""
        config = simple_arc_graph.to_training_config(
            {
                "epochs": 5,
                "batch_size": 16,
            }
        )
        assert config.epochs == 5
        assert config.batch_size == 16
        assert config.learning_rate == 0.01  # Not overridden

    def test_services_integration(self, ml_data_service, job_service, training_service):
        """Test services can work together."""
        # All services should be operational
        assert ml_data_service.list_datasets() == []
        assert job_service.list_jobs() == []
        assert training_service.list_active_jobs() == []

        # This validates that the services are properly connected
        # and that the database connections work
        assert ml_data_service.db_manager is not None
        assert job_service.db_manager is not None

    def test_training_config_missing_raises(self):
        """Graphs without trainer.config should raise when requesting config."""
        yaml_content = """
version: "0.1"
model_name: "missing_config_model"

features:
  feature_columns: [x1, x2]
  target_columns: [y]

model:
  inputs:
    features: {dtype: float32, shape: [null, 2]}
  graph:
    - name: linear
      type: core.Linear
      params: {in_features: 2, out_features: 1}
      inputs: {input: features}
  outputs:
    prediction: linear.output

trainer:
  optimizer: {type: AdamW}
  loss: {type: core.MSELoss}
"""

        graph = ArcGraph.from_yaml(yaml_content)
        with pytest.raises(ValueError, match="trainer.config is required"):
            graph.to_training_config()
