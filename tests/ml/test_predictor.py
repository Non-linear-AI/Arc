"""Tests for ArcPredictor streaming functionality."""

import pytest
import torch
import torch.nn as nn

from arc.database.manager import DatabaseManager
from arc.database.services import MLDataService
from arc.graph.spec import ArcGraph
from arc.ml.artifacts import ModelArtifact
from arc.ml.predictor import ArcPredictor, PredictionError


@pytest.fixture
def db_manager():
    """Create in-memory database manager for testing."""
    return DatabaseManager(system_db_path=":memory:", user_db_path=":memory:")


@pytest.fixture
def ml_data_service(db_manager):
    """Create ML data service for testing."""
    return MLDataService(db_manager)


@pytest.fixture
def sample_dataset(ml_data_service):
    """Create sample dataset with test data for streaming tests."""
    # Create test table with enough rows to test chunking
    create_sql = """
    CREATE TABLE stream_test_dataset (
        id INTEGER PRIMARY KEY,
        feature1 DOUBLE,
        feature2 DOUBLE,
        target INTEGER
    )
    """
    ml_data_service.db_manager.user_execute(create_sql)

    # Insert 15 rows of test data (larger than typical chunk size of 5 for testing)
    values = []
    for i in range(15):
        feature1 = float(i)
        feature2 = float(i * 2)
        target = i % 2
        values.append(f"({i + 1}, {feature1}, {feature2}, {target})")

    insert_sql = f"INSERT INTO stream_test_dataset VALUES {', '.join(values)}"
    ml_data_service.db_manager.user_execute(insert_sql)

    return "stream_test_dataset"


@pytest.fixture
def simple_arc_graph():
    """Create simple ArcGraph for testing."""
    return ArcGraph.from_dict(
        {
            "version": "0.1",
            "model_name": "test_model",
            "features": {"feature_columns": ["feature1", "feature2"]},
            "model": {
                "inputs": {"features": {"dtype": "float32", "shape": [None, 2]}},
                "graph": [
                    {
                        "name": "linear",
                        "type": "core.Linear",
                        "params": {"in_features": 2, "out_features": 1, "bias": True},
                        "inputs": {"input": "features"},
                    }
                ],
                "outputs": {"prediction": "linear.output"},
            },
            "trainer": {
                "optimizer": {"type": "adam"},
                "loss": {"type": "mse"},
                "config": {"epochs": 1, "batch_size": 2, "learning_rate": 0.01},
            },
        }
    )


@pytest.fixture
def simple_model():
    """Create simple PyTorch model for testing."""
    return nn.Linear(2, 1)


@pytest.fixture
def arc_predictor(simple_model, simple_arc_graph):
    """Create ArcPredictor for testing."""
    artifact = ModelArtifact(
        model_id="test_model",
        model_name="Test Model",
        version=1,
        model_state_path="model.pt",
    )

    return ArcPredictor(
        model=simple_model, arc_graph=simple_arc_graph, artifact=artifact, device="cpu"
    )


class TestArcPredictorStreaming:
    """Test ArcPredictor streaming prediction functionality."""

    def test_predict_from_table_streaming_basic(
        self, arc_predictor, ml_data_service, sample_dataset
    ):
        """Test basic streaming prediction functionality."""
        predictions = arc_predictor.predict_from_table_streaming(
            ml_data_service=ml_data_service,
            table_name=sample_dataset,
            feature_columns=["feature1", "feature2"],
            batch_size=2,
            chunk_size=5,  # Small chunk size to test streaming
        )

        # Verify predictions structure
        assert isinstance(predictions, dict)
        assert "prediction" in predictions
        assert isinstance(predictions["prediction"], torch.Tensor)

        # Verify we got predictions for all 15 rows
        assert predictions["prediction"].shape[0] == 15

        # Verify prediction shape (15 samples, 1 output)
        assert predictions["prediction"].shape == (15, 1)

    def test_predict_from_table_streaming_chunking(
        self, arc_predictor, ml_data_service, sample_dataset
    ):
        """Test that chunking produces same results as different chunk sizes."""
        # Predict with small chunks
        predictions_small = arc_predictor.predict_from_table_streaming(
            ml_data_service=ml_data_service, table_name=sample_dataset, chunk_size=3
        )

        # Predict with large chunks
        predictions_large = arc_predictor.predict_from_table_streaming(
            ml_data_service=ml_data_service, table_name=sample_dataset, chunk_size=10
        )

        # Results should be identical regardless of chunk size
        torch.testing.assert_close(
            predictions_small["prediction"], predictions_large["prediction"]
        )

    def test_predict_from_table_streaming_missing_columns(
        self, arc_predictor, ml_data_service, sample_dataset
    ):
        """Test error handling when feature columns are missing."""
        with pytest.raises(PredictionError, match="Missing feature columns"):
            arc_predictor.predict_from_table_streaming(
                ml_data_service=ml_data_service,
                table_name=sample_dataset,
                feature_columns=["feature1", "nonexistent_column"],
            )

    def test_predict_from_table_streaming_nonexistent_table(
        self, arc_predictor, ml_data_service
    ):
        """Test error handling when table doesn't exist."""
        with pytest.raises(PredictionError, match="does not exist"):
            arc_predictor.predict_from_table_streaming(
                ml_data_service=ml_data_service, table_name="nonexistent_table"
            )

    def test_predict_from_table_streaming_with_limit(
        self, arc_predictor, ml_data_service, sample_dataset
    ):
        """Test streaming prediction with row limit."""
        predictions = arc_predictor.predict_from_table_streaming(
            ml_data_service=ml_data_service,
            table_name=sample_dataset,
            limit=7,  # Only process first 7 rows
        )

        # Should only get 7 predictions
        assert predictions["prediction"].shape[0] == 7
