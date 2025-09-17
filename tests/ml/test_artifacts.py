"""Tests for model artifact management."""

import json
import tempfile
from pathlib import Path

import pytest
import torch
import torch.nn as nn

from arc.ml.artifacts import (
    ModelArtifact,
    ModelArtifactManager,
    create_artifact_from_training,
)
from arc.ml.trainer import TrainingConfig, TrainingResult


class SimpleModel(nn.Module):
    """Simple model for testing."""

    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(10, 2)

    def forward(self, x):
        return self.linear(x)


@pytest.fixture
def temp_artifacts_dir():
    """Create temporary artifacts directory."""
    with tempfile.TemporaryDirectory() as temp_dir:
        yield Path(temp_dir)


@pytest.fixture
def sample_artifact():
    """Create sample artifact metadata."""
    return ModelArtifact(
        model_id="test_model",
        model_name="Test Model",
        version=1,
        description="A test model",
        tags=["test", "simple"],
    )


@pytest.fixture
def sample_training_result():
    """Create sample training result."""
    return TrainingResult(
        success=True,
        total_epochs=5,
        best_epoch=3,
        final_train_loss=0.5,
        final_val_loss=0.3,
        best_val_loss=0.25,
        train_losses=[1.0, 0.8, 0.6, 0.5, 0.5],
        val_losses=[0.9, 0.7, 0.4, 0.3, 0.3],
        training_time=120.5,
    )


class TestModelArtifact:
    """Test model artifact data structure."""

    def test_artifact_creation(self):
        """Test creating model artifact."""
        artifact = ModelArtifact(
            model_id="test_model",
            model_name="Test Model",
            version=1,
        )

        assert artifact.model_id == "test_model"
        assert artifact.model_name == "Test Model"
        assert artifact.version == 1
        assert artifact.created_at != ""
        assert artifact.updated_at != ""

    def test_artifact_with_metadata(self):
        """Test artifact with additional metadata."""
        artifact = ModelArtifact(
            model_id="test_model",
            model_name="Test Model",
            version=1,
            description="Test description",
            tags=["tag1", "tag2"],
            final_metrics={"accuracy": 0.95},
        )

        assert artifact.description == "Test description"
        assert artifact.tags == ["tag1", "tag2"]
        assert artifact.final_metrics["accuracy"] == 0.95

    def test_artifact_timestamps(self):
        """Test artifact timestamp handling."""
        # Without timestamps
        artifact1 = ModelArtifact(
            model_id="test1",
            model_name="Test 1",
            version=1,
        )

        assert artifact1.created_at != ""
        assert artifact1.updated_at == artifact1.created_at

        # With custom timestamps
        custom_time = "2023-01-01T00:00:00"
        artifact2 = ModelArtifact(
            model_id="test2",
            model_name="Test 2",
            version=1,
            created_at=custom_time,
            updated_at=custom_time,
        )

        assert artifact2.created_at == custom_time
        assert artifact2.updated_at == custom_time


class TestModelArtifactManager:
    """Test model artifact manager."""

    def test_manager_initialization(self, temp_artifacts_dir):
        """Test manager initialization."""
        manager = ModelArtifactManager(temp_artifacts_dir)

        assert manager.artifacts_dir == temp_artifacts_dir
        assert temp_artifacts_dir.exists()

    def test_get_artifact_path(self, temp_artifacts_dir):
        """Test artifact path generation."""
        manager = ModelArtifactManager(temp_artifacts_dir)

        path = manager.get_artifact_path("model1", 2)
        expected = temp_artifacts_dir / "model1" / "2"

        assert path == expected

    def test_save_model_artifact(self, temp_artifacts_dir, sample_artifact):
        """Test saving model artifact."""
        manager = ModelArtifactManager(temp_artifacts_dir)
        model = SimpleModel()

        artifact_dir = manager.save_model_artifact(model, sample_artifact)

        # Check directory structure
        assert artifact_dir.exists()
        assert (artifact_dir / "model_state.pt").exists()
        assert (artifact_dir / "metadata.json").exists()

        # Check metadata file
        with open(artifact_dir / "metadata.json") as f:
            metadata = json.load(f)
            assert metadata["model_id"] == sample_artifact.model_id
            assert metadata["version"] == sample_artifact.version

    def test_save_with_optimizer(self, temp_artifacts_dir, sample_artifact):
        """Test saving with optimizer state."""
        manager = ModelArtifactManager(temp_artifacts_dir)
        model = SimpleModel()
        optimizer = torch.optim.Adam(model.parameters())

        artifact_dir = manager.save_model_artifact(
            model, sample_artifact, optimizer=optimizer
        )

        assert (artifact_dir / "optimizer_state.pt").exists()

    def test_save_with_training_history(self, temp_artifacts_dir, sample_artifact):
        """Test saving with training history."""
        manager = ModelArtifactManager(temp_artifacts_dir)
        model = SimpleModel()

        training_history = {
            "train_losses": [1.0, 0.8, 0.6],
            "val_losses": [0.9, 0.7, 0.5],
        }

        artifact_dir = manager.save_model_artifact(
            model, sample_artifact, training_history=training_history
        )

        history_file = artifact_dir / "training_history.json"
        assert history_file.exists()

        with open(history_file) as f:
            saved_history = json.load(f)
            assert saved_history["train_losses"] == [1.0, 0.8, 0.6]

    def test_save_overwrite_protection(self, temp_artifacts_dir, sample_artifact):
        """Test overwrite protection."""
        manager = ModelArtifactManager(temp_artifacts_dir)
        model = SimpleModel()

        # Save first time
        manager.save_model_artifact(model, sample_artifact)

        # Try to save again without overwrite
        with pytest.raises(FileExistsError):
            manager.save_model_artifact(model, sample_artifact, overwrite=False)

        # Should work with overwrite=True
        manager.save_model_artifact(model, sample_artifact, overwrite=True)

    def test_load_model_state_dict(self, temp_artifacts_dir, sample_artifact):
        """Test loading model state dict."""
        manager = ModelArtifactManager(temp_artifacts_dir)
        model = SimpleModel()

        # Save artifact
        manager.save_model_artifact(model, sample_artifact)

        # Load state dict
        state_dict, loaded_artifact = manager.load_model_state_dict(
            sample_artifact.model_id, sample_artifact.version
        )

        assert isinstance(state_dict, dict)
        assert "linear.weight" in state_dict
        assert loaded_artifact.model_id == sample_artifact.model_id

    def test_load_latest_version(self, temp_artifacts_dir, sample_artifact):
        """Test loading latest version."""
        manager = ModelArtifactManager(temp_artifacts_dir)
        model = SimpleModel()

        # Save multiple versions
        artifact_v1 = sample_artifact
        artifact_v2 = ModelArtifact(
            model_id=sample_artifact.model_id,
            model_name=sample_artifact.model_name,
            version=2,
        )

        manager.save_model_artifact(model, artifact_v1)
        manager.save_model_artifact(model, artifact_v2)

        # Load without specifying version (should get latest)
        _, loaded_artifact = manager.load_model_state_dict(sample_artifact.model_id)

        assert loaded_artifact.version == 2

    def test_load_nonexistent_artifact(self, temp_artifacts_dir):
        """Test loading nonexistent artifact."""
        manager = ModelArtifactManager(temp_artifacts_dir)

        with pytest.raises(FileNotFoundError):
            manager.load_model_state_dict("nonexistent_model", 1)

    def test_list_artifacts(self, temp_artifacts_dir):
        """Test listing artifacts."""
        manager = ModelArtifactManager(temp_artifacts_dir)
        model = SimpleModel()

        # Save multiple artifacts
        artifact1 = ModelArtifact(model_id="model1", model_name="Model 1", version=1)
        artifact2 = ModelArtifact(model_id="model1", model_name="Model 1", version=2)
        artifact3 = ModelArtifact(model_id="model2", model_name="Model 2", version=1)

        manager.save_model_artifact(model, artifact1)
        manager.save_model_artifact(model, artifact2)
        manager.save_model_artifact(model, artifact3)

        # List all artifacts
        all_artifacts = manager.list_artifacts()
        assert len(all_artifacts) == 3

        # List artifacts for specific model
        model1_artifacts = manager.list_artifacts("model1")
        assert len(model1_artifacts) == 2
        assert all(a.model_id == "model1" for a in model1_artifacts)

    def test_get_latest_version(self, temp_artifacts_dir):
        """Test getting latest version."""
        manager = ModelArtifactManager(temp_artifacts_dir)
        model = SimpleModel()

        # No versions exist
        with pytest.raises(FileNotFoundError):
            manager.get_latest_version("nonexistent_model")

        # Save multiple versions
        for version in [1, 3, 2]:  # Not in order
            artifact = ModelArtifact(
                model_id="test_model",
                model_name="Test Model",
                version=version,
            )
            manager.save_model_artifact(model, artifact)

        latest = manager.get_latest_version("test_model")
        assert latest == 3

    def test_delete_artifact(self, temp_artifacts_dir, sample_artifact):
        """Test deleting specific artifact version."""
        manager = ModelArtifactManager(temp_artifacts_dir)
        model = SimpleModel()

        # Save artifact
        manager.save_model_artifact(model, sample_artifact)
        artifact_path = manager.get_artifact_path(
            sample_artifact.model_id, sample_artifact.version
        )
        assert artifact_path.exists()

        # Delete artifact
        manager.delete_artifact(sample_artifact.model_id, sample_artifact.version)
        assert not artifact_path.exists()

    def test_delete_model_artifacts(self, temp_artifacts_dir):
        """Test deleting all artifacts for a model."""
        manager = ModelArtifactManager(temp_artifacts_dir)
        model = SimpleModel()

        # Save multiple versions
        for version in [1, 2, 3]:
            artifact = ModelArtifact(
                model_id="test_model",
                model_name="Test Model",
                version=version,
            )
            manager.save_model_artifact(model, artifact)

        model_dir = temp_artifacts_dir / "test_model"
        assert model_dir.exists()

        # Delete all artifacts
        manager.delete_model_artifacts("test_model")
        assert not model_dir.exists()

    def test_copy_artifact(self, temp_artifacts_dir, sample_artifact):
        """Test copying artifact."""
        manager = ModelArtifactManager(temp_artifacts_dir)
        model = SimpleModel()

        # Save original artifact
        manager.save_model_artifact(model, sample_artifact)

        # Copy to new location
        target_dir = manager.copy_artifact("test_model", 1, "copied_model", 1)

        assert target_dir.exists()
        assert (target_dir / "model_state.pt").exists()
        assert (target_dir / "metadata.json").exists()

        # Check metadata was updated
        with open(target_dir / "metadata.json") as f:
            metadata = json.load(f)
            assert metadata["model_id"] == "copied_model"
            assert metadata["version"] == 1

    def test_export_artifact(self, temp_artifacts_dir, sample_artifact):
        """Test exporting artifact."""
        manager = ModelArtifactManager(temp_artifacts_dir)
        model = SimpleModel()

        # Save artifact
        manager.save_model_artifact(model, sample_artifact)

        # Export
        export_path = temp_artifacts_dir / "exported_model.pt"
        manager.export_artifact("test_model", 1, export_path, format="torch")

        assert export_path.exists()

        # Load exported file
        exported_data = torch.load(export_path)
        assert "model_state_dict" in exported_data
        assert "metadata" in exported_data

    def test_export_invalid_format(self, temp_artifacts_dir, sample_artifact):
        """Test exporting with invalid format."""
        manager = ModelArtifactManager(temp_artifacts_dir)
        model = SimpleModel()

        manager.save_model_artifact(model, sample_artifact)

        with pytest.raises(ValueError, match="Unsupported export format"):
            manager.export_artifact("test_model", 1, "export.txt", format="invalid")


class TestCreateArtifactFromTraining:
    """Test creating artifact from training results."""

    def test_create_basic_artifact(self, sample_training_result):
        """Test creating basic artifact from training."""
        config = TrainingConfig(epochs=5, learning_rate=0.01)

        artifact = create_artifact_from_training(
            model_id="test_model",
            model_name="Test Model",
            version=1,
            training_config=config,
            training_result=sample_training_result,
        )

        assert artifact.model_id == "test_model"
        assert artifact.training_config == config
        assert artifact.training_result == sample_training_result
        assert artifact.final_metrics["final_train_loss"] == 0.5
        assert artifact.final_metrics["final_val_loss"] == 0.3

    def test_create_artifact_with_model_info(self, sample_training_result):
        """Test creating artifact with model information."""
        config = TrainingConfig()
        model_info = {
            "model_class": "SimpleModel",
            "input_shape": [10],
            "output_shape": [2],
        }

        artifact = create_artifact_from_training(
            model_id="test_model",
            model_name="Test Model",
            version=1,
            training_config=config,
            training_result=sample_training_result,
            model_info=model_info,
        )

        assert artifact.model_class == "SimpleModel"
        assert artifact.input_shape == [10]
        assert artifact.output_shape == [2]

    def test_create_artifact_with_metrics_history(self):
        """Test creating artifact with metrics history."""
        result = TrainingResult(
            success=True,
            total_epochs=3,
            best_epoch=2,
            final_train_loss=0.3,
            metrics_history={
                "accuracy": [0.7, 0.8, 0.9],
                "val_loss": [0.8, 0.5, 0.3],
            },
        )

        config = TrainingConfig()

        artifact = create_artifact_from_training(
            model_id="test_model",
            model_name="Test Model",
            version=1,
            training_config=config,
            training_result=result,
        )

        assert "best_accuracy" in artifact.best_metrics
        assert "best_val_loss" in artifact.best_metrics
        assert artifact.best_metrics["best_accuracy"] == 0.9
        assert artifact.best_metrics["best_val_loss"] == 0.3
