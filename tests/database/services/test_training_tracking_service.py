"""Tests for TrainingTrackingService."""

import pytest

from arc.database import DatabaseManager
from arc.database.models.training import (
    CheckpointStatus,
    MetricType,
    TrainingStatus,
)
from arc.database.services.training_tracking_service import (
    TrainingTrackingService,
)


@pytest.fixture
def db_manager(tmp_path):
    """Create temporary file database manager for testing."""
    system_db = tmp_path / "system.db"
    user_db = tmp_path / "user.db"
    with DatabaseManager(str(system_db), str(user_db)) as manager:
        yield manager


@pytest.fixture
def tracking_service(db_manager):
    """Create a TrainingTrackingService instance for testing."""
    return TrainingTrackingService(db_manager)


class TestTrainingStatusEnum:
    """Test TrainingStatus enum."""

    def test_training_status_values(self):
        """Test TrainingStatus enum has expected values."""
        values = {s.value for s in TrainingStatus}
        assert values == {
            "pending",
            "running",
            "paused",
            "stopped",
            "completed",
            "failed",
        }

    def test_training_status_to_string(self):
        """Test TrainingStatus to_string conversion."""
        assert TrainingStatus.to_string(TrainingStatus.PENDING) == "pending"
        assert TrainingStatus.to_string(TrainingStatus.RUNNING) == "running"

    def test_training_status_from_string(self):
        """Test TrainingStatus from_string conversion."""
        assert TrainingStatus.from_string("pending") == TrainingStatus.PENDING
        assert TrainingStatus.from_string("completed") == TrainingStatus.COMPLETED

    def test_training_status_invalid_string(self):
        """Test TrainingStatus from_string with invalid value."""
        with pytest.raises(ValueError, match="Invalid training status"):
            TrainingStatus.from_string("invalid")


class TestCheckpointStatusEnum:
    """Test CheckpointStatus enum."""

    def test_checkpoint_status_values(self):
        """Test CheckpointStatus enum has expected values."""
        values = {s.value for s in CheckpointStatus}
        assert values == {"saved", "deleted", "corrupted"}

    def test_checkpoint_status_conversions(self):
        """Test CheckpointStatus conversions."""
        assert CheckpointStatus.to_string(CheckpointStatus.SAVED) == "saved"
        assert CheckpointStatus.from_string("deleted") == CheckpointStatus.DELETED


class TestMetricTypeEnum:
    """Test MetricType enum."""

    def test_metric_type_values(self):
        """Test MetricType enum has expected values."""
        values = {t.value for t in MetricType}
        assert values == {"train", "validation", "test"}

    def test_metric_type_conversions(self):
        """Test MetricType conversions."""
        assert MetricType.to_string(MetricType.TRAIN) == "train"
        assert MetricType.from_string("validation") == MetricType.VALIDATION


class TestTrainingRunOperations:
    """Test TrainingRun CRUD operations."""

    def test_create_run_minimal(self, tracking_service):
        """Test creating a minimal training run."""
        run = tracking_service.create_run()

        assert run.run_id is not None
        assert run.status == TrainingStatus.PENDING
        assert run.tensorboard_enabled is True
        assert run.metric_log_frequency == 100
        assert run.checkpoint_frequency == 5
        assert run.created_at is not None
        assert run.updated_at is not None

    def test_create_run_with_config(self, tracking_service):
        """Test creating a training run with full configuration."""
        config = {
            "epochs": 50,
            "batch_size": 32,
            "learning_rate": 0.001,
        }

        run = tracking_service.create_run(
            job_id="job-123",
            model_id="model-456",
            run_name="Test Run",
            description="Test training run",
            tensorboard_enabled=True,
            tensorboard_log_dir="/tmp/tb_logs",
            metric_log_frequency=50,
            checkpoint_frequency=10,
            config=config,
        )

        assert run.job_id == "job-123"
        assert run.model_id == "model-456"
        assert run.run_name == "Test Run"
        assert run.description == "Test training run"
        assert run.tensorboard_log_dir == "/tmp/tb_logs"
        assert run.metric_log_frequency == 50
        assert run.checkpoint_frequency == 10
        assert run.training_config is not None
        assert "epochs" in run.training_config

    def test_get_run_by_id(self, tracking_service):
        """Test retrieving a run by ID."""
        created_run = tracking_service.create_run(run_name="Test Get")

        retrieved_run = tracking_service.get_run_by_id(created_run.run_id)

        assert retrieved_run is not None
        assert retrieved_run.run_id == created_run.run_id
        assert retrieved_run.run_name == "Test Get"

    def test_get_run_by_id_not_found(self, tracking_service):
        """Test retrieving non-existent run returns None."""
        result = tracking_service.get_run_by_id("nonexistent-run-id")
        assert result is None

    def test_get_run_by_job_id(self, tracking_service):
        """Test retrieving a run by job ID."""
        created_run = tracking_service.create_run(job_id="job-abc")

        retrieved_run = tracking_service.get_run_by_job_id("job-abc")

        assert retrieved_run is not None
        assert retrieved_run.job_id == "job-abc"
        assert retrieved_run.run_id == created_run.run_id

    def test_list_runs(self, tracking_service):
        """Test listing all runs."""
        run1 = tracking_service.create_run(run_name="Run 1")
        run2 = tracking_service.create_run(run_name="Run 2")

        runs = tracking_service.list_runs()

        assert len(runs) >= 2
        run_ids = {r.run_id for r in runs}
        assert run1.run_id in run_ids
        assert run2.run_id in run_ids

    def test_list_runs_with_status_filter(self, tracking_service):
        """Test listing runs filtered by status."""
        run1 = tracking_service.create_run(run_name="Pending Run")
        run2 = tracking_service.create_run(run_name="Running Run")

        # Update run2 status to running
        tracking_service.update_run_status(run2.run_id, TrainingStatus.RUNNING)

        # List only running runs
        running_runs = tracking_service.list_runs(status=TrainingStatus.RUNNING)

        assert len(running_runs) == 1
        assert running_runs[0].run_id == run2.run_id

        # List only pending runs
        pending_runs = tracking_service.list_runs(status=TrainingStatus.PENDING)
        pending_ids = {r.run_id for r in pending_runs}
        assert run1.run_id in pending_ids

    def test_list_runs_with_model_filter(self, tracking_service):
        """Test listing runs filtered by model ID."""
        run1 = tracking_service.create_run(model_id="model-1")
        tracking_service.create_run(model_id="model-2")

        model1_runs = tracking_service.list_runs(model_id="model-1")

        assert len(model1_runs) == 1
        assert model1_runs[0].run_id == run1.run_id

    def test_list_runs_with_limit(self, tracking_service):
        """Test listing runs with limit."""
        for i in range(5):
            tracking_service.create_run(run_name=f"Run {i}")

        runs = tracking_service.list_runs(limit=3)

        assert len(runs) == 3

    def test_update_run_status(self, tracking_service):
        """Test updating run status."""
        run = tracking_service.create_run()

        tracking_service.update_run_status(run.run_id, TrainingStatus.RUNNING)

        updated_run = tracking_service.get_run_by_id(run.run_id)
        assert updated_run.status == TrainingStatus.RUNNING

    def test_update_run_status_with_timestamp(self, tracking_service):
        """Test updating run status with timestamp field."""
        run = tracking_service.create_run()

        tracking_service.update_run_status(
            run.run_id, TrainingStatus.RUNNING, timestamp_field="started_at"
        )

        updated_run = tracking_service.get_run_by_id(run.run_id)
        assert updated_run.status == TrainingStatus.RUNNING
        assert updated_run.started_at is not None

    def test_delete_run(self, tracking_service):
        """Test deleting a run."""
        run = tracking_service.create_run()

        result = tracking_service.delete_run(run.run_id)
        assert result is True

        # Verify run is deleted
        deleted_run = tracking_service.get_run_by_id(run.run_id)
        assert deleted_run is None

    def test_delete_run_not_found(self, tracking_service):
        """Test deleting non-existent run returns False."""
        result = tracking_service.delete_run("nonexistent-run-id")
        assert result is False


class TestTrainingMetricOperations:
    """Test TrainingMetric operations."""

    def test_log_metric(self, tracking_service):
        """Test logging a training metric."""
        run = tracking_service.create_run()

        metric = tracking_service.log_metric(
            run_id=run.run_id,
            metric_name="loss",
            metric_type=MetricType.TRAIN,
            step=100,
            epoch=1,
            value=0.5,
        )

        assert metric.metric_id is not None
        assert metric.run_id == run.run_id
        assert metric.metric_name == "loss"
        assert metric.metric_type == MetricType.TRAIN
        assert metric.step == 100
        assert metric.epoch == 1
        assert metric.value == 0.5
        assert metric.timestamp is not None

    def test_get_metrics(self, tracking_service):
        """Test retrieving metrics for a run."""
        run = tracking_service.create_run()

        # Log multiple metrics
        tracking_service.log_metric(run.run_id, "loss", MetricType.TRAIN, 100, 1, 0.5)
        tracking_service.log_metric(
            run.run_id, "accuracy", MetricType.TRAIN, 100, 1, 0.8
        )

        metrics = tracking_service.get_metrics(run.run_id)

        assert len(metrics) == 2
        metric_names = {m.metric_name for m in metrics}
        assert "loss" in metric_names
        assert "accuracy" in metric_names

    def test_get_metrics_with_name_filter(self, tracking_service):
        """Test retrieving metrics filtered by name."""
        run = tracking_service.create_run()

        tracking_service.log_metric(run.run_id, "loss", MetricType.TRAIN, 100, 1, 0.5)
        tracking_service.log_metric(
            run.run_id, "accuracy", MetricType.TRAIN, 100, 1, 0.8
        )

        loss_metrics = tracking_service.get_metrics(run.run_id, metric_name="loss")

        assert len(loss_metrics) == 1
        assert loss_metrics[0].metric_name == "loss"

    def test_get_metrics_with_type_filter(self, tracking_service):
        """Test retrieving metrics filtered by type."""
        run = tracking_service.create_run()

        tracking_service.log_metric(run.run_id, "loss", MetricType.TRAIN, 100, 1, 0.5)
        tracking_service.log_metric(
            run.run_id, "loss", MetricType.VALIDATION, 100, 1, 0.6
        )

        train_metrics = tracking_service.get_metrics(
            run.run_id, metric_type=MetricType.TRAIN
        )

        assert len(train_metrics) == 1
        assert train_metrics[0].metric_type == MetricType.TRAIN

    def test_get_metrics_with_limit(self, tracking_service):
        """Test retrieving metrics with limit."""
        run = tracking_service.create_run()

        for i in range(10):
            tracking_service.log_metric(
                run.run_id, "loss", MetricType.TRAIN, i * 100, 1, 0.5 - i * 0.01
            )

        metrics = tracking_service.get_metrics(run.run_id, limit=5)

        assert len(metrics) == 5

    def test_get_latest_metric(self, tracking_service):
        """Test retrieving latest metric value."""
        run = tracking_service.create_run()

        # Log multiple metric values
        tracking_service.log_metric(run.run_id, "loss", MetricType.TRAIN, 100, 1, 0.5)
        tracking_service.log_metric(run.run_id, "loss", MetricType.TRAIN, 200, 1, 0.4)
        tracking_service.log_metric(run.run_id, "loss", MetricType.TRAIN, 300, 1, 0.3)

        latest = tracking_service.get_latest_metric(
            run.run_id, "loss", MetricType.TRAIN
        )

        assert latest is not None
        assert latest.step == 300
        assert latest.value == 0.3

    def test_get_latest_metric_not_found(self, tracking_service):
        """Test retrieving latest metric when none exist."""
        run = tracking_service.create_run()

        latest = tracking_service.get_latest_metric(
            run.run_id, "nonexistent", MetricType.TRAIN
        )

        assert latest is None


class TestTrainingCheckpointOperations:
    """Test TrainingCheckpoint operations."""

    def test_create_checkpoint(self, tracking_service):
        """Test creating a checkpoint."""
        run = tracking_service.create_run()

        metrics = {"loss": 0.3, "accuracy": 0.85}
        checkpoint = tracking_service.create_checkpoint(
            run_id=run.run_id,
            epoch=5,
            step=500,
            checkpoint_path="/tmp/checkpoint_epoch5.pt",
            metrics=metrics,
            is_best=True,
            file_size_bytes=1024000,
        )

        assert checkpoint.checkpoint_id is not None
        assert checkpoint.run_id == run.run_id
        assert checkpoint.epoch == 5
        assert checkpoint.step == 500
        assert checkpoint.checkpoint_path == "/tmp/checkpoint_epoch5.pt"
        assert checkpoint.is_best is True
        assert checkpoint.file_size_bytes == 1024000
        assert checkpoint.status == CheckpointStatus.SAVED
        assert "loss" in checkpoint.metrics

    def test_get_checkpoints(self, tracking_service):
        """Test retrieving all checkpoints for a run."""
        run = tracking_service.create_run()

        # Create multiple checkpoints
        tracking_service.create_checkpoint(run.run_id, 1, 100, "/tmp/checkpoint1.pt")
        tracking_service.create_checkpoint(run.run_id, 2, 200, "/tmp/checkpoint2.pt")

        checkpoints = tracking_service.get_checkpoints(run.run_id)

        assert len(checkpoints) == 2

    def test_get_best_checkpoint(self, tracking_service):
        """Test retrieving best checkpoint."""
        run = tracking_service.create_run()

        # Create regular checkpoint
        tracking_service.create_checkpoint(
            run.run_id, 1, 100, "/tmp/checkpoint1.pt", is_best=False
        )

        # Create best checkpoint
        best_cp = tracking_service.create_checkpoint(
            run.run_id, 5, 500, "/tmp/checkpoint5.pt", is_best=True
        )

        retrieved_best = tracking_service.get_best_checkpoint(run.run_id)

        assert retrieved_best is not None
        assert retrieved_best.checkpoint_id == best_cp.checkpoint_id
        assert retrieved_best.is_best is True

    def test_get_best_checkpoint_not_found(self, tracking_service):
        """Test retrieving best checkpoint when none exists."""
        run = tracking_service.create_run()

        best = tracking_service.get_best_checkpoint(run.run_id)
        assert best is None

    def test_update_checkpoint_status(self, tracking_service):
        """Test updating checkpoint status."""
        run = tracking_service.create_run()
        checkpoint = tracking_service.create_checkpoint(
            run.run_id, 1, 100, "/tmp/checkpoint.pt"
        )

        tracking_service.update_checkpoint_status(
            checkpoint.checkpoint_id, CheckpointStatus.DELETED
        )

        checkpoints = tracking_service.get_checkpoints(run.run_id)
        assert checkpoints[0].status == CheckpointStatus.DELETED

    def test_mark_checkpoint_as_best(self, tracking_service):
        """Test marking checkpoint as best."""
        run = tracking_service.create_run()

        # Create two checkpoints
        tracking_service.create_checkpoint(
            run.run_id, 1, 100, "/tmp/cp1.pt", is_best=True
        )
        cp2 = tracking_service.create_checkpoint(
            run.run_id, 2, 200, "/tmp/cp2.pt", is_best=False
        )

        # Mark cp2 as best
        tracking_service.mark_checkpoint_as_best(cp2.checkpoint_id, run.run_id)

        # Verify cp2 is now best
        best = tracking_service.get_best_checkpoint(run.run_id)
        assert best.checkpoint_id == cp2.checkpoint_id

    def test_delete_run_cascades_to_metrics_and_checkpoints(self, tracking_service):
        """Test that deleting a run cascades to metrics and checkpoints."""
        run = tracking_service.create_run()

        # Create metrics and checkpoints
        tracking_service.log_metric(run.run_id, "loss", MetricType.TRAIN, 100, 1, 0.5)
        tracking_service.create_checkpoint(run.run_id, 1, 100, "/tmp/checkpoint.pt")

        # Delete run
        tracking_service.delete_run(run.run_id)

        # Verify metrics and checkpoints are also deleted
        metrics = tracking_service.get_metrics(run.run_id)
        checkpoints = tracking_service.get_checkpoints(run.run_id)

        assert len(metrics) == 0
        assert len(checkpoints) == 0
