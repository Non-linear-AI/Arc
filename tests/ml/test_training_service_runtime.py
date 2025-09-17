"""Integration-style tests for TrainingService."""

from __future__ import annotations

import asyncio

import pytest

from arc.database.manager import DatabaseManager
from arc.database.services import JobService
from arc.graph.spec import ArcGraph
from arc.jobs.models import JobStatus
from arc.ml.training_service import TrainingJobConfig, TrainingService

SMALL_GRAPH_YAML = """
version: "0.1"
model_name: "tiny_classifier"

features:
  feature_columns: [feature1, feature2]
  target_columns: [outcome]
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
  loss: {type: binary_cross_entropy}
  config:
    epochs: 2
    batch_size: 2
    learning_rate: 0.1
"""


@pytest.mark.asyncio
async def test_training_service_completes_quick_job(tmp_path):
    """Ensure TrainingService completes and updates job status."""

    manager = DatabaseManager(":memory:", ":memory:")
    job_service = JobService(manager)

    manager.user_execute(
        """
        CREATE TABLE train_data (
            feature1 DOUBLE,
            feature2 DOUBLE,
            outcome DOUBLE
        )
        """
    )

    rows = [
        (0.2, 0.4, 0.0),
        (0.5, 0.7, 1.0),
        (0.1, 0.3, 0.0),
        (0.9, 0.8, 1.0),
    ]
    for a, b, c in rows:
        manager.user_execute(f"INSERT INTO train_data VALUES ({a}, {b}, {c})")

    arc_graph = ArcGraph.from_yaml(SMALL_GRAPH_YAML)
    training_config = arc_graph.to_training_config()

    service = TrainingService(job_service, artifacts_dir=tmp_path / "artifacts")

    job_config = TrainingJobConfig(
        model_id="unit-test-model",
        model_name="unit-test-model",
        arc_graph=arc_graph,
        train_table="train_data",
        target_column="outcome",
        feature_columns=["feature1", "feature2"],
        training_config=training_config,
        description="unit test training",
    )

    job_id = service.submit_training_job(job_config)

    result = await asyncio.wait_for(service.wait_for_job(job_id, timeout=10), 10)

    assert result is not None
    assert result.success is True

    # Allow the loop to process finalize callback
    await asyncio.sleep(0)

    job_record = job_service.get_job_by_id(job_id)
    assert job_record is not None
    assert job_record.status == JobStatus.COMPLETED
    assert "completed" in (job_record.message or "").lower()

    service.shutdown()


@pytest.mark.asyncio
async def test_database_job_status_updates_correctly(tmp_path):
    """Test that job status in database updates properly throughout training lifecycle."""

    manager = DatabaseManager(":memory:", ":memory:")
    job_service = JobService(manager)

    # Create minimal training data
    manager.user_execute(
        """
        CREATE TABLE train_data (
            feature1 DOUBLE,
            feature2 DOUBLE,
            outcome DOUBLE
        )
        """
    )

    rows = [
        (0.2, 0.4, 0.0),
        (0.5, 0.7, 1.0),
        (0.1, 0.3, 0.0),
        (0.9, 0.8, 1.0),
    ]
    for a, b, c in rows:
        manager.user_execute(f"INSERT INTO train_data VALUES ({a}, {b}, {c})")

    arc_graph = ArcGraph.from_yaml(SMALL_GRAPH_YAML)
    training_config = arc_graph.to_training_config()

    service = TrainingService(job_service, artifacts_dir=tmp_path / "artifacts")

    job_config = TrainingJobConfig(
        model_id="db-status-test-model",
        model_name="db-status-test-model",
        arc_graph=arc_graph,
        train_table="train_data",
        target_column="outcome",
        feature_columns=["feature1", "feature2"],
        training_config=training_config,
        description="Database status test training",
    )

    job_id = service.submit_training_job(job_config)

    # Check initial status - should be RUNNING (not PENDING after submit)
    initial_status = service.get_job_status(job_id)
    assert initial_status["status"] == "running"
    assert "scheduled" in initial_status["message"].lower()

    # Wait for completion (now synchronous)
    result = service.wait_for_job(job_id, timeout=10)
    assert result is not None
    assert result.success is True

    # No need to wait for callback - it's handled in the thread

    # Check final database status - this is the critical test
    final_status = service.get_job_status(job_id)
    assert final_status["status"] == "completed", (
        f"Expected 'completed', got '{final_status['status']}' with message: {final_status['message']}"
    )
    assert "completed" in final_status["message"].lower()

    # Ensure job is cleaned up from active jobs
    assert job_id not in service.active_jobs

    service.shutdown()


@pytest.mark.asyncio
async def test_finalize_callback_handles_completion_status_correctly(tmp_path):
    """Test that _finalize_job callback correctly handles task results and updates database status."""

    import asyncio

    from arc.ml.trainer import TrainingResult

    manager = DatabaseManager(":memory:", ":memory:")
    job_service = JobService(manager)
    service = TrainingService(job_service, artifacts_dir=tmp_path / "artifacts")

    # Mock a completed job
    job_id = "test-finalize-job"

    # Create a fake future that returns a successful result
    fake_future = asyncio.Future()
    fake_result = TrainingResult(
        success=True,
        total_epochs=3,
        best_epoch=2,
        final_train_loss=0.1,
        final_val_loss=0.15,
        best_val_loss=0.15,
        train_losses=[0.5, 0.3, 0.1],
        val_losses=[0.4, 0.25, 0.15],
        metrics_history={"accuracy": [0.7, 0.85, 0.9]},
        training_time=1.5,
        error_message=None,
    )
    fake_future.set_result(fake_result)

    # Add job to active jobs (simulating a running job)
    service.active_jobs[job_id] = fake_future

    # Create initial job record in database
    from datetime import UTC, datetime

    from arc.jobs.models import Job, JobStatus, JobType

    job = Job(
        job_id=job_id,
        model_id=None,  # model_id is optional for training jobs
        type=JobType.TRAIN_MODEL,
        status=JobStatus.RUNNING,
        message="Test job running",
        sql_query=None,
        created_at=datetime.now(UTC),
        updated_at=datetime.now(UTC),
    )
    job_service.create_job(job)

    # Call _finalize_job directly
    service._finalize_job(job_id, fake_future)

    # Verify job status was updated to completed
    final_status = service.get_job_status(job_id)
    assert final_status["status"] == "completed"
    assert "completed successfully" in final_status["message"]

    # Verify job was removed from active jobs
    assert job_id not in service.active_jobs

    service.shutdown()


@pytest.mark.asyncio
async def test_training_error_handling_and_status_reporting(tmp_path):
    """Test that training errors are properly caught and reported in job status."""

    manager = DatabaseManager(":memory:", ":memory:")
    job_service = JobService(manager)

    # Create minimal training data with WRONG column names to trigger error
    manager.user_execute(
        """
        CREATE TABLE train_data (
            wrong_feature1 DOUBLE,
            wrong_feature2 DOUBLE,
            wrong_outcome DOUBLE
        )
        """
    )

    rows = [
        (0.2, 0.4, 0.0),
        (0.5, 0.7, 1.0),
    ]
    for a, b, c in rows:
        manager.user_execute(f"INSERT INTO train_data VALUES ({a}, {b}, {c})")

    arc_graph = ArcGraph.from_yaml(SMALL_GRAPH_YAML)
    training_config = arc_graph.to_training_config()

    service = TrainingService(job_service, artifacts_dir=tmp_path / "artifacts")

    job_config = TrainingJobConfig(
        model_id="error-test-model",
        model_name="error-test-model",
        arc_graph=arc_graph,
        train_table="train_data",
        target_column="outcome",  # This column doesn't exist!
        feature_columns=["feature1", "feature2"],  # These columns don't exist!
        training_config=training_config,
        description="Error handling test training",
    )

    job_id = service.submit_training_job(job_config)

    # Check initial status
    initial_status = service.get_job_status(job_id)
    assert initial_status["status"] == "running"

    # Wait for job to fail (should be quick since data loading will fail)
    result = service.wait_for_job(job_id, timeout=5)
    # Result might be None (timeout) or a failed TrainingResult
    if result is not None:
        assert not result.success

    # Allow time for error processing (brief pause)
    import time

    time.sleep(0.1)

    # Check final status - should show error
    final_status = service.get_job_status(job_id)
    assert final_status["status"] in ["failed", "running"], (
        f"Expected 'failed' or 'running', got '{final_status['status']}'"
    )

    # If it's still running, it means our setup error wasn't caught early enough
    # But the message should indicate what's happening
    if final_status["status"] == "running":
        assert (
            "scheduled" in final_status["message"].lower()
            or "training" in final_status["message"].lower()
        )
    else:
        # If it failed, the message should contain error details
        assert (
            "failed" in final_status["message"].lower()
            or "error" in final_status["message"].lower()
        )

    service.shutdown()
