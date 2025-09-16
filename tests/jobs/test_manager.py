"""Tests for JobManager operations."""

import pytest

from src.arc.database.manager import DatabaseManager
from src.arc.jobs.manager import JobManager
from src.arc.jobs.models import JobStatus, JobType


@pytest.fixture
def db_manager():
    """Use in-memory DuckDB for isolated manager tests."""
    return DatabaseManager(":memory:", ":memory:")


@pytest.fixture
def job_manager(db_manager):
    return JobManager(db_manager)


def test_cancel_job_success(job_manager):
    """Cancelling a pending job should succeed and mark it cancelled."""
    job = job_manager.create_job(JobType.TRAIN_MODEL, model_id=1, message="go")
    assert job.status == JobStatus.PENDING

    assert job_manager.cancel_job(job.job_id) is True

    updated = job_manager.get_job(job.job_id)
    assert updated is not None
    assert updated.status == JobStatus.CANCELLED
    assert "Cancelled" in updated.message


def test_cancel_job_running(job_manager):
    """Cancelling a running job should succeed."""
    job = job_manager.create_job(JobType.PREDICT_MODEL, model_id=2, message="run")
    job.start("started")
    job_manager.update_job(job)

    assert job_manager.cancel_job(job.job_id) is True
    assert job_manager.get_job(job.job_id).status == JobStatus.CANCELLED


def test_cancel_job_not_cancellable(job_manager):
    """Completed jobs should not be cancellable."""
    job = job_manager.create_job(JobType.VALIDATE_SCHEMA, message="done soon")
    # Move through RUNNING -> COMPLETED to satisfy model state transitions
    job.start("started")
    job_manager.update_job(job)
    job.complete("finished")
    job_manager.update_job(job)

    assert job_manager.cancel_job(job.job_id) is False
    assert job_manager.get_job(job.job_id).status == JobStatus.COMPLETED


def test_cancel_job_not_found(job_manager):
    """Cancelling a non-existent job returns False."""
    assert job_manager.cancel_job("nonexistent-job") is False
