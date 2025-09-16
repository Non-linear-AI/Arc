"""Edge-case tests for JobExecutor behavior."""

import asyncio

import pytest

from src.arc.database.manager import DatabaseManager
from src.arc.database.services.job_service import JobService
from src.arc.jobs.executor import JobExecutor
from src.arc.jobs.manager import JobManager
from src.arc.jobs.models import Job, JobStatus, JobType


@pytest.fixture
def db_manager():
    """Use in-memory DuckDB for fast, isolated tests."""
    return DatabaseManager(":memory:", ":memory:")


@pytest.fixture
def job_manager(db_manager):
    service = JobService(db_manager)
    return JobManager(service)


@pytest.fixture
def executor(job_manager):
    ex = JobExecutor(job_manager, max_workers=2)
    yield ex
    if ex.is_running:
        ex.stop()


def test_start_is_idempotent(executor, caplog):
    """Calling start twice should warn and remain running without error."""
    executor.start()
    assert executor.is_running

    with caplog.at_level("WARNING"):
        executor.start()

    assert executor.is_running
    assert any("already running" in rec.message for rec in caplog.records)


def test_stop_when_not_running_noop(executor):
    """Stopping a non-running executor should be a no-op."""
    assert not executor.is_running
    # Should not raise
    executor.stop()
    assert not executor.is_running


def test_wait_for_unknown_job_returns_quickly(executor):
    """wait_for_job on unknown ID should not block when not running."""
    # Not started; should return without blocking
    executor.wait_for_job("non-existent-id")


def test_async_failure_marks_job_failed(executor):
    """Async handler exceptions should mark the job as FAILED with message."""

    async def failing(job: Job) -> None:
        job.update_message("about to fail")
        await asyncio.sleep(0)
        raise RuntimeError("boom")

    executor.register_async_handler(JobType.PREDICT_MODEL, failing)
    executor.start()

    job = executor.submit_job(JobType.PREDICT_MODEL, model_id=7, message="f")
    executor.wait_for_job(job.job_id)

    final = executor.job_manager.get_job(job.job_id)
    assert final.status == JobStatus.FAILED
    assert "boom" in final.message
