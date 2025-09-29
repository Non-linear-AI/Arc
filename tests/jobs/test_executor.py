"""Edge-case tests for JobExecutor behavior."""

import pytest

from arc.database.manager import DatabaseManager
from arc.database.services.job_service import JobService
from arc.jobs.executor import JobExecutor
from arc.jobs.manager import JobManager


@pytest.fixture
def db_manager(tmp_path):
    """Use file-based DuckDB for thread-safe testing."""
    system_db = tmp_path / "system.db"
    user_db = tmp_path / "user.db"
    return DatabaseManager(str(system_db), str(user_db))


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
