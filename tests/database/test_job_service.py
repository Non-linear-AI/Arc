"""Tests for JobService."""

from datetime import UTC, datetime, timedelta

import pytest

from arc.database import DatabaseManager
from arc.database.services.job_service import (
    Job,
    JobService,
    JobStatus,
)


@pytest.fixture
def db_manager():
    """Create an in-memory database manager for testing."""
    with DatabaseManager(":memory:") as manager:
        yield manager


@pytest.fixture
def job_service(db_manager):
    """Create a JobService instance for testing."""
    return JobService(db_manager)


@pytest.fixture
def sample_job():
    """Create a sample job for testing."""
    now = datetime.now(UTC)
    return Job(
        job_id="test-job-1",
        model_id=42,
        type="training",
        status=JobStatus.PENDING,
        message="Job initialized",
        sql_query="SELECT * FROM data WHERE id > 100",
        created_at=now,
        updated_at=now,
    )


@pytest.fixture
def sample_jobs():
    """Create multiple sample jobs for testing."""
    now = datetime.now(UTC)
    return [
        Job(
            job_id="job-1",
            model_id=1,
            type="training",
            status=JobStatus.PENDING,
            message="Training job pending",
            sql_query="SELECT * FROM training_data",
            created_at=now,
            updated_at=now,
        ),
        Job(
            job_id="job-2",
            model_id=2,
            type="inference",
            status=JobStatus.RUNNING,
            message="Inference job running",
            sql_query=None,  # No SQL query
            created_at=now,
            updated_at=now,
        ),
        Job(
            job_id="job-3",
            model_id=None,  # No model ID
            type="cleanup",
            status=JobStatus.COMPLETED,
            message="Cleanup completed successfully",
            sql_query=None,
            created_at=now - timedelta(hours=1),
            updated_at=now - timedelta(minutes=30),
        ),
    ]


class TestJobStatusEnum:
    """Test the JobStatus enum and utility functions."""

    def test_job_status_enum_values(self):
        """Test JobStatus enum has correct values."""
        assert JobStatus.UNKNOWN.value == "UNKNOWN"
        assert JobStatus.PENDING.value == "PENDING"
        assert JobStatus.RUNNING.value == "RUNNING"
        assert JobStatus.COMPLETED.value == "COMPLETED"
        assert JobStatus.FAILED.value == "FAILED"
        assert JobStatus.CANCELLED.value == "CANCELLED"

    def test_job_status_to_string(self):
        """Test converting JobStatus to string."""
        assert JobStatus.job_status_to_string(JobStatus.PENDING) == "PENDING"
        assert JobStatus.job_status_to_string(JobStatus.RUNNING) == "RUNNING"
        assert JobStatus.job_status_to_string(JobStatus.COMPLETED) == "COMPLETED"

    def test_string_to_job_status(self):
        """Test converting string to JobStatus."""
        assert JobStatus.string_to_job_status("PENDING") == JobStatus.PENDING
        assert JobStatus.string_to_job_status("RUNNING") == JobStatus.RUNNING
        assert JobStatus.string_to_job_status("COMPLETED") == JobStatus.COMPLETED

    def test_string_to_job_status_invalid(self):
        """Test converting invalid string to JobStatus raises ValueError."""
        with pytest.raises(ValueError, match="Invalid job status: INVALID"):
            JobStatus.string_to_job_status("INVALID")


class TestJobDataClass:
    """Test the Job dataclass."""

    def test_job_creation(self, sample_job):
        """Test creating a Job instance."""
        assert sample_job.job_id == "test-job-1"
        assert sample_job.model_id == 42
        assert sample_job.type == "training"
        assert sample_job.status == JobStatus.PENDING
        assert sample_job.message == "Job initialized"
        assert sample_job.sql_query == "SELECT * FROM data WHERE id > 100"
        assert isinstance(sample_job.created_at, datetime)
        assert isinstance(sample_job.updated_at, datetime)

    def test_job_with_optional_fields_none(self):
        """Test creating a Job with None optional fields."""
        now = datetime.now(UTC)
        job = Job(
            job_id="optional-job",
            model_id=None,  # Optional field
            type="test",
            status=JobStatus.UNKNOWN,
            message="Test job",
            sql_query=None,  # Optional field
            created_at=now,
            updated_at=now,
        )
        assert job.model_id is None
        assert job.sql_query is None


class TestJobServiceCRUD:
    """Test CRUD operations in JobService."""

    def test_create_job(self, job_service, sample_job):
        """Test creating a job."""
        # Create job
        job_service.create_job(sample_job)

        # Verify job was created
        retrieved = job_service.get_job_by_id(sample_job.job_id)
        assert retrieved is not None
        assert retrieved.job_id == sample_job.job_id
        assert retrieved.type == sample_job.type
        assert retrieved.status == sample_job.status

    def test_get_job_by_id(self, job_service, sample_job):
        """Test getting a job by ID."""
        # Job doesn't exist initially
        result = job_service.get_job_by_id(sample_job.job_id)
        assert result is None

        # Create job
        job_service.create_job(sample_job)

        # Get job by ID
        result = job_service.get_job_by_id(sample_job.job_id)
        assert result is not None
        assert result.job_id == sample_job.job_id
        assert result.model_id == sample_job.model_id
        assert result.type == sample_job.type
        assert result.status == sample_job.status
        assert result.message == sample_job.message
        assert result.sql_query == sample_job.sql_query

    def test_update_job_status(self, job_service, sample_job):
        """Test updating job status."""
        # Create job
        job_service.create_job(sample_job)

        # Update status
        new_status = JobStatus.RUNNING
        new_message = "Job started running"
        job_service.update_job_status(sample_job.job_id, new_status, new_message)

        # Verify update
        updated_job = job_service.get_job_by_id(sample_job.job_id)
        assert updated_job is not None
        assert updated_job.status == new_status
        assert updated_job.message == new_message
        # updated_at should be more recent
        assert updated_job.updated_at >= sample_job.updated_at

    def test_cancel_job_success(self, job_service, sample_job):
        """Test successfully cancelling a job."""
        # Create job with PENDING status
        sample_job.status = JobStatus.PENDING
        job_service.create_job(sample_job)

        # Cancel job
        result = job_service.cancel_job(sample_job.job_id)
        assert result is True

        # Verify cancellation
        cancelled_job = job_service.get_job_by_id(sample_job.job_id)
        assert cancelled_job is not None
        assert cancelled_job.status == JobStatus.CANCELLED
        assert cancelled_job.message == "Cancelled by user"

    def test_cancel_job_running(self, job_service, sample_job):
        """Test cancelling a running job."""
        # Create job with RUNNING status
        sample_job.status = JobStatus.RUNNING
        job_service.create_job(sample_job)

        # Cancel job
        result = job_service.cancel_job(sample_job.job_id)
        assert result is True

        # Verify cancellation
        cancelled_job = job_service.get_job_by_id(sample_job.job_id)
        assert cancelled_job.status == JobStatus.CANCELLED

    def test_cancel_job_not_cancellable(self, job_service, sample_job):
        """Test cancelling a job that cannot be cancelled."""
        # Create job with COMPLETED status
        sample_job.status = JobStatus.COMPLETED
        job_service.create_job(sample_job)

        # Try to cancel job
        result = job_service.cancel_job(sample_job.job_id)
        assert result is False

        # Verify job status unchanged
        job = job_service.get_job_by_id(sample_job.job_id)
        assert job.status == JobStatus.COMPLETED

    def test_cancel_job_not_found(self, job_service):
        """Test cancelling a non-existent job."""
        result = job_service.cancel_job("nonexistent-job")
        assert result is False


class TestJobServiceQueries:
    """Test job query operations in JobService."""

    def test_get_recent_jobs(self, job_service, sample_jobs):
        """Test getting recent jobs."""
        # Create jobs
        for job in sample_jobs:
            job_service.create_job(job)

        # Get recent jobs
        recent_jobs = job_service.get_recent_jobs(2)
        assert len(recent_jobs) == 2

        # Should be ordered by created_at DESC
        job_ids = {job.job_id for job in recent_jobs}
        # Most recent jobs should be included
        assert len(job_ids) == 2

    def test_get_recent_jobs_limit(self, job_service, sample_jobs):
        """Test limit parameter in get_recent_jobs."""
        # Create jobs
        for job in sample_jobs:
            job_service.create_job(job)

        # Get only 1 recent job
        recent_jobs = job_service.get_recent_jobs(1)
        assert len(recent_jobs) == 1

        # Get more than available
        recent_jobs = job_service.get_recent_jobs(10)
        assert len(recent_jobs) == 3  # Only 3 jobs exist

    def test_get_jobs_by_status(self, job_service, sample_jobs):
        """Test getting jobs by status."""
        # Create jobs
        for job in sample_jobs:
            job_service.create_job(job)

        # Get pending jobs
        pending_jobs = job_service.get_jobs_by_status(JobStatus.PENDING)
        assert len(pending_jobs) == 1
        assert pending_jobs[0].job_id == "job-1"

        # Get running jobs
        running_jobs = job_service.get_jobs_by_status(JobStatus.RUNNING)
        assert len(running_jobs) == 1
        assert running_jobs[0].job_id == "job-2"

        # Get completed jobs
        completed_jobs = job_service.get_jobs_by_status(JobStatus.COMPLETED)
        assert len(completed_jobs) == 1
        assert completed_jobs[0].job_id == "job-3"

        # Get non-existent status
        failed_jobs = job_service.get_jobs_by_status(JobStatus.FAILED)
        assert len(failed_jobs) == 0

    def test_get_job_status(self, job_service, sample_job):
        """Test getting job status."""
        # Create job
        job_service.create_job(sample_job)

        # Get status
        status = job_service.get_job_status(sample_job.job_id)
        assert status == sample_job.status

    def test_get_job_status_not_found(self, job_service):
        """Test getting status of non-existent job."""
        with pytest.raises(Exception, match="Job not found: nonexistent"):
            job_service.get_job_status("nonexistent")


class TestJobServiceUtilities:
    """Test utility methods in JobService."""

    def test_job_exists(self, job_service, sample_job):
        """Test checking if a job exists."""
        # Job doesn't exist initially
        assert not job_service.job_exists(sample_job.job_id)

        # Create job
        job_service.create_job(sample_job)

        # Job now exists
        assert job_service.job_exists(sample_job.job_id)

    def test_cleanup_old_jobs(self, job_service, sample_jobs):
        """Test cleaning up old jobs."""
        # Create jobs
        for job in sample_jobs:
            job_service.create_job(job)

        # All jobs should exist
        assert len(job_service.get_recent_jobs(10)) == 3

        # Cleanup jobs older than 30 minutes ago
        cutoff = datetime.now(UTC) - timedelta(minutes=30)
        job_service.cleanup_old_jobs(cutoff)

        # Should have fewer jobs now
        remaining_jobs = job_service.get_recent_jobs(10)
        # job-3 was created 1 hour ago, so it should be deleted
        remaining_ids = {job.job_id for job in remaining_jobs}
        assert "job-3" not in remaining_ids or len(remaining_jobs) < 3

    def test_cleanup_completed_jobs_older_than_days(self, job_service):
        """Test cleaning up old completed jobs."""
        now = datetime.now(UTC)
        old_time = now - timedelta(days=5)

        # Create old completed job
        old_job = Job(
            job_id="old-completed",
            model_id=1,
            type="training",
            status=JobStatus.COMPLETED,
            message="Old completed job",
            sql_query=None,
            created_at=old_time,
            updated_at=old_time,
        )

        # Create recent completed job
        recent_job = Job(
            job_id="recent-completed",
            model_id=2,
            type="training",
            status=JobStatus.COMPLETED,
            message="Recent completed job",
            sql_query=None,
            created_at=now,
            updated_at=now,
        )

        # Create old running job (should not be deleted)
        old_running_job = Job(
            job_id="old-running",
            model_id=3,
            type="training",
            status=JobStatus.RUNNING,
            message="Old running job",
            sql_query=None,
            created_at=old_time,
            updated_at=old_time,
        )

        # Create all jobs
        job_service.create_job(old_job)
        job_service.create_job(recent_job)
        job_service.create_job(old_running_job)

        # Cleanup completed jobs older than 3 days
        job_service.cleanup_completed_jobs_older_than_days(3)

        # Check remaining jobs
        assert not job_service.job_exists("old-completed")  # Should be deleted
        assert job_service.job_exists("recent-completed")  # Should remain
        assert job_service.job_exists("old-running")  # Should remain (not completed)


class TestJobServiceHelpers:
    """Test helper methods in JobService."""

    def test_escape_string(self, job_service):
        """Test SQL string escaping."""
        # Normal string
        assert job_service._escape_string("test") == "test"

        # String with single quote
        assert job_service._escape_string("test's job") == "test''s job"

        # String with multiple quotes
        assert job_service._escape_string("'test' 'data'") == "''test'' ''data''"

        # Empty string
        assert job_service._escape_string("") == ""

        # None input
        assert job_service._escape_string(None) == ""

    def test_result_to_job_conversion(self, job_service, sample_job):
        """Test converting database row to Job object."""
        # Create and retrieve job to test conversion
        job_service.create_job(sample_job)
        retrieved = job_service.get_job_by_id(sample_job.job_id)

        assert retrieved is not None
        assert isinstance(retrieved, Job)
        assert retrieved.job_id == sample_job.job_id
        assert retrieved.model_id == sample_job.model_id
        assert retrieved.type == sample_job.type
        assert retrieved.status == sample_job.status
        assert retrieved.message == sample_job.message
        assert retrieved.sql_query == sample_job.sql_query
        assert isinstance(retrieved.created_at, datetime)
        assert isinstance(retrieved.updated_at, datetime)


class TestJobServiceEdgeCases:
    """Test edge cases and error conditions."""

    def test_job_with_special_characters(self, job_service):
        """Test job with special characters in fields."""
        now = datetime.now(UTC)
        special_job = Job(
            job_id="special-job",
            model_id=1,
            type="test'type",
            status=JobStatus.PENDING,
            message="Message with 'quotes' and \"double quotes\"",
            sql_query="SELECT * FROM table WHERE name = 'test''s data'",
            created_at=now,
            updated_at=now,
        )

        # Should handle special characters properly
        job_service.create_job(special_job)
        retrieved = job_service.get_job_by_id("special-job")

        assert retrieved is not None
        assert retrieved.type == "test'type"
        assert retrieved.message == "Message with 'quotes' and \"double quotes\""

    def test_job_with_none_optional_fields(self, job_service):
        """Test job with None optional fields."""
        now = datetime.now(UTC)
        job = Job(
            job_id="no-optionals",
            model_id=None,
            type="test",
            status=JobStatus.PENDING,
            message="Test",
            sql_query=None,
            created_at=now,
            updated_at=now,
        )

        job_service.create_job(job)
        retrieved = job_service.get_job_by_id("no-optionals")

        assert retrieved is not None
        assert retrieved.model_id is None
        assert retrieved.sql_query is None

    def test_empty_database_operations(self, job_service):
        """Test operations on empty database."""
        # Get operations on empty database
        assert job_service.get_job_by_id("nonexistent") is None
        assert job_service.get_recent_jobs(10) == []
        assert job_service.get_jobs_by_status(JobStatus.PENDING) == []

        # Utility methods on empty database
        assert not job_service.job_exists("nonexistent")
        assert not job_service.cancel_job("nonexistent")

        # Cleanup on empty database should not error
        job_service.cleanup_old_jobs(datetime.now(UTC))
        job_service.cleanup_completed_jobs_older_than_days(7)
