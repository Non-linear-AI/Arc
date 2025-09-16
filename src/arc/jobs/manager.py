"""Job management for Arc system using service layer."""

import logging
from typing import Any

from ..database.services.job_service import JobService
from .models import Job, JobStatus, JobType

logger = logging.getLogger(__name__)


class JobManager:
    """Manages job lifecycle using the service layer."""

    def __init__(self, db_manager):
        """Initialize JobManager.

        Args:
            db_manager: DatabaseManager instance for persistence
        """
        self.db_manager = db_manager
        self.job_service = JobService(db_manager)

    def create_job(
        self,
        job_type: JobType,
        model_id: int | None = None,
        message: str = "",
        sql_query: str | None = None,
    ) -> Job:
        """Create a new job and persist it.

        Args:
            job_type: Type of job to create
            model_id: ID of the model being processed (for train/predict jobs)
            message: Initial job message
            sql_query: SQL query for data (for train jobs: "on <table>")

        Returns:
            Created Job object

        Raises:
            DatabaseError: If job creation fails
        """
        job = Job.create(job_type, model_id, message, sql_query)
        self.job_service.create_job(job)
        logger.info(f"Created job {job.job_id}: {message or job_type.value}")
        return job

    def get_job(self, job_id: str) -> Job | None:
        """Get a job by ID.

        Args:
            job_id: Job ID to retrieve

        Returns:
            Job object if found, None otherwise

        Raises:
            DatabaseError: If query fails
        """
        return self.job_service.get_job_by_id(job_id)

    def list_jobs(self, limit: int = 100, active_only: bool = False) -> list[Job]:
        """List jobs ordered by creation date (newest first).

        Args:
            limit: Maximum number of jobs to return
            active_only: If True, only return active jobs

        Returns:
            List of Job objects

        Raises:
            DatabaseError: If query fails
        """
        return self.job_service.list_jobs(limit, active_only)

    def update_job(self, job: Job) -> None:
        """Update an existing job in the database.

        Args:
            job: Job object with updated data

        Raises:
            DatabaseError: If update fails
        """
        self.job_service.update_job(job)

    def delete_job(self, job_id: str) -> bool:
        """Delete a job by ID.

        Args:
            job_id: Job ID to delete

        Returns:
            True if job was deleted, False if not found

        Raises:
            DatabaseError: If deletion fails
        """
        return self.job_service.delete_job(job_id)

    def cancel_job(self, job_id: str) -> bool:
        """Cancel a job by ID.

        Args:
            job_id: Job ID to cancel

        Returns:
            True if job was cancelled, False if not found or not cancellable

        Raises:
            DatabaseError: If operation fails
        """
        job = self.job_service.get_job_by_id(job_id)
        if not job:
            return False

        if not job.is_active:
            return False

        job.cancel()
        self.job_service.update_job(job)
        return True

    def get_active_jobs(self) -> list[Job]:
        """Get all active (pending or running) jobs.

        Returns:
            List of active Job objects

        Raises:
            DatabaseError: If query fails
        """
        return self.job_service.get_active_jobs()

    def get_jobs_by_status(self, status: JobStatus) -> list[Job]:
        """Get all jobs with a specific status.

        Args:
            status: Job status to filter by

        Returns:
            List of Job objects with the specified status

        Raises:
            DatabaseError: If query execution fails
        """
        return self.job_service.get_jobs_by_status(status)

    def get_jobs_by_type(self, job_type: JobType) -> list[Job]:
        """Get all jobs with a specific type.

        Args:
            job_type: Job type to filter by

        Returns:
            List of Job objects with the specified type

        Raises:
            DatabaseError: If query execution fails
        """
        return self.job_service.get_jobs_by_type(job_type)

    def get_jobs_by_model_id(self, model_id: int) -> list[Job]:
        """Get all jobs for a specific model.

        Args:
            model_id: Model ID to filter by

        Returns:
            List of Job objects for the specified model

        Raises:
            DatabaseError: If query execution fails
        """
        return self.job_service.get_jobs_by_model_id(model_id)

    def cleanup_old_jobs(self, days_old: int = 30) -> int:
        """Clean up jobs older than specified days.

        Args:
            days_old: Remove jobs older than this many days

        Returns:
            Number of jobs deleted

        Raises:
            DatabaseError: If cleanup fails
        """
        count = self.job_service.cleanup_old_jobs(days_old)
        logger.info(f"Cleaned up {count} old jobs")
        return count

    def get_job_statistics(self) -> dict[str, Any]:
        """Get job statistics and counts.

        Returns:
            Dictionary with job statistics

        Raises:
            DatabaseError: If query fails
        """
        try:
            status_counts = self.job_service.get_job_counts_by_status()
            total_jobs = sum(status_counts.values())

            return {
                "total_jobs": total_jobs,
                "pending_jobs": status_counts.get(JobStatus.PENDING.value, 0),
                "running_jobs": status_counts.get(JobStatus.RUNNING.value, 0),
                "completed_jobs": status_counts.get(JobStatus.COMPLETED.value, 0),
                "failed_jobs": status_counts.get(JobStatus.FAILED.value, 0),
                "cancelled_jobs": status_counts.get(JobStatus.CANCELLED.value, 0),
            }
        except Exception as e:
            logger.error(f"Failed to get job statistics: {e}")
            raise
