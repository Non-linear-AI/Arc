"""Enhanced job service for managing Arc job system."""

from dataclasses import asdict
from datetime import UTC
from typing import Any

from ...jobs.models import Job, JobStatus, JobType
from ..base import DatabaseError
from .base import BaseService


class JobService(BaseService):
    """Enhanced service for managing jobs in the system database.

    Handles operations on the jobs table including:
    - Job lifecycle management (create, read, update, delete)
    - Job status tracking and progress monitoring
    - Job querying and filtering
    - CRUD operations with proper parameterized queries
    """

    def __init__(self, db_manager):
        """Initialize JobService.

        Args:
            db_manager: DatabaseManager instance
        """
        super().__init__(db_manager)

    def create_job(self, job: Job) -> None:
        """Create a new job in the database.

        Args:
            job: Job object to create

        Raises:
            DatabaseError: If job creation fails
        """
        try:
            sql = """
            INSERT INTO jobs (
                job_id, model_id, type, status, message, sql_query,
                created_at, updated_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """

            job_data = asdict(job)
            params = [
                job_data["job_id"],
                job_data["model_id"],
                job_data["type"].value
                if hasattr(job_data["type"], "value")
                else job_data["type"],
                job_data["status"].value
                if hasattr(job_data["status"], "value")
                else job_data["status"],
                job_data["message"],
                job_data["sql_query"],
                job_data["created_at"],
                job_data["updated_at"],
            ]

            self.db_manager.system_execute(sql, params)
        except Exception as e:
            raise DatabaseError(f"Failed to create job {job.job_id}: {e}") from e

    def get_job_by_id(self, job_id: str) -> Job | None:
        """Get a job by its ID.

        Args:
            job_id: Job ID to search for

        Returns:
            Job object if found, None otherwise

        Raises:
            DatabaseError: If query execution fails
        """
        try:
            sql = "SELECT * FROM jobs WHERE job_id = ?"
            result = self._system_query(sql, [job_id])

            if result.empty():
                return None

            return Job.from_dict(result.first())
        except Exception as e:
            raise DatabaseError(f"Failed to get job by id {job_id}: {e}") from e

    def update_job(self, job: Job) -> None:
        """Update an existing job in the database.

        Args:
            job: Job object with updated data

        Raises:
            DatabaseError: If job update fails
        """
        try:
            sql = """
            UPDATE jobs SET
                model_id = ?, type = ?, status = ?, message = ?,
                sql_query = ?, created_at = ?, updated_at = ?
            WHERE job_id = ?
            """

            job_data = asdict(job)
            params = [
                job_data["model_id"],
                job_data["type"].value
                if hasattr(job_data["type"], "value")
                else job_data["type"],
                job_data["status"].value
                if hasattr(job_data["status"], "value")
                else job_data["status"],
                job_data["message"],
                job_data["sql_query"],
                job_data["created_at"],
                job_data["updated_at"],
                job_data["job_id"],
            ]

            self.db_manager.system_execute(sql, params)
        except Exception as e:
            raise DatabaseError(f"Failed to update job {job.job_id}: {e}") from e

    def update_job_status(
        self, job_id: str, status: JobStatus, message: str = ""
    ) -> None:
        """Update job status and message.

        Args:
            job_id: Job identifier
            status: New job status
            message: Optional status message

        Raises:
            DatabaseError: If update fails
        """
        try:
            from datetime import datetime

            sql = """
            UPDATE jobs SET status = ?, message = ?, updated_at = ?
            WHERE job_id = ?
            """

            params = [
                status.value,
                message,
                datetime.now(UTC),
                job_id,
            ]

            self.db_manager.system_execute(sql, params)
        except Exception as e:
            raise DatabaseError(f"Failed to update job status {job_id}: {e}") from e

    def delete_job(self, job_id: str) -> bool:
        """Delete a job by ID.

        Args:
            job_id: Job ID to delete

        Returns:
            True if job was deleted, False if not found

        Raises:
            DatabaseError: If deletion fails
        """
        try:
            # Check if job exists first
            if self.get_job_by_id(job_id) is None:
                return False

            sql = "DELETE FROM jobs WHERE job_id = ?"
            self.db_manager.system_execute(sql, [job_id])
            return True
        except Exception as e:
            raise DatabaseError(f"Failed to delete job {job_id}: {e}") from e

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
        try:
            if active_only:
                sql = """
                SELECT * FROM jobs
                WHERE status IN (?, ?)
                ORDER BY created_at DESC
                LIMIT ?
                """
                params = [JobStatus.PENDING.value, JobStatus.RUNNING.value, limit]
            else:
                sql = "SELECT * FROM jobs ORDER BY created_at DESC LIMIT ?"
                params = [limit]

            result = self._system_query(sql, params)
            return [Job.from_dict(row) for row in result.rows]
        except Exception as e:
            raise DatabaseError(f"Failed to list jobs: {e}") from e

    def get_jobs_by_status(self, status: JobStatus) -> list[Job]:
        """Get all jobs with a specific status.

        Args:
            status: Job status to filter by

        Returns:
            List of Job objects with the specified status

        Raises:
            DatabaseError: If query execution fails
        """
        try:
            sql = "SELECT * FROM jobs WHERE status = ? ORDER BY created_at DESC"
            result = self._system_query(sql, [status.value])
            return [Job.from_dict(row) for row in result.rows]
        except Exception as e:
            raise DatabaseError(f"Failed to get jobs by status {status}: {e}") from e

    def get_jobs_by_type(self, job_type: JobType) -> list[Job]:
        """Get all jobs with a specific type.

        Args:
            job_type: Job type to filter by

        Returns:
            List of Job objects with the specified type

        Raises:
            DatabaseError: If query execution fails
        """
        try:
            sql = "SELECT * FROM jobs WHERE type = ? ORDER BY created_at DESC"
            result = self._system_query(sql, [job_type.value])
            return [Job.from_dict(row) for row in result.rows]
        except Exception as e:
            raise DatabaseError(f"Failed to get jobs by type {job_type}: {e}") from e

    def get_job_counts_by_status(self) -> dict[str, int]:
        """Get count of jobs by status.

        Returns:
            Dictionary mapping status strings to counts

        Raises:
            DatabaseError: If query execution fails
        """
        try:
            sql = "SELECT status, COUNT(*) as count FROM jobs GROUP BY status"
            result = self._system_query(sql)

            counts = {}
            for row in result.rows:
                counts[row["status"]] = row["count"]

            # Ensure all statuses are represented
            for status in JobStatus:
                if status.value not in counts:
                    counts[status.value] = 0

            return counts
        except Exception as e:
            raise DatabaseError(f"Failed to get job counts by status: {e}") from e

    def get_jobs_by_model_id(self, model_id: int) -> list[Job]:
        """Get all jobs for a specific model.

        Args:
            model_id: Model ID to filter by

        Returns:
            List of Job objects for the specified model

        Raises:
            DatabaseError: If query execution fails
        """
        try:
            sql = "SELECT * FROM jobs WHERE model_id = ? ORDER BY created_at DESC"
            result = self._system_query(sql, [model_id])
            return [Job.from_dict(row) for row in result.rows]
        except Exception as e:
            raise DatabaseError(
                f"Failed to get jobs by model_id {model_id}: {e}"
            ) from e

    def cleanup_old_jobs(self, days_old: int = 30) -> int:
        """Clean up jobs older than specified days.

        Args:
            days_old: Remove jobs older than this many days

        Returns:
            Number of jobs deleted

        Raises:
            DatabaseError: If cleanup fails
        """
        try:
            # Get count before deletion
            count_sql = f"""
            SELECT COUNT(*) as count FROM jobs
            WHERE status IN (?, ?, ?)
            AND created_at < (CURRENT_TIMESTAMP - INTERVAL '{days_old} days')
            """

            params = [
                JobStatus.COMPLETED.value,
                JobStatus.FAILED.value,
                JobStatus.CANCELLED.value,
            ]

            result = self._system_query(count_sql, params)
            count = result.first()["count"] if not result.empty() else 0

            # Perform deletion
            delete_sql = f"""
            DELETE FROM jobs
            WHERE status IN (?, ?, ?)
            AND created_at < (CURRENT_TIMESTAMP - INTERVAL '{days_old} days')
            """

            self.db_manager.system_execute(delete_sql, params)
            return count
        except Exception as e:
            raise DatabaseError(f"Failed to cleanup old jobs: {e}") from e

    def get_active_jobs(self) -> list[Job]:
        """Get all active (pending or running) jobs.

        Returns:
            List of active Job objects

        Raises:
            DatabaseError: If query fails
        """
        return self.list_jobs(active_only=True)

    def _system_query(self, sql: str, params: list | None = None) -> Any:
        """Execute a parameterized query against the system database.

        Args:
            sql: SQL query with parameter placeholders
            params: List of parameters for the query

        Returns:
            QueryResult from the database

        Raises:
            DatabaseError: If query execution fails
        """
        return self.db_manager.system_query(sql, params or [])
