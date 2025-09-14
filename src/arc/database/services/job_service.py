"""Job service for managing Arc job tracking."""

from datetime import UTC, datetime
from typing import Any

from ..base import DatabaseError
from ..models.job import Job, JobStatus
from .base import BaseService


class JobService(BaseService):
    """Service for managing job tracking in the system database.

    Handles operations on jobs and trained_models tables including:
    - Job lifecycle management
    - Training artifact tracking
    - Job status monitoring
    - CRUD operations with proper SQL escaping
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
            sql = self._build_job_insert_sql(job)
            self._system_execute(sql)
        except Exception as e:
            raise DatabaseError(f"Failed to create job {job.job_id}: {e}") from e

    def update_job_status(self, job_id: str, status: JobStatus, message: str) -> None:
        """Update job status and message.

        Args:
            job_id: Job ID to update
            status: New job status
            message: Status message

        Raises:
            DatabaseError: If job update fails
        """
        try:
            sql = self._build_job_update_sql(job_id, status, message)
            self._system_execute(sql)
        except Exception as e:
            raise DatabaseError(f"Failed to update job {job_id}: {e}") from e

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
            sql = f"SELECT * FROM jobs WHERE job_id = '{self._escape_string(job_id)}'"
            result = self._system_query(sql)
            if result.empty():
                return None
            return self._result_to_job(result.first())
        except Exception as e:
            raise DatabaseError(f"Failed to get job by id {job_id}: {e}") from e

    def cancel_job(self, job_id: str) -> bool:
        """Cancel a job if it's in a cancellable state.

        Args:
            job_id: Job ID to cancel

        Returns:
            True if job was cancelled, False if not found or not cancellable

        Raises:
            DatabaseError: If query execution fails
        """
        try:
            job = self.get_job_by_id(job_id)
            if not job:
                return False

            # Only cancel jobs that are running or pending
            if job.status in (JobStatus.RUNNING, JobStatus.PENDING):
                self.update_job_status(job_id, JobStatus.CANCELLED, "Cancelled by user")
                return True

            return False
        except Exception as e:
            raise DatabaseError(f"Failed to cancel job {job_id}: {e}") from e

    def get_recent_jobs(self, limit: int) -> list[Job]:
        """Get recent jobs ordered by creation date (newest first).

        Args:
            limit: Maximum number of jobs to return

        Returns:
            List of Job objects ordered by created_at DESC

        Raises:
            DatabaseError: If query execution fails
        """
        try:
            sql = f"SELECT * FROM jobs ORDER BY created_at DESC LIMIT {limit}"
            result = self._system_query(sql)
            return self._results_to_jobs(result)
        except Exception as e:
            raise DatabaseError(f"Failed to get recent jobs: {e}") from e

    def get_jobs_by_status(self, status: JobStatus) -> list[Job]:
        """Get all jobs with a specific status.

        Args:
            status: Job status to filter by

        Returns:
            List of Job objects with the specified status, ordered by created_at DESC

        Raises:
            DatabaseError: If query execution fails
        """
        try:
            status_str = JobStatus.job_status_to_string(status)
            escaped_status = self._escape_string(status_str)
            sql = f"""SELECT * FROM jobs WHERE status = '{escaped_status}'
                ORDER BY created_at DESC"""
            result = self._system_query(sql)
            return self._results_to_jobs(result)
        except Exception as e:
            raise DatabaseError(f"Failed to get jobs by status {status}: {e}") from e

    def get_job_status(self, job_id: str) -> JobStatus:
        """Get the status of a job.

        Args:
            job_id: Job ID to get status for

        Returns:
            JobStatus of the job

        Raises:
            DatabaseError: If job not found or query fails
        """
        try:
            job = self.get_job_by_id(job_id)
            if not job:
                raise DatabaseError(f"Job not found: {job_id}")
            return job.status
        except DatabaseError:
            raise
        except Exception as e:
            raise DatabaseError(f"Failed to get job status for {job_id}: {e}") from e

    def job_exists(self, job_id: str) -> bool:
        """Check if a job exists by ID.

        Args:
            job_id: Job ID to check

        Returns:
            True if job exists, False otherwise

        Raises:
            DatabaseError: If query execution fails
        """
        return self.get_job_by_id(job_id) is not None

    def cleanup_old_jobs(self, before: datetime) -> None:
        """Delete jobs created before the specified datetime.

        Args:
            before: Datetime cutoff - jobs created before this will be deleted

        Raises:
            DatabaseError: If cleanup operation fails
        """
        try:
            before_str = before.isoformat()
            sql = f"DELETE FROM jobs WHERE created_at < '{before_str}'"
            self._system_execute(sql)
        except Exception as e:
            raise DatabaseError(f"Failed to cleanup old jobs: {e}") from e

    def cleanup_completed_jobs_older_than_days(self, days: int) -> None:
        """Delete completed jobs older than specified number of days.

        Args:
            days: Number of days - completed jobs older than this will be deleted

        Raises:
            DatabaseError: If cleanup operation fails
        """
        try:
            from datetime import timedelta

            cutoff_time = datetime.now(UTC) - timedelta(days=days)

            # Only delete completed, failed, or cancelled jobs
            cutoff_str = cutoff_time.isoformat()
            completed = JobStatus.COMPLETED.value
            failed = JobStatus.FAILED.value
            cancelled = JobStatus.CANCELLED.value
            sql = (
                f"DELETE FROM jobs WHERE created_at < '{cutoff_str}' "
                f"AND status IN ('{completed}', '{failed}', '{cancelled}')"
            )
            self._system_execute(sql)
        except Exception as e:
            msg = f"Failed to cleanup completed jobs older than {days} days: {e}"
            raise DatabaseError(msg) from e

    def _result_to_job(self, row: dict[str, Any]) -> Job:
        """Convert a database row to a Job object.

        Args:
            row: Database row as dictionary

        Returns:
            Job object created from row data

        Raises:
            DatabaseError: If conversion fails
        """
        try:
            # Handle timestamp conversion
            created_at = row.get("created_at")
            updated_at = row.get("updated_at")

            if isinstance(created_at, str):
                created_at = datetime.fromisoformat(created_at)
            elif isinstance(created_at, datetime):
                # Database returns naive datetime, assume UTC
                if created_at.tzinfo is None:
                    created_at = created_at.replace(tzinfo=UTC)
            elif created_at is None:
                created_at = datetime.now(UTC)

            if isinstance(updated_at, str):
                updated_at = datetime.fromisoformat(updated_at)
            elif isinstance(updated_at, datetime):
                # Database returns naive datetime, assume UTC
                if updated_at.tzinfo is None:
                    updated_at = updated_at.replace(tzinfo=UTC)
            elif updated_at is None:
                updated_at = datetime.now(UTC)

            return Job(
                job_id=str(row["job_id"]),
                model_id=row.get("model_id"),  # Can be None
                type=str(row["type"]),
                status=JobStatus.string_to_job_status(str(row["status"])),
                message=str(row["message"]),
                sql_query=row.get("sql_query"),  # Can be None
                created_at=created_at,
                updated_at=updated_at,
            )
        except (KeyError, ValueError, TypeError) as e:
            raise DatabaseError(f"Failed to convert row to Job: {e}") from e

    def _results_to_jobs(self, result) -> list[Job]:
        """Convert query results to list of Job objects.

        Args:
            result: QueryResult object from database query

        Returns:
            List of Job objects

        Raises:
            DatabaseError: If conversion fails
        """
        try:
            return [self._result_to_job(row) for row in result.rows]
        except Exception as e:
            raise DatabaseError(f"Failed to convert results to jobs: {e}") from e

    def _build_job_insert_sql(self, job: Job) -> str:
        """Build INSERT SQL statement for a job.

        Args:
            job: Job object to insert

        Returns:
            SQL INSERT statement string

        Raises:
            DatabaseError: If SQL building fails
        """
        try:
            # Format timestamps for SQL
            created_at_str = job.created_at.isoformat()
            updated_at_str = job.updated_at.isoformat()

            # Handle optional model_id
            model_id_sql = str(job.model_id) if job.model_id is not None else "NULL"

            # Handle optional sql_query
            sql_query_sql = (
                f"'{self._escape_string(job.sql_query)}'"
                if job.sql_query is not None
                else "NULL"
            )

            sql = f"""INSERT INTO jobs (
                job_id, model_id, type, status, message, sql_query,
                created_at, updated_at
            ) VALUES (
                '{self._escape_string(job.job_id)}',
                {model_id_sql},
                '{self._escape_string(job.type)}',
                '{self._escape_string(JobStatus.job_status_to_string(job.status))}',
                '{self._escape_string(job.message)}',
                {sql_query_sql},
                '{created_at_str}',
                '{updated_at_str}'
            )"""

            return sql
        except Exception as e:
            raise DatabaseError(f"Failed to build insert SQL: {e}") from e

    def _build_job_update_sql(
        self, job_id: str, status: JobStatus, message: str
    ) -> str:
        """Build UPDATE SQL statement for job status.

        Args:
            job_id: Job ID to update
            status: New status
            message: Status message

        Returns:
            SQL UPDATE statement string

        Raises:
            DatabaseError: If SQL building fails
        """
        try:
            # Current timestamp for updated_at
            updated_at_str = datetime.now(UTC).isoformat()

            status_str = JobStatus.job_status_to_string(status)
            sql = f"""UPDATE jobs SET
                status = '{self._escape_string(status_str)}',
                message = '{self._escape_string(message)}',
                updated_at = '{updated_at_str}'
            WHERE job_id = '{self._escape_string(job_id)}'"""

            return sql
        except Exception as e:
            raise DatabaseError(f"Failed to build update SQL: {e}") from e

    def _escape_string(self, value: str) -> str:
        """Escape string values for SQL to prevent injection.

        Args:
            value: String value to escape

        Returns:
            Escaped string safe for SQL

        Raises:
            DatabaseError: If escaping fails
        """
        if value is None:
            return ""

        try:
            # Basic SQL string escaping - replace single quotes with double quotes
            return str(value).replace("'", "''")
        except Exception as e:
            raise DatabaseError(f"Failed to escape string '{value}': {e}") from e
