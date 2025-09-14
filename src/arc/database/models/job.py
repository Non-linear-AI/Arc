from dataclasses import dataclass
from datetime import datetime
from enum import Enum


class JobStatus(Enum):
    """Enumeration of possible job statuses."""

    UNKNOWN = "UNKNOWN"
    PENDING = "PENDING"
    RUNNING = "RUNNING"
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"
    CANCELLED = "CANCELLED"

    @staticmethod
    def job_status_to_string(status: "JobStatus") -> str:
        """Convert JobStatus to string representation.

        Args:
            status: JobStatus enum value

        Returns:
            String representation of the status
        """
        return status.value

    @staticmethod
    def string_to_job_status(status_str: str) -> "JobStatus":
        """Convert string to JobStatus enum.

        Args:
            status_str: String representation of status

        Returns:
            JobStatus enum value

        Raises:
            ValueError: If status string is not valid
        """
        try:
            return JobStatus(status_str)
        except ValueError as e:
            raise ValueError(f"Invalid job status: {status_str}") from e


@dataclass
class Job:
    """Data class representing a job in the Arc system.

    Mirrors the C++ JobRecord struct with exact field mapping.
    """

    job_id: str
    model_id: int | None
    type: str
    status: JobStatus
    message: str
    sql_query: str | None
    created_at: datetime
    updated_at: datetime
