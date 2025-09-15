"""Job data models and types."""

import uuid
from dataclasses import asdict, dataclass
from datetime import datetime
from enum import Enum
from typing import Any


class JobStatus(Enum):
    """Job execution status."""

    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class JobType(Enum):
    """Types of jobs that can be executed."""

    TRAIN_MODEL = "train_model"
    PREDICT_MODEL = "predict_model"
    CREATE_MODEL = "create_model"
    VALIDATE_SCHEMA = "validate_schema"


@dataclass
class Job:
    """Represents a job in the Arc system (matches arc-cpp schema)."""

    job_id: str
    model_id: int | None
    type: JobType
    status: JobStatus
    message: str
    sql_query: str | None
    created_at: datetime
    updated_at: datetime

    @classmethod
    def create(
        cls,
        job_type: JobType,
        model_id: int | None = None,
        message: str = "",
        sql_query: str | None = None,
    ) -> "Job":
        """Create a new job with generated ID and current timestamp."""
        now = datetime.now()
        return cls(
            job_id=str(uuid.uuid4()),
            model_id=model_id,
            type=job_type,
            status=JobStatus.PENDING,
            message=message,
            sql_query=sql_query,
            created_at=now,
            updated_at=now,
        )

    def start(self, message: str = "Started") -> None:
        """Mark job as started."""
        if self.status != JobStatus.PENDING:
            raise ValueError(f"Cannot start job in status: {self.status}")

        self.status = JobStatus.RUNNING
        self.message = message
        self.updated_at = datetime.now()

    def complete(self, message: str = "Completed successfully") -> None:
        """Mark job as completed."""
        if self.status != JobStatus.RUNNING:
            raise ValueError(f"Cannot complete job in status: {self.status}")

        self.status = JobStatus.COMPLETED
        self.message = message
        self.updated_at = datetime.now()

    def fail(self, error_message: str) -> None:
        """Mark job as failed with error message."""
        if self.status not in (JobStatus.PENDING, JobStatus.RUNNING):
            raise ValueError(f"Cannot fail job in status: {self.status}")

        self.status = JobStatus.FAILED
        self.message = error_message
        self.updated_at = datetime.now()

    def cancel(self, message: str = "Cancelled by user") -> None:
        """Mark job as cancelled."""
        if self.status not in (JobStatus.PENDING, JobStatus.RUNNING):
            raise ValueError(f"Cannot cancel job in status: {self.status}")

        self.status = JobStatus.CANCELLED
        self.message = message
        self.updated_at = datetime.now()

    def update_message(self, message: str) -> None:
        """Update job message."""
        self.message = message
        self.updated_at = datetime.now()

    def to_dict(self) -> dict[str, Any]:
        """Convert job to dictionary for database storage."""
        data = asdict(self)

        # Convert enum values to strings
        data["type"] = self.type.value
        data["status"] = self.status.value

        # Convert datetime objects to ISO strings
        data["created_at"] = self.created_at.isoformat()
        data["updated_at"] = self.updated_at.isoformat()

        return data

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Job":
        """Create job from dictionary (from database)."""
        # Convert string values back to enums
        job_type = JobType(data["type"])
        status = JobStatus(data["status"])

        # Convert datetime fields - handle both ISO strings and datetime objects
        created_at = data["created_at"]
        if isinstance(created_at, str):
            created_at = datetime.fromisoformat(created_at)
        elif not isinstance(created_at, datetime):
            raise ValueError(f"Invalid created_at type: {type(created_at)}")

        updated_at = data["updated_at"]
        if isinstance(updated_at, str):
            updated_at = datetime.fromisoformat(updated_at)
        elif not isinstance(updated_at, datetime):
            raise ValueError(f"Invalid updated_at type: {type(updated_at)}")

        return cls(
            job_id=data["job_id"],
            model_id=data["model_id"],
            type=job_type,
            status=status,
            message=data["message"] or "",
            sql_query=data["sql_query"],
            created_at=created_at,
            updated_at=updated_at,
        )

    @property
    def is_active(self) -> bool:
        """Check if job is currently active (pending or running)."""
        return self.status in (JobStatus.PENDING, JobStatus.RUNNING)

    @property
    def is_finished(self) -> bool:
        """Check if job is finished (completed, failed, or cancelled)."""
        return self.status in (
            JobStatus.COMPLETED,
            JobStatus.FAILED,
            JobStatus.CANCELLED,
        )

    @property
    def duration_seconds(self) -> float | None:
        """Get job duration in seconds from creation to last update."""
        return (self.updated_at - self.created_at).total_seconds()
