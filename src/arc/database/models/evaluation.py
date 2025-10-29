"""Models for evaluation tracking."""

from dataclasses import dataclass
from datetime import datetime
from enum import Enum


class EvaluationStatus(Enum):
    """Enumeration of possible evaluation run statuses."""

    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"

    @staticmethod
    def from_string(status_str: str) -> "EvaluationStatus":
        """Convert string to EvaluationStatus enum.

        Args:
            status_str: String representation of status

        Returns:
            EvaluationStatus enum value

        Raises:
            ValueError: If status string is not valid
        """
        try:
            return EvaluationStatus(status_str.lower())
        except ValueError as e:
            raise ValueError(f"Invalid evaluation status: {status_str}") from e


@dataclass
class EvaluationRun:
    """Data class representing an evaluation run."""

    run_id: str
    evaluator_id: str
    job_id: str | None
    model_id: str
    dataset: str
    target_column: str
    status: EvaluationStatus
    started_at: datetime | None
    completed_at: datetime | None
    metrics_result: str | None  # JSON string
    prediction_table: str | None
    error_message: str | None
    created_at: datetime
    updated_at: datetime
