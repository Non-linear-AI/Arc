"""Data models for training tracking system."""

from dataclasses import dataclass
from datetime import datetime
from enum import Enum


class TrainingStatus(Enum):
    """Enumeration of possible training run statuses."""

    PENDING = "pending"
    RUNNING = "running"
    PAUSED = "paused"
    STOPPED = "stopped"
    COMPLETED = "completed"
    FAILED = "failed"

    @staticmethod
    def to_string(status: "TrainingStatus") -> str:
        """Convert TrainingStatus to string representation.

        Args:
            status: TrainingStatus enum value

        Returns:
            String representation of the status
        """
        return status.value

    @staticmethod
    def from_string(status_str: str) -> "TrainingStatus":
        """Convert string to TrainingStatus enum.

        Args:
            status_str: String representation of status

        Returns:
            TrainingStatus enum value

        Raises:
            ValueError: If status string is not valid
        """
        try:
            return TrainingStatus(status_str)
        except ValueError as e:
            raise ValueError(f"Invalid training status: {status_str}") from e


class CheckpointStatus(Enum):
    """Enumeration of possible checkpoint statuses."""

    SAVED = "saved"
    DELETED = "deleted"
    CORRUPTED = "corrupted"

    @staticmethod
    def to_string(status: "CheckpointStatus") -> str:
        """Convert CheckpointStatus to string representation.

        Args:
            status: CheckpointStatus enum value

        Returns:
            String representation of the status
        """
        return status.value

    @staticmethod
    def from_string(status_str: str) -> "CheckpointStatus":
        """Convert string to CheckpointStatus enum.

        Args:
            status_str: String representation of status

        Returns:
            CheckpointStatus enum value

        Raises:
            ValueError: If status string is not valid
        """
        try:
            return CheckpointStatus(status_str)
        except ValueError as e:
            raise ValueError(f"Invalid checkpoint status: {status_str}") from e


class MetricType(Enum):
    """Enumeration of metric types."""

    TRAIN = "train"
    VALIDATION = "validation"
    TEST = "test"

    @staticmethod
    def to_string(metric_type: "MetricType") -> str:
        """Convert MetricType to string representation.

        Args:
            metric_type: MetricType enum value

        Returns:
            String representation of the metric type
        """
        return metric_type.value

    @staticmethod
    def from_string(type_str: str) -> "MetricType":
        """Convert string to MetricType enum.

        Args:
            type_str: String representation of metric type

        Returns:
            MetricType enum value

        Raises:
            ValueError: If type string is not valid
        """
        try:
            return MetricType(type_str)
        except ValueError as e:
            raise ValueError(f"Invalid metric type: {type_str}") from e


@dataclass
class TrainingRun:
    """Data class representing a training run in the Arc system."""

    run_id: str
    job_id: str | None
    model_id: str | None
    run_name: str | None
    description: str | None
    tensorboard_enabled: bool
    tensorboard_log_dir: str | None
    metric_log_frequency: int
    checkpoint_frequency: int
    status: TrainingStatus
    started_at: datetime | None
    paused_at: datetime | None
    resumed_at: datetime | None
    completed_at: datetime | None
    artifact_path: str | None
    final_metrics: str | None  # JSON string
    training_config: str | None  # JSON string - snapshot of training config used
    original_config: str | None  # JSON string
    current_config: str | None  # JSON string
    config_history: str | None  # JSON string
    created_at: datetime
    updated_at: datetime


@dataclass
class TrainingMetric:
    """Data class representing a training metric."""

    metric_id: str
    run_id: str
    metric_name: str
    metric_type: MetricType
    step: int
    epoch: int
    value: float
    timestamp: datetime


@dataclass
class TrainingCheckpoint:
    """Data class representing a training checkpoint."""

    checkpoint_id: str
    run_id: str
    epoch: int
    step: int
    checkpoint_path: str
    metrics: str  # JSON string
    is_best: bool
    file_size_bytes: int | None
    status: CheckpointStatus
    created_at: datetime
