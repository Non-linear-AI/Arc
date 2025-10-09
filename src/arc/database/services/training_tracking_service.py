"""Training tracking service for managing training runs, metrics, and checkpoints."""

import json
from datetime import UTC, datetime
from typing import Any
from uuid import uuid4

from arc.database.base import DatabaseError
from arc.database.models.training import (
    CheckpointStatus,
    MetricType,
    TrainingCheckpoint,
    TrainingMetric,
    TrainingRun,
    TrainingStatus,
)
from arc.database.services.base import BaseService


class TrainingTrackingService(BaseService):
    """Service for managing training tracking data in system database.

    Handles operations on training_runs, training_metrics, and
    training_checkpoints tables including:
    - Training run lifecycle management
    - Metric logging and retrieval
    - Checkpoint management
    - Status tracking and updates
    """

    def __init__(self, db_manager):
        """Initialize TrainingTrackingService.

        Args:
            db_manager: DatabaseManager instance
        """
        super().__init__(db_manager)

    # ========== TrainingRun Operations ==========

    def create_run(
        self,
        job_id: str | None = None,
        model_id: str | None = None,
        trainer_id: str | None = None,
        run_name: str | None = None,
        description: str | None = None,
        tensorboard_enabled: bool = True,
        tensorboard_log_dir: str | None = None,
        metric_log_frequency: int = 100,
        checkpoint_frequency: int = 5,
        config: dict[str, Any] | None = None,
    ) -> TrainingRun:
        """Create a new training run.

        Args:
            job_id: Associated job ID
            model_id: Model being trained
            trainer_id: Trainer being used
            run_name: Optional run name
            description: Optional run description
            tensorboard_enabled: Enable TensorBoard logging
            tensorboard_log_dir: TensorBoard log directory
            metric_log_frequency: Steps between metric logs
            checkpoint_frequency: Epochs between checkpoints
            config: Training configuration dictionary

        Returns:
            Created TrainingRun object

        Raises:
            DatabaseError: If run creation fails
        """
        try:
            run_id = str(uuid4())
            now = datetime.now(UTC)
            config_json = json.dumps(config) if config else None

            sql = """
            INSERT INTO training_runs (
                run_id, job_id, model_id, trainer_id,
                run_name, description,
                tensorboard_enabled, tensorboard_log_dir,
                metric_log_frequency, checkpoint_frequency,
                status, original_config, current_config,
                created_at, updated_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """

            params = [
                run_id,
                job_id,
                model_id,
                trainer_id,
                run_name,
                description,
                tensorboard_enabled,
                tensorboard_log_dir,
                metric_log_frequency,
                checkpoint_frequency,
                TrainingStatus.PENDING.value,
                config_json,
                config_json,
                now,
                now,
            ]

            self.db_manager.system_execute(sql, params)

            return TrainingRun(
                run_id=run_id,
                job_id=job_id,
                model_id=model_id,
                trainer_id=trainer_id,
                run_name=run_name,
                description=description,
                tensorboard_enabled=tensorboard_enabled,
                tensorboard_log_dir=tensorboard_log_dir,
                metric_log_frequency=metric_log_frequency,
                checkpoint_frequency=checkpoint_frequency,
                status=TrainingStatus.PENDING,
                started_at=None,
                paused_at=None,
                resumed_at=None,
                completed_at=None,
                artifact_path=None,
                final_metrics=None,
                original_config=config_json,
                current_config=config_json,
                config_history=None,
                created_at=now,
                updated_at=now,
            )
        except Exception as e:
            raise DatabaseError(f"Failed to create training run: {e}") from e

    def get_run_by_id(self, run_id: str) -> TrainingRun | None:
        """Get a training run by ID.

        Args:
            run_id: Training run ID

        Returns:
            TrainingRun object if found, None otherwise

        Raises:
            DatabaseError: If query fails
        """
        try:
            sql = "SELECT * FROM training_runs WHERE run_id = ?"
            result = self.db_manager.system_query(sql, [run_id])

            if result.empty():
                return None

            return self._row_to_training_run(result.first())
        except Exception as e:
            raise DatabaseError(f"Failed to get run {run_id}: {e}") from e

    def get_run_by_job_id(self, job_id: str) -> TrainingRun | None:
        """Get a training run by associated job ID.

        Args:
            job_id: Job ID

        Returns:
            TrainingRun object if found, None otherwise

        Raises:
            DatabaseError: If query fails
        """
        try:
            sql = "SELECT * FROM training_runs WHERE job_id = ?"
            result = self.db_manager.system_query(sql, [job_id])

            if result.empty():
                return None

            return self._row_to_training_run(result.first())
        except Exception as e:
            raise DatabaseError(f"Failed to get run by job {job_id}: {e}") from e

    def list_runs(
        self,
        limit: int = 100,
        status: TrainingStatus | None = None,
        model_id: str | None = None,
        trainer_id: str | None = None,
    ) -> list[TrainingRun]:
        """List training runs with optional filters.

        Args:
            limit: Maximum number of runs to return
            status: Filter by status
            model_id: Filter by model ID
            trainer_id: Filter by trainer ID

        Returns:
            List of TrainingRun objects

        Raises:
            DatabaseError: If query fails
        """
        try:
            conditions = []
            params = []

            if status:
                conditions.append("status = ?")
                params.append(status.value)

            if model_id:
                conditions.append("model_id = ?")
                params.append(model_id)

            if trainer_id:
                conditions.append("trainer_id = ?")
                params.append(trainer_id)

            where_clause = f"WHERE {' AND '.join(conditions)}" if conditions else ""
            sql = f"""
            SELECT * FROM training_runs
            {where_clause}
            ORDER BY created_at DESC
            LIMIT ?
            """
            params.append(limit)

            result = self.db_manager.system_query(sql, params)
            return [self._row_to_training_run(row) for row in result.rows]
        except Exception as e:
            raise DatabaseError(f"Failed to list runs: {e}") from e

    def update_run_status(
        self,
        run_id: str,
        status: TrainingStatus,
        timestamp_field: str | None = None,
    ) -> None:
        """Update training run status and optional timestamp.

        Args:
            run_id: Training run ID
            status: New status
            timestamp_field: Optional timestamp field to update
                           (started_at, paused_at, resumed_at, completed_at)

        Raises:
            DatabaseError: If update fails
        """
        try:
            now = datetime.now(UTC)
            timestamp_updates = []

            if timestamp_field:
                timestamp_updates.append(f"{timestamp_field} = ?")

            timestamp_clause = ", ".join(timestamp_updates) if timestamp_updates else ""
            if timestamp_clause:
                timestamp_clause = ", " + timestamp_clause

            sql = f"""
            UPDATE training_runs
            SET status = ?, updated_at = ?{timestamp_clause}
            WHERE run_id = ?
            """

            params = [status.value, now]
            if timestamp_field:
                params.append(now)
            params.append(run_id)

            self.db_manager.system_execute(sql, params)
        except Exception as e:
            raise DatabaseError(f"Failed to update run status {run_id}: {e}") from e

    def update_run_artifact(
        self,
        run_id: str,
        artifact_path: str,
        final_metrics: dict[str, Any] | None = None,
    ) -> None:
        """Update training run with artifact information after successful completion.

        Args:
            run_id: Training run ID
            artifact_path: Path to the saved model artifact
            final_metrics: Final metrics from training

        Raises:
            DatabaseError: If update fails
        """
        try:
            now = datetime.now(UTC)
            metrics_json = json.dumps(final_metrics) if final_metrics else None

            sql = """
            UPDATE training_runs
            SET artifact_path = ?, final_metrics = ?, updated_at = ?
            WHERE run_id = ?
            """

            params = [artifact_path, metrics_json, now, run_id]
            self.db_manager.system_execute(sql, params)
        except Exception as e:
            raise DatabaseError(f"Failed to update run artifact {run_id}: {e}") from e

    def update_run_config(self, run_id: str, new_config: dict[str, Any]) -> None:
        """Update training run configuration and add to history.

        Args:
            run_id: Training run ID
            new_config: New configuration dictionary

        Raises:
            DatabaseError: If update fails
        """
        try:
            # Get current run to access config history
            run = self.get_run_by_id(run_id)
            if not run:
                raise DatabaseError(f"Run {run_id} not found")

            # Build config history
            history = []
            if run.config_history:
                history = json.loads(run.config_history)

            if run.current_config:
                history.append(
                    {
                        "timestamp": datetime.now(UTC).isoformat(),
                        "config": run.current_config,
                    }
                )

            new_config_json = json.dumps(new_config)
            history_json = json.dumps(history)
            now = datetime.now(UTC)

            sql = """
            UPDATE training_runs
            SET current_config = ?, config_history = ?, updated_at = ?
            WHERE run_id = ?
            """

            params = [new_config_json, history_json, now, run_id]
            self.db_manager.system_execute(sql, params)
        except Exception as e:
            raise DatabaseError(f"Failed to update run config {run_id}: {e}") from e

    def delete_run(self, run_id: str) -> bool:
        """Delete a training run (manually deletes metrics and checkpoints).

        Args:
            run_id: Training run ID

        Returns:
            True if deleted, False if not found

        Raises:
            DatabaseError: If deletion fails
        """
        try:
            if self.get_run_by_id(run_id) is None:
                return False

            # Manually delete related metrics and checkpoints
            # (DuckDB doesn't support CASCADE)
            self.db_manager.system_execute(
                "DELETE FROM training_metrics WHERE run_id = ?", [run_id]
            )
            self.db_manager.system_execute(
                "DELETE FROM training_checkpoints WHERE run_id = ?", [run_id]
            )

            # Delete the run itself
            sql = "DELETE FROM training_runs WHERE run_id = ?"
            self.db_manager.system_execute(sql, [run_id])
            return True
        except Exception as e:
            raise DatabaseError(f"Failed to delete run {run_id}: {e}") from e

    # ========== TrainingMetric Operations ==========

    def log_metric(
        self,
        run_id: str,
        metric_name: str,
        metric_type: MetricType,
        step: int,
        epoch: int,
        value: float,
    ) -> TrainingMetric:
        """Log a training metric.

        Args:
            run_id: Training run ID
            metric_name: Name of metric (e.g., 'loss', 'accuracy')
            metric_type: Type of metric (train, validation, test)
            step: Training step number
            epoch: Training epoch number
            value: Metric value

        Returns:
            Created TrainingMetric object

        Raises:
            DatabaseError: If metric logging fails
        """
        try:
            metric_id = str(uuid4())
            now = datetime.now(UTC)

            sql = """
            INSERT INTO training_metrics (
                metric_id, run_id, metric_name, metric_type,
                step, epoch, value, timestamp
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """

            params = [
                metric_id,
                run_id,
                metric_name,
                metric_type.value,
                step,
                epoch,
                value,
                now,
            ]

            self.db_manager.system_execute(sql, params)

            return TrainingMetric(
                metric_id=metric_id,
                run_id=run_id,
                metric_name=metric_name,
                metric_type=metric_type,
                step=step,
                epoch=epoch,
                value=value,
                timestamp=now,
            )
        except Exception as e:
            raise DatabaseError(f"Failed to log metric: {e}") from e

    def get_metrics(
        self,
        run_id: str,
        metric_name: str | None = None,
        metric_type: MetricType | None = None,
        limit: int | None = None,
    ) -> list[TrainingMetric]:
        """Get metrics for a training run with optional filters.

        Args:
            run_id: Training run ID
            metric_name: Optional metric name filter
            metric_type: Optional metric type filter
            limit: Optional limit on results

        Returns:
            List of TrainingMetric objects

        Raises:
            DatabaseError: If query fails
        """
        try:
            conditions = ["run_id = ?"]
            params = [run_id]

            if metric_name:
                conditions.append("metric_name = ?")
                params.append(metric_name)

            if metric_type:
                conditions.append("metric_type = ?")
                params.append(metric_type.value)

            where_clause = " AND ".join(conditions)
            limit_clause = f"LIMIT {limit}" if limit else ""

            sql = f"""
            SELECT * FROM training_metrics
            WHERE {where_clause}
            ORDER BY step ASC
            {limit_clause}
            """

            result = self.db_manager.system_query(sql, params)
            return [self._row_to_metric(row) for row in result.rows]
        except Exception as e:
            raise DatabaseError(f"Failed to get metrics for run {run_id}: {e}") from e

    def get_latest_metric(
        self, run_id: str, metric_name: str, metric_type: MetricType
    ) -> TrainingMetric | None:
        """Get the latest metric value for a run.

        Args:
            run_id: Training run ID
            metric_name: Metric name
            metric_type: Metric type

        Returns:
            Latest TrainingMetric object if found, None otherwise

        Raises:
            DatabaseError: If query fails
        """
        try:
            sql = """
            SELECT * FROM training_metrics
            WHERE run_id = ? AND metric_name = ? AND metric_type = ?
            ORDER BY step DESC
            LIMIT 1
            """

            params = [run_id, metric_name, metric_type.value]
            result = self.db_manager.system_query(sql, params)

            if result.empty():
                return None

            return self._row_to_metric(result.first())
        except Exception as e:
            raise DatabaseError(f"Failed to get latest metric: {e}") from e

    # ========== TrainingCheckpoint Operations ==========

    def create_checkpoint(
        self,
        run_id: str,
        epoch: int,
        step: int,
        checkpoint_path: str,
        metrics: dict[str, float] | None = None,
        is_best: bool = False,
        file_size_bytes: int | None = None,
    ) -> TrainingCheckpoint:
        """Create a checkpoint record.

        Args:
            run_id: Training run ID
            epoch: Epoch number
            step: Step number
            checkpoint_path: Path to checkpoint file
            metrics: Optional metrics at checkpoint time
            is_best: Whether this is the best checkpoint
            file_size_bytes: Optional file size

        Returns:
            Created TrainingCheckpoint object

        Raises:
            DatabaseError: If checkpoint creation fails
        """
        try:
            checkpoint_id = str(uuid4())
            now = datetime.now(UTC)
            metrics_json = json.dumps(metrics) if metrics else None

            sql = """
            INSERT INTO training_checkpoints (
                checkpoint_id, run_id, epoch, step,
                checkpoint_path, metrics, is_best,
                file_size_bytes, status, created_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """

            params = [
                checkpoint_id,
                run_id,
                epoch,
                step,
                checkpoint_path,
                metrics_json,
                is_best,
                file_size_bytes,
                CheckpointStatus.SAVED.value,
                now,
            ]

            self.db_manager.system_execute(sql, params)

            return TrainingCheckpoint(
                checkpoint_id=checkpoint_id,
                run_id=run_id,
                epoch=epoch,
                step=step,
                checkpoint_path=checkpoint_path,
                metrics=metrics_json,
                is_best=is_best,
                file_size_bytes=file_size_bytes,
                status=CheckpointStatus.SAVED,
                created_at=now,
            )
        except Exception as e:
            raise DatabaseError(f"Failed to create checkpoint: {e}") from e

    def get_checkpoints(self, run_id: str) -> list[TrainingCheckpoint]:
        """Get all checkpoints for a training run.

        Args:
            run_id: Training run ID

        Returns:
            List of TrainingCheckpoint objects

        Raises:
            DatabaseError: If query fails
        """
        try:
            sql = """
            SELECT * FROM training_checkpoints
            WHERE run_id = ?
            ORDER BY epoch DESC
            """

            result = self.db_manager.system_query(sql, [run_id])
            return [self._row_to_checkpoint(row) for row in result.rows]
        except Exception as e:
            raise DatabaseError(
                f"Failed to get checkpoints for run {run_id}: {e}"
            ) from e

    def get_best_checkpoint(self, run_id: str) -> TrainingCheckpoint | None:
        """Get the best checkpoint for a training run.

        Args:
            run_id: Training run ID

        Returns:
            Best TrainingCheckpoint object if found, None otherwise

        Raises:
            DatabaseError: If query fails
        """
        try:
            sql = """
            SELECT * FROM training_checkpoints
            WHERE run_id = ? AND is_best = ?
            LIMIT 1
            """

            result = self.db_manager.system_query(sql, [run_id, True])

            if result.empty():
                return None

            return self._row_to_checkpoint(result.first())
        except Exception as e:
            raise DatabaseError(f"Failed to get best checkpoint: {e}") from e

    def update_checkpoint_status(
        self, checkpoint_id: str, status: CheckpointStatus
    ) -> None:
        """Update checkpoint status.

        Args:
            checkpoint_id: Checkpoint ID
            status: New status

        Raises:
            DatabaseError: If update fails
        """
        try:
            sql = "UPDATE training_checkpoints SET status = ? WHERE checkpoint_id = ?"
            params = [status.value, checkpoint_id]
            self.db_manager.system_execute(sql, params)
        except Exception as e:
            raise DatabaseError(f"Failed to update checkpoint status: {e}") from e

    def mark_checkpoint_as_best(self, checkpoint_id: str, run_id: str) -> None:
        """Mark a checkpoint as best (unmarks previous best).

        Args:
            checkpoint_id: Checkpoint ID to mark as best
            run_id: Training run ID

        Raises:
            DatabaseError: If update fails
        """
        try:
            # Unmark all previous best checkpoints for this run
            sql1 = """
            UPDATE training_checkpoints
            SET is_best = ?
            WHERE run_id = ? AND is_best = ?
            """
            self.db_manager.system_execute(sql1, [False, run_id, True])

            # Mark the new best checkpoint
            sql2 = "UPDATE training_checkpoints SET is_best = ? WHERE checkpoint_id = ?"
            self.db_manager.system_execute(sql2, [True, checkpoint_id])
        except Exception as e:
            raise DatabaseError(f"Failed to mark checkpoint as best: {e}") from e

    # ========== Helper Methods ==========

    def _row_to_training_run(self, row: dict[str, Any]) -> TrainingRun:
        """Convert database row to TrainingRun object.

        Args:
            row: Database row as dictionary

        Returns:
            TrainingRun object

        Raises:
            DatabaseError: If conversion fails
        """
        try:
            return TrainingRun(
                run_id=str(row["run_id"]),
                job_id=str(row["job_id"]) if row.get("job_id") else None,
                model_id=str(row["model_id"]) if row.get("model_id") else None,
                trainer_id=str(row["trainer_id"]) if row.get("trainer_id") else None,
                run_name=str(row["run_name"]) if row.get("run_name") else None,
                description=str(row["description"]) if row.get("description") else None,
                tensorboard_enabled=bool(row["tensorboard_enabled"]),
                tensorboard_log_dir=(
                    str(row["tensorboard_log_dir"])
                    if row.get("tensorboard_log_dir")
                    else None
                ),
                metric_log_frequency=int(row["metric_log_frequency"]),
                checkpoint_frequency=int(row["checkpoint_frequency"]),
                status=TrainingStatus.from_string(row["status"]),
                started_at=(
                    self._parse_timestamp(row["started_at"])
                    if row.get("started_at")
                    else None
                ),
                paused_at=(
                    self._parse_timestamp(row["paused_at"])
                    if row.get("paused_at")
                    else None
                ),
                resumed_at=(
                    self._parse_timestamp(row["resumed_at"])
                    if row.get("resumed_at")
                    else None
                ),
                completed_at=(
                    self._parse_timestamp(row["completed_at"])
                    if row.get("completed_at")
                    else None
                ),
                artifact_path=(
                    str(row["artifact_path"]) if row.get("artifact_path") else None
                ),
                final_metrics=(
                    str(row["final_metrics"]) if row.get("final_metrics") else None
                ),
                original_config=(
                    str(row["original_config"]) if row.get("original_config") else None
                ),
                current_config=(
                    str(row["current_config"]) if row.get("current_config") else None
                ),
                config_history=(
                    str(row["config_history"]) if row.get("config_history") else None
                ),
                created_at=self._parse_timestamp(row["created_at"]),
                updated_at=self._parse_timestamp(row["updated_at"]),
            )
        except Exception as e:
            raise DatabaseError(f"Failed to convert row to TrainingRun: {e}") from e

    def _row_to_metric(self, row: dict[str, Any]) -> TrainingMetric:
        """Convert database row to TrainingMetric object.

        Args:
            row: Database row as dictionary

        Returns:
            TrainingMetric object

        Raises:
            DatabaseError: If conversion fails
        """
        try:
            return TrainingMetric(
                metric_id=str(row["metric_id"]),
                run_id=str(row["run_id"]),
                metric_name=str(row["metric_name"]),
                metric_type=MetricType.from_string(row["metric_type"]),
                step=int(row["step"]),
                epoch=int(row["epoch"]),
                value=float(row["value"]),
                timestamp=self._parse_timestamp(row["timestamp"]),
            )
        except Exception as e:
            raise DatabaseError(f"Failed to convert row to TrainingMetric: {e}") from e

    def _row_to_checkpoint(self, row: dict[str, Any]) -> TrainingCheckpoint:
        """Convert database row to TrainingCheckpoint object.

        Args:
            row: Database row as dictionary

        Returns:
            TrainingCheckpoint object

        Raises:
            DatabaseError: If conversion fails
        """
        try:
            return TrainingCheckpoint(
                checkpoint_id=str(row["checkpoint_id"]),
                run_id=str(row["run_id"]),
                epoch=int(row["epoch"]),
                step=int(row["step"]),
                checkpoint_path=str(row["checkpoint_path"]),
                metrics=str(row["metrics"]) if row.get("metrics") else None,
                is_best=bool(row["is_best"]),
                file_size_bytes=(
                    int(row["file_size_bytes"]) if row.get("file_size_bytes") else None
                ),
                status=CheckpointStatus.from_string(row["status"]),
                created_at=self._parse_timestamp(row["created_at"]),
            )
        except Exception as e:
            raise DatabaseError(
                f"Failed to convert row to TrainingCheckpoint: {e}"
            ) from e

    def _parse_timestamp(self, ts: Any) -> datetime:
        """Parse timestamp from database.

        Args:
            ts: Timestamp value (string or datetime)

        Returns:
            datetime object

        Raises:
            DatabaseError: If parsing fails
        """
        try:
            if isinstance(ts, datetime):
                return ts
            if isinstance(ts, str):
                return datetime.fromisoformat(ts)
            return datetime.now(UTC)
        except Exception as e:
            raise DatabaseError(f"Failed to parse timestamp {ts}: {e}") from e
