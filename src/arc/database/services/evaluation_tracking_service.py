"""Evaluation tracking service for managing evaluation runs and metrics."""

import json
from datetime import UTC, datetime
from typing import Any
from uuid import uuid4

from arc.database.base import DatabaseError
from arc.database.models.evaluation import EvaluationRun, EvaluationStatus
from arc.database.services.base import BaseService


class EvaluationTrackingService(BaseService):
    """Service for managing evaluation tracking data in system database.

    Handles operations on evaluation_runs table including:
    - Evaluation run lifecycle management
    - Metrics storage and retrieval
    - Status tracking and updates
    """

    def __init__(self, db_manager):
        """Initialize EvaluationTrackingService.

        Args:
            db_manager: DatabaseManager instance
        """
        super().__init__(db_manager)

    def create_run(
        self,
        evaluator_id: str,
        trainer_id: str,
        dataset: str,
        target_column: str,
        job_id: str | None = None,
    ) -> EvaluationRun:
        """Create a new evaluation run.

        Args:
            evaluator_id: Evaluator ID
            trainer_id: Trainer ID being evaluated
            dataset: Dataset table name
            target_column: Target column name
            job_id: Optional associated job ID

        Returns:
            Created EvaluationRun object

        Raises:
            DatabaseError: If run creation fails
        """
        try:
            run_id = str(uuid4())
            now = datetime.now(UTC)

            sql = """
            INSERT INTO evaluation_runs (
                run_id, evaluator_id, job_id, trainer_id,
                dataset, target_column, status,
                created_at, updated_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """

            params = [
                run_id,
                evaluator_id,
                job_id,
                trainer_id,
                dataset,
                target_column,
                EvaluationStatus.PENDING.value,
                now,
                now,
            ]

            self.db_manager.system_execute(sql, params)

            return EvaluationRun(
                run_id=run_id,
                evaluator_id=evaluator_id,
                job_id=job_id,
                trainer_id=trainer_id,
                dataset=dataset,
                target_column=target_column,
                status=EvaluationStatus.PENDING,
                started_at=None,
                completed_at=None,
                metrics_result=None,
                prediction_table=None,
                error_message=None,
                created_at=now,
                updated_at=now,
            )
        except Exception as e:
            raise DatabaseError(f"Failed to create evaluation run: {e}") from e

    def get_run_by_id(self, run_id: str) -> EvaluationRun | None:
        """Get an evaluation run by ID.

        Args:
            run_id: Evaluation run ID

        Returns:
            EvaluationRun object if found, None otherwise

        Raises:
            DatabaseError: If query fails
        """
        try:
            sql = "SELECT * FROM evaluation_runs WHERE run_id = ?"
            result = self.db_manager.system_query(sql, [run_id])

            if result.empty():
                return None

            return self._row_to_evaluation_run(result.first())
        except Exception as e:
            raise DatabaseError(f"Failed to get run {run_id}: {e}") from e

    def list_runs(
        self,
        limit: int = 100,
        status: EvaluationStatus | None = None,
        evaluator_id: str | None = None,
        trainer_id: str | None = None,
    ) -> list[EvaluationRun]:
        """List evaluation runs with optional filters.

        Args:
            limit: Maximum number of runs to return
            status: Filter by status
            evaluator_id: Filter by evaluator ID
            trainer_id: Filter by trainer ID

        Returns:
            List of EvaluationRun objects

        Raises:
            DatabaseError: If query fails
        """
        try:
            conditions = []
            params = []

            if status:
                conditions.append("status = ?")
                params.append(status.value)

            if evaluator_id:
                conditions.append("evaluator_id = ?")
                params.append(evaluator_id)

            if trainer_id:
                conditions.append("trainer_id = ?")
                params.append(trainer_id)

            where = f"WHERE {' AND '.join(conditions)}" if conditions else ""
            sql = f"""
            SELECT * FROM evaluation_runs
            {where}
            ORDER BY created_at DESC
            LIMIT ?
            """
            params.append(limit)

            result = self.db_manager.system_query(sql, params)
            return [self._row_to_evaluation_run(row) for row in result.rows]
        except Exception as e:
            raise DatabaseError(f"Failed to list runs: {e}") from e

    def update_run_status(
        self,
        run_id: str,
        status: EvaluationStatus,
        timestamp_field: str | None = None,
    ) -> None:
        """Update evaluation run status and optional timestamp.

        Args:
            run_id: Evaluation run ID
            status: New status
            timestamp_field: Optional timestamp field to update
                           (started_at, completed_at)

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
            UPDATE evaluation_runs
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

    def update_run_result(
        self,
        run_id: str,
        metrics_result: dict[str, Any],
        prediction_table: str | None = None,
    ) -> None:
        """Update evaluation run with results after successful completion.

        Args:
            run_id: Evaluation run ID
            metrics_result: Evaluation metrics as dictionary
            prediction_table: Optional name of predictions table

        Raises:
            DatabaseError: If update fails
        """
        try:
            now = datetime.now(UTC)
            metrics_json = json.dumps(metrics_result) if metrics_result else None

            sql = """
            UPDATE evaluation_runs
            SET metrics_result = ?, prediction_table = ?, updated_at = ?
            WHERE run_id = ?
            """

            params = [metrics_json, prediction_table, now, run_id]
            self.db_manager.system_execute(sql, params)
        except Exception as e:
            raise DatabaseError(f"Failed to update run result {run_id}: {e}") from e

    def update_run_error(
        self,
        run_id: str,
        error_message: str,
    ) -> None:
        """Update evaluation run with error message.

        Args:
            run_id: Evaluation run ID
            error_message: Error message

        Raises:
            DatabaseError: If update fails
        """
        try:
            now = datetime.now(UTC)

            sql = """
            UPDATE evaluation_runs
            SET error_message = ?, updated_at = ?
            WHERE run_id = ?
            """

            params = [error_message, now, run_id]
            self.db_manager.system_execute(sql, params)
        except Exception as e:
            raise DatabaseError(f"Failed to update run error {run_id}: {e}") from e

    def delete_run(self, run_id: str) -> bool:
        """Delete an evaluation run.

        Args:
            run_id: Evaluation run ID

        Returns:
            True if deleted, False if not found

        Raises:
            DatabaseError: If deletion fails
        """
        try:
            if self.get_run_by_id(run_id) is None:
                return False

            sql = "DELETE FROM evaluation_runs WHERE run_id = ?"
            self.db_manager.system_execute(sql, [run_id])
            return True
        except Exception as e:
            raise DatabaseError(f"Failed to delete run {run_id}: {e}") from e

    def _row_to_evaluation_run(self, row: dict[str, Any]) -> EvaluationRun:
        """Convert database row to EvaluationRun object.

        Args:
            row: Database row as dictionary

        Returns:
            EvaluationRun object

        Raises:
            DatabaseError: If conversion fails
        """
        try:
            return EvaluationRun(
                run_id=str(row["run_id"]),
                evaluator_id=str(row["evaluator_id"]),
                job_id=str(row["job_id"]) if row.get("job_id") else None,
                trainer_id=str(row["trainer_id"]) if row.get("trainer_id") else None,
                dataset=str(row["dataset"]) if row.get("dataset") else "",
                target_column=str(row["target_column"])
                if row.get("target_column")
                else "",
                status=EvaluationStatus.from_string(row["status"]),
                started_at=(
                    self._parse_timestamp(row["started_at"])
                    if row.get("started_at")
                    else None
                ),
                completed_at=(
                    self._parse_timestamp(row["completed_at"])
                    if row.get("completed_at")
                    else None
                ),
                metrics_result=(
                    str(row["metrics_result"]) if row.get("metrics_result") else None
                ),
                prediction_table=(
                    str(row["prediction_table"])
                    if row.get("prediction_table")
                    else None
                ),
                error_message=(
                    str(row["error_message"]) if row.get("error_message") else None
                ),
                created_at=self._parse_timestamp(row["created_at"]),
                updated_at=self._parse_timestamp(row["updated_at"]),
            )
        except Exception as e:
            msg = f"Failed to convert row to EvaluationRun: {e}"
            raise DatabaseError(msg) from e

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
