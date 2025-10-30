"""Service for managing plan execution records."""

import json
from datetime import datetime
from typing import Any, TYPE_CHECKING

if TYPE_CHECKING:
    from arc.database.manager import DatabaseManager


class PlanExecutionService:
    """Service for storing and retrieving plan execution records.

    Tracks execution of plan steps (data processing, training, evaluation, etc.)
    with their SQL/YAML context and outputs.
    """

    def __init__(self, db_manager: "DatabaseManager"):
        """Initialize service.

        Args:
            db_manager: DatabaseManager instance
        """
        self.db_manager = db_manager

    def store_execution(
        self,
        execution_id: str,
        plan_id: str,
        step_type: str,
        context: str,
        outputs: list[dict[str, Any]],
        status: str = "completed",
        error_message: str | None = None
    ) -> None:
        """Store execution record.

        Args:
            execution_id: Unique execution ID
            plan_id: Plan this execution belongs to
            step_type: Type of step (data_processing, training, evaluation, etc.)
            context: Execution context (SQL for data processing, YAML for training, etc.)
            outputs: List of outputs (tables, models, metrics, etc.)
            status: Execution status (completed, failed)
            error_message: Error message if failed
        """
        now = datetime.now()

        self.db_manager.system_execute("""
            INSERT INTO plan_executions
            (id, plan_id, step_type, status, started_at, completed_at, context, outputs, error_message)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, [
            execution_id,
            plan_id,
            step_type,
            status,
            now,
            now,
            context,
            json.dumps(outputs),
            error_message
        ])

    def get_execution(self, execution_id: str) -> dict[str, Any] | None:
        """Load execution by ID.

        Args:
            execution_id: Execution ID to load

        Returns:
            Execution record with context and outputs, or None if not found
        """
        result = self.db_manager.system_query("""
            SELECT id, plan_id, step_type, status, started_at, completed_at,
                   context, outputs, error_message
            FROM plan_executions
            WHERE id = ?
        """, [execution_id])

        if result.empty():
            return None

        row = result.first()
        return {
            "id": row["id"],
            "plan_id": row["plan_id"],
            "step_type": row["step_type"],
            "status": row["status"],
            "started_at": row["started_at"],
            "completed_at": row["completed_at"],
            "context": row["context"],
            "outputs": json.loads(row["outputs"]),
            "error_message": row["error_message"]
        }

    def get_latest_execution(
        self,
        plan_id: str,
        step_type: str
    ) -> dict[str, Any] | None:
        """Get most recent execution of a step type for a plan.

        Args:
            plan_id: Plan ID
            step_type: Step type to find

        Returns:
            Latest execution record, or None if not found
        """
        result = self.db_manager.system_query("""
            SELECT id, plan_id, step_type, status, started_at, completed_at,
                   context, outputs, error_message
            FROM plan_executions
            WHERE plan_id = ? AND step_type = ? AND status = 'completed'
            ORDER BY completed_at DESC
            LIMIT 1
        """, [plan_id, step_type])

        if result.empty():
            return None

        row = result.first()
        return {
            "id": row["id"],
            "plan_id": row["plan_id"],
            "step_type": row["step_type"],
            "status": row["status"],
            "started_at": row["started_at"],
            "completed_at": row["completed_at"],
            "context": row["context"],
            "outputs": json.loads(row["outputs"]),
            "error_message": row["error_message"]
        }

    def get_all_executions(self, plan_id: str) -> list[dict[str, Any]]:
        """Get all completed executions for a plan, ordered by completion time.

        Args:
            plan_id: Plan ID

        Returns:
            List of execution records
        """
        result = self.db_manager.system_query("""
            SELECT id, plan_id, step_type, status, started_at, completed_at,
                   context, outputs, error_message
            FROM plan_executions
            WHERE plan_id = ? AND status = 'completed'
            ORDER BY completed_at ASC
        """, [plan_id])

        return [
            {
                "id": row["id"],
                "plan_id": row["plan_id"],
                "step_type": row["step_type"],
                "status": row["status"],
                "started_at": row["started_at"],
                "completed_at": row["completed_at"],
                "context": row["context"],
                "outputs": json.loads(row["outputs"]),
                "error_message": row["error_message"]
            }
            for row in result.rows
        ]
