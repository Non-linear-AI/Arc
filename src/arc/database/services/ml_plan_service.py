"""Service for managing ML plans in the database."""

from datetime import UTC, datetime

from arc.database.models.ml_plan import MLPlan
from arc.database.services.base import BaseService


class MLPlanService(BaseService):
    """Service for CRUD operations on ML plans."""

    def create_plan(self, plan: MLPlan) -> None:
        """Create a new ML plan in the database.

        Args:
            plan: MLPlan object to create

        Raises:
            DatabaseError: If plan creation fails
        """
        sql = """
            INSERT INTO plans (
                plan_id, version, user_context, data_table, target_column,
                plan_json, status, created_at, updated_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """
        self.db_manager.system_execute(
            sql,
            [
                plan.plan_id,
                plan.version,
                plan.user_context,
                plan.data_table,
                plan.target_column,
                plan.plan_json,
                plan.status,
                plan.created_at,
                plan.updated_at,
            ],
        )

    def get_plan_by_id(self, plan_id: str) -> MLPlan | None:
        """Get an ML plan by its ID.

        Args:
            plan_id: Unique plan identifier

        Returns:
            MLPlan object if found, None otherwise
        """
        sql = """
            SELECT plan_id, version, user_context, data_table, target_column,
                   plan_json, status, created_at, updated_at
            FROM plans
            WHERE plan_id = ?
        """
        result = self.db_manager.system_query(sql, [plan_id])

        if not result.rows:
            return None

        row = result.rows[0]
        return MLPlan(
            plan_id=row["plan_id"],
            version=row["version"],
            user_context=row["user_context"],
            data_table=row["data_table"],
            target_column=row["target_column"],
            plan_json=row["plan_json"],
            status=row["status"],
            created_at=row["created_at"],
            updated_at=row["updated_at"],
        )

    def get_latest_plan_for_table(
        self, data_table: str, target_column: str | None = None
    ) -> MLPlan | None:
        """Get the most recent plan for a given data table.

        Args:
            data_table: Name of the data table
            target_column: Optional target column filter

        Returns:
            Most recent MLPlan for the table, or None if not found
        """
        if target_column:
            sql = """
                SELECT plan_id, version, user_context, data_table, target_column,
                       plan_json, status, created_at, updated_at
                FROM plans
                WHERE data_table = ? AND target_column = ?
                ORDER BY created_at DESC
                LIMIT 1
            """
            result = self.db_manager.system_query(sql, [data_table, target_column])
        else:
            sql = """
                SELECT plan_id, version, user_context, data_table, target_column,
                       plan_json, status, created_at, updated_at
                FROM plans
                WHERE data_table = ?
                ORDER BY created_at DESC
                LIMIT 1
            """
            result = self.db_manager.system_query(sql, [data_table])

        if not result.rows:
            return None

        row = result.rows[0]
        return MLPlan(
            plan_id=row["plan_id"],
            version=row["version"],
            user_context=row["user_context"],
            data_table=row["data_table"],
            target_column=row["target_column"],
            plan_json=row["plan_json"],
            status=row["status"],
            created_at=row["created_at"],
            updated_at=row["updated_at"],
        )

    def update_plan(self, plan: MLPlan) -> None:
        """Update an existing ML plan.

        Args:
            plan: MLPlan object with updated values

        Raises:
            DatabaseError: If update fails
        """
        sql = """
            UPDATE plans
            SET user_context = ?, data_table = ?, target_column = ?,
                plan_json = ?, status = ?, updated_at = ?
            WHERE plan_id = ?
        """
        self.db_manager.system_execute(
            sql,
            [
                plan.user_context,
                plan.data_table,
                plan.target_column,
                plan.plan_json,
                plan.status,
                plan.updated_at,
                plan.plan_id,
            ],
        )

    def get_next_version(self, base_name: str) -> int:
        """Get the next available version number for a plan base name.

        Args:
            base_name: Base name for the plan (without version suffix)

        Returns:
            Next version number (1 if no existing plans)
        """
        sql = """
            SELECT MAX(version) as max_version
            FROM plans
            WHERE plan_id LIKE ?
        """
        result = self.db_manager.system_query(sql, [f"{base_name}-%"])

        if not result.rows or result.rows[0]["max_version"] is None:
            return 1

        return result.rows[0]["max_version"] + 1

    def list_plans(
        self, data_table: str | None = None, status: str | None = None, limit: int = 50
    ) -> list[MLPlan]:
        """List ML plans with optional filters.

        Args:
            data_table: Optional filter by data table
            status: Optional filter by status
            limit: Maximum number of plans to return

        Returns:
            List of MLPlan objects
        """
        conditions = []
        params = []

        if data_table:
            conditions.append("data_table = ?")
            params.append(data_table)

        if status:
            conditions.append("status = ?")
            params.append(status)

        where_clause = f"WHERE {' AND '.join(conditions)}" if conditions else ""

        sql = f"""
            SELECT plan_id, version, user_context, data_table, target_column,
                   plan_json, status, created_at, updated_at
            FROM plans
            {where_clause}
            ORDER BY created_at DESC
            LIMIT ?
        """
        params.append(limit)

        result = self.db_manager.system_query(sql, params if params else None)

        return [
            MLPlan(
                plan_id=row["plan_id"],
                version=row["version"],
                user_context=row["user_context"],
                data_table=row["data_table"],
                target_column=row["target_column"],
                plan_json=row["plan_json"],
                status=row["status"],
                created_at=row["created_at"],
                updated_at=row["updated_at"],
            )
            for row in result.rows
        ]

    def delete_plan(self, plan_id: str) -> None:
        """Delete an ML plan from the database.

        Args:
            plan_id: Unique plan identifier

        Raises:
            DatabaseError: If deletion fails
        """
        sql = "DELETE FROM plans WHERE plan_id = ?"
        self.db_manager.system_execute(sql, [plan_id])

    def mark_plan_implemented(self, plan_id: str) -> None:
        """Mark a plan as implemented after model generation.

        Args:
            plan_id: Unique plan identifier

        Raises:
            DatabaseError: If update fails
        """
        sql = """
            UPDATE plans
            SET status = 'implemented', updated_at = ?
            WHERE plan_id = ?
        """
        self.db_manager.system_execute(sql, [datetime.now(UTC), plan_id])
