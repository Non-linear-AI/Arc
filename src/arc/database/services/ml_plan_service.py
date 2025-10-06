"""Service for managing ML plans in the database."""

from datetime import UTC, datetime

from arc.database.base import Database
from arc.database.models.ml_plan import MLPlan


class MLPlanService:
    """Service for CRUD operations on ML plans."""

    def __init__(self, db: Database):
        """Initialize the ML plan service.

        Args:
            db: Database instance for executing queries
        """
        self.db = db

    def create_plan(self, plan: MLPlan) -> None:
        """Create a new ML plan in the database.

        Args:
            plan: MLPlan object to create

        Raises:
            DatabaseError: If plan creation fails
        """
        sql = """
            INSERT INTO ml_plans (
                plan_id, version, user_context, data_table, target_column,
                plan_json, status, created_at, updated_at, parent_plan_id
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """
        self.db.execute(
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
                plan.parent_plan_id,
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
                   plan_json, status, created_at, updated_at, parent_plan_id
            FROM ml_plans
            WHERE plan_id = ?
        """
        result = self.db.query(sql, [plan_id])

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
            parent_plan_id=row["parent_plan_id"],
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
                       plan_json, status, created_at, updated_at, parent_plan_id
                FROM ml_plans
                WHERE data_table = ? AND target_column = ?
                ORDER BY created_at DESC
                LIMIT 1
            """
            result = self.db.query(sql, [data_table, target_column])
        else:
            sql = """
                SELECT plan_id, version, user_context, data_table, target_column,
                       plan_json, status, created_at, updated_at, parent_plan_id
                FROM ml_plans
                WHERE data_table = ?
                ORDER BY created_at DESC
                LIMIT 1
            """
            result = self.db.query(sql, [data_table])

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
            parent_plan_id=row["parent_plan_id"],
        )

    def update_plan(self, plan: MLPlan) -> None:
        """Update an existing ML plan.

        Args:
            plan: MLPlan object with updated values

        Raises:
            DatabaseError: If update fails
        """
        sql = """
            UPDATE ml_plans
            SET user_context = ?, data_table = ?, target_column = ?,
                plan_json = ?, status = ?, updated_at = ?,
                parent_plan_id = ?
            WHERE plan_id = ?
        """
        self.db.execute(
            sql,
            [
                plan.user_context,
                plan.data_table,
                plan.target_column,
                plan.plan_json,
                plan.status,
                plan.updated_at,
                plan.parent_plan_id,
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
            FROM ml_plans
            WHERE plan_id LIKE ?
        """
        result = self.db.query(sql, [f"{base_name}-%"])

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
                   plan_json, status, created_at, updated_at, parent_plan_id
            FROM ml_plans
            {where_clause}
            ORDER BY created_at DESC
            LIMIT ?
        """
        params.append(limit)

        result = self.db.query(sql, params if params else None)

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
                parent_plan_id=row["parent_plan_id"],
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
        sql = "DELETE FROM ml_plans WHERE plan_id = ?"
        self.db.execute(sql, [plan_id])

    def mark_plan_implemented(self, plan_id: str) -> None:
        """Mark a plan as implemented after model generation.

        Args:
            plan_id: Unique plan identifier

        Raises:
            DatabaseError: If update fails
        """
        sql = """
            UPDATE ml_plans
            SET status = 'implemented', updated_at = ?
            WHERE plan_id = ?
        """
        self.db.execute(sql, [datetime.now(UTC), plan_id])
