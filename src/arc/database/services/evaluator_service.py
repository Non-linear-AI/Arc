"""Evaluator service for managing Arc evaluator metadata."""

from datetime import datetime
from typing import Any

from arc.database.base import DatabaseError
from arc.database.models.evaluator import Evaluator
from arc.database.services.base import BaseService


class EvaluatorService(BaseService):
    """Service for managing evaluator metadata in the system database.

    Handles operations on the evaluators table including:
    - Evaluator registration and versioning
    - Evaluator-model relationship tracking
    - Evaluator lifecycle management
    - CRUD operations with proper SQL escaping
    """

    def __init__(self, db_manager):
        """Initialize EvaluatorService.

        Args:
            db_manager: DatabaseManager instance
        """
        super().__init__(db_manager)

    def list_all_evaluators(self) -> list[Evaluator]:
        """List all evaluators ordered by creation date (newest first).

        Returns:
            List of Evaluator objects ordered by created_at DESC

        Raises:
            DatabaseError: If query execution fails
        """
        try:
            result = self._system_query(
                "SELECT * FROM evaluators ORDER BY created_at DESC"
            )
            return self._results_to_evaluators(result)
        except Exception as e:
            raise DatabaseError(f"Failed to list evaluators: {e}") from e

    def get_evaluator_by_id(self, id: str) -> Evaluator | None:
        """Get an evaluator by its ID.

        Args:
            id: Evaluator ID to search for

        Returns:
            Evaluator object if found, None otherwise

        Raises:
            DatabaseError: If query execution fails
        """
        try:
            sql = f"SELECT * FROM evaluators WHERE id = '{self._escape_string(id)}'"
            result = self._system_query(sql)
            if result.empty():
                return None
            return self._result_to_evaluator(result.first())
        except Exception as e:
            raise DatabaseError(f"Failed to get evaluator by id {id}: {e}") from e

    def get_evaluator_by_name_version(
        self, name: str, version: int
    ) -> Evaluator | None:
        """Get an evaluator by name and version.

        Args:
            name: Evaluator name
            version: Evaluator version

        Returns:
            Evaluator object if found, None otherwise

        Raises:
            DatabaseError: If query execution fails
        """
        try:
            escaped_name = self._escape_string(name)
            sql = f"""SELECT * FROM evaluators WHERE name = '{escaped_name}'
                AND version = {version}"""
            result = self._system_query(sql)
            if result.empty():
                return None
            return self._result_to_evaluator(result.first())
        except Exception as e:
            msg = f"Failed to get evaluator {name} version {version}: {e}"
            raise DatabaseError(msg) from e

    def get_latest_evaluator_by_name(self, name: str) -> Evaluator | None:
        """Get the latest version of an evaluator by name.

        Args:
            name: Evaluator name

        Returns:
            Latest Evaluator object if found, None otherwise

        Raises:
            DatabaseError: If query execution fails
        """
        try:
            escaped_name = self._escape_string(name)
            sql = f"""SELECT * FROM evaluators WHERE name = '{escaped_name}'
                ORDER BY version DESC LIMIT 1"""
            result = self._system_query(sql)
            if result.empty():
                return None
            return self._result_to_evaluator(result.first())
        except Exception as e:
            raise DatabaseError(
                f"Failed to get latest evaluator for {name}: {e}"
            ) from e

    def get_evaluators_by_name(self, name: str) -> list[Evaluator]:
        """Get all versions of an evaluator by name.

        Args:
            name: Evaluator name

        Returns:
            List of Evaluator objects ordered by version DESC

        Raises:
            DatabaseError: If query execution fails
        """
        try:
            escaped_name = self._escape_string(name)
            sql = f"""SELECT * FROM evaluators WHERE name = '{escaped_name}'
                ORDER BY version DESC"""
            result = self._system_query(sql)
            return self._results_to_evaluators(result)
        except Exception as e:
            raise DatabaseError(f"Failed to get evaluators by name {name}: {e}") from e

    def get_evaluators_by_model(self, model_id: str) -> list[Evaluator]:
        """Get all evaluators for a specific model.

        Args:
            model_id: Model ID to search for

        Returns:
            List of Evaluator objects for the model

        Raises:
            DatabaseError: If query execution fails
        """
        try:
            escaped_model_id = self._escape_string(model_id)
            sql = f"""SELECT * FROM evaluators WHERE model_id = '{escaped_model_id}'
                ORDER BY created_at DESC"""
            result = self._system_query(sql)
            return self._results_to_evaluators(result)
        except Exception as e:
            raise DatabaseError(
                f"Failed to get evaluators for model {model_id}: {e}"
            ) from e

    def create_evaluator(self, evaluator: Evaluator) -> None:
        """Create a new evaluator in the database.

        Args:
            evaluator: Evaluator object to create

        Raises:
            DatabaseError: If evaluator creation fails
        """
        try:
            sql = self._build_evaluator_insert_sql(evaluator)
            self._system_execute(sql)
        except Exception as e:
            raise DatabaseError(
                f"Failed to create evaluator {evaluator.id}: {e}"
            ) from e

    def update_evaluator(self, evaluator: Evaluator) -> None:
        """Update an existing evaluator in the database.

        Args:
            evaluator: Evaluator object with updated data

        Raises:
            DatabaseError: If evaluator update fails
        """
        try:
            sql = self._build_evaluator_update_sql(evaluator)
            self._system_execute(sql)
        except Exception as e:
            raise DatabaseError(
                f"Failed to update evaluator {evaluator.id}: {e}"
            ) from e

    def delete_evaluator(self, id: str) -> None:
        """Delete an evaluator by ID.

        Args:
            id: Evaluator ID to delete

        Raises:
            DatabaseError: If evaluator deletion fails
        """
        try:
            sql = f"DELETE FROM evaluators WHERE id = '{self._escape_string(id)}'"
            self._system_execute(sql)
        except Exception as e:
            raise DatabaseError(f"Failed to delete evaluator {id}: {e}") from e

    def evaluator_exists(self, id: str) -> bool:
        """Check if an evaluator exists by ID.

        Args:
            id: Evaluator ID to check

        Returns:
            True if evaluator exists, False otherwise

        Raises:
            DatabaseError: If query execution fails
        """
        return self.get_evaluator_by_id(id) is not None

    def evaluator_name_version_exists(self, name: str, version: int) -> bool:
        """Check if an evaluator with specific name and version exists.

        Args:
            name: Evaluator name
            version: Evaluator version

        Returns:
            True if evaluator exists, False otherwise

        Raises:
            DatabaseError: If query execution fails
        """
        return self.get_evaluator_by_name_version(name, version) is not None

    def get_next_version_for_name(self, name: str) -> int:
        """Get the next version number for an evaluator name.

        Uses a direct SQL MAX query to ensure we get the latest version
        even in concurrent scenarios.

        Args:
            name: Evaluator name

        Returns:
            Next version number (max_version + 1)

        Raises:
            DatabaseError: If version calculation fails
        """
        try:
            escaped_name = self._escape_string(name)
            sql = f"""SELECT COALESCE(MAX(version), 0) as max_version
                     FROM evaluators
                     WHERE name = '{escaped_name}'"""
            result = self._system_query(sql)

            if result.empty():
                return 1

            max_version = result.first()["max_version"]
            return int(max_version) + 1
        except Exception as e:
            raise DatabaseError(f"Failed to get next version for {name}: {e}") from e

    def get_next_version_for_id_prefix(self, id_prefix: str) -> int:
        """Get the next version number for an evaluator ID prefix (slug).

        Queries by ID pattern to ensure version lookup matches the ID namespace.
        This prevents conflicts when different names slugify to the same ID.

        Args:
            id_prefix: Evaluator ID prefix (slugified name)

        Returns:
            Next version number (max_version + 1)

        Raises:
            DatabaseError: If version calculation fails
        """
        try:
            escaped_prefix = self._escape_string(id_prefix)
            # Query for IDs that match the pattern: prefix-v*
            sql = f"""SELECT COALESCE(MAX(version), 0) as max_version
                     FROM evaluators
                     WHERE id LIKE '{escaped_prefix}-v%'"""
            result = self._system_query(sql)

            if result.empty():
                return 1

            max_version = result.first()["max_version"]
            return int(max_version) + 1
        except Exception as e:
            raise DatabaseError(
                f"Failed to get next version for ID prefix {id_prefix}: {e}"
            ) from e

    def _result_to_evaluator(self, row: dict[str, Any]) -> Evaluator:
        """Convert a database row to an Evaluator object.

        Args:
            row: Database row as dictionary

        Returns:
            Evaluator object created from row data

        Raises:
            DatabaseError: If conversion fails
        """
        try:
            # Handle timestamp conversion
            created_at = row.get("created_at")
            updated_at = row.get("updated_at")

            if isinstance(created_at, str):
                created_at = datetime.fromisoformat(created_at)
            elif created_at is None:
                created_at = datetime.now()

            if isinstance(updated_at, str):
                updated_at = datetime.fromisoformat(updated_at)
            elif updated_at is None:
                updated_at = datetime.now()

            return Evaluator(
                id=str(row["id"]),
                name=str(row["name"]),
                version=int(row["version"]),
                model_id=str(row["model_id"]),
                spec=str(row["spec"]),
                description=str(row["description"]),
                plan_id=str(row["plan_id"]) if row.get("plan_id") else None,
                created_at=created_at,
                updated_at=updated_at,
            )
        except (KeyError, ValueError, TypeError) as e:
            raise DatabaseError(f"Failed to convert row to Evaluator: {e}") from e

    def _results_to_evaluators(self, result) -> list[Evaluator]:
        """Convert query results to list of Evaluator objects.

        Args:
            result: QueryResult object from database query

        Returns:
            List of Evaluator objects

        Raises:
            DatabaseError: If conversion fails
        """
        try:
            return [self._result_to_evaluator(row) for row in result.rows]
        except Exception as e:
            raise DatabaseError(f"Failed to convert results to evaluators: {e}") from e

    def _build_evaluator_insert_sql(self, evaluator: Evaluator) -> str:
        """Build INSERT SQL statement for an evaluator.

        Args:
            evaluator: Evaluator object to insert

        Returns:
            SQL INSERT statement string

        Raises:
            DatabaseError: If SQL building fails
        """
        try:
            # Format timestamps for SQL
            created_at_str = evaluator.created_at.isoformat()
            updated_at_str = evaluator.updated_at.isoformat()

            plan_id_value = (
                f"'{self._escape_string(evaluator.plan_id)}'"
                if evaluator.plan_id
                else "NULL"
            )

            sql = f"""INSERT INTO evaluators (
                id, name, version, model_id,
                spec, description, plan_id, created_at, updated_at
            ) VALUES (
                '{self._escape_string(evaluator.id)}',
                '{self._escape_string(evaluator.name)}',
                {evaluator.version},
                '{self._escape_string(evaluator.model_id)}',
                '{self._escape_string(evaluator.spec)}',
                '{self._escape_string(evaluator.description)}',
                {plan_id_value},
                '{created_at_str}',
                '{updated_at_str}'
            )"""

            return sql
        except Exception as e:
            raise DatabaseError(f"Failed to build insert SQL: {e}") from e

    def _build_evaluator_update_sql(self, evaluator: Evaluator) -> str:
        """Build UPDATE SQL statement for an evaluator.

        Args:
            evaluator: Evaluator object with updated data

        Returns:
            SQL UPDATE statement string

        Raises:
            DatabaseError: If SQL building fails
        """
        try:
            # Format timestamps for SQL
            updated_at_str = evaluator.updated_at.isoformat()
            plan_id_value = (
                f"'{self._escape_string(evaluator.plan_id)}'"
                if evaluator.plan_id
                else "NULL"
            )

            sql = f"""UPDATE evaluators SET
                name = '{self._escape_string(evaluator.name)}',
                version = {evaluator.version},
                model_id = '{self._escape_string(evaluator.model_id)}',
                spec = '{self._escape_string(evaluator.spec)}',
                description = '{self._escape_string(evaluator.description)}',
                plan_id = {plan_id_value},
                updated_at = '{updated_at_str}'
            WHERE id = '{self._escape_string(evaluator.id)}'"""

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
