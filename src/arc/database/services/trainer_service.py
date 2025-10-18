"""Trainer service for managing Arc trainer metadata."""

from datetime import datetime
from typing import Any

from arc.database.base import DatabaseError
from arc.database.models.trainer import Trainer
from arc.database.services.base import BaseService


class TrainerService(BaseService):
    """Service for managing trainer metadata in the system database.

    Handles operations on the trainers table including:
    - Trainer registration and versioning
    - Trainer-model relationship tracking
    - Trainer lifecycle management
    - CRUD operations with proper SQL escaping
    """

    def __init__(self, db_manager):
        """Initialize TrainerService.

        Args:
            db_manager: DatabaseManager instance
        """
        super().__init__(db_manager)

    def list_all_trainers(self) -> list[Trainer]:
        """List all trainers ordered by creation date (newest first).

        Returns:
            List of Trainer objects ordered by created_at DESC

        Raises:
            DatabaseError: If query execution fails
        """
        try:
            result = self._system_query(
                "SELECT * FROM trainers ORDER BY created_at DESC"
            )
            return self._results_to_trainers(result)
        except Exception as e:
            raise DatabaseError(f"Failed to list trainers: {e}") from e

    def get_trainer_by_id(self, id: str) -> Trainer | None:
        """Get a trainer by its ID.

        Args:
            id: Trainer ID to search for

        Returns:
            Trainer object if found, None otherwise

        Raises:
            DatabaseError: If query execution fails
        """
        try:
            sql = f"SELECT * FROM trainers WHERE id = '{self._escape_string(id)}'"
            result = self._system_query(sql)
            if result.empty():
                return None
            return self._result_to_trainer(result.first())
        except Exception as e:
            raise DatabaseError(f"Failed to get trainer by id {id}: {e}") from e

    def get_trainer_by_name_version(self, name: str, version: int) -> Trainer | None:
        """Get a trainer by name and version.

        Args:
            name: Trainer name
            version: Trainer version

        Returns:
            Trainer object if found, None otherwise

        Raises:
            DatabaseError: If query execution fails
        """
        try:
            escaped_name = self._escape_string(name)
            sql = f"""SELECT * FROM trainers WHERE name = '{escaped_name}'
                AND version = {version}"""
            result = self._system_query(sql)
            if result.empty():
                return None
            return self._result_to_trainer(result.first())
        except Exception as e:
            msg = f"Failed to get trainer {name} version {version}: {e}"
            raise DatabaseError(msg) from e

    def get_latest_trainer_by_name(self, name: str) -> Trainer | None:
        """Get the latest version of a trainer by name.

        Args:
            name: Trainer name

        Returns:
            Latest Trainer object if found, None otherwise

        Raises:
            DatabaseError: If query execution fails
        """
        try:
            escaped_name = self._escape_string(name)
            sql = f"""SELECT * FROM trainers WHERE name = '{escaped_name}'
                ORDER BY version DESC LIMIT 1"""
            result = self._system_query(sql)
            if result.empty():
                return None
            return self._result_to_trainer(result.first())
        except Exception as e:
            raise DatabaseError(f"Failed to get latest trainer for {name}: {e}") from e

    def get_trainers_by_name(self, name: str) -> list[Trainer]:
        """Get all versions of a trainer by name.

        Args:
            name: Trainer name

        Returns:
            List of Trainer objects ordered by version DESC

        Raises:
            DatabaseError: If query execution fails
        """
        try:
            escaped_name = self._escape_string(name)
            sql = f"""SELECT * FROM trainers WHERE name = '{escaped_name}'
                ORDER BY version DESC"""
            result = self._system_query(sql)
            return self._results_to_trainers(result)
        except Exception as e:
            raise DatabaseError(f"Failed to get trainers by name {name}: {e}") from e

    def get_trainers_by_model(self, model_id: str) -> list[Trainer]:
        """Get all trainers for a specific model.

        Args:
            model_id: Model ID to search for

        Returns:
            List of Trainer objects for the model

        Raises:
            DatabaseError: If query execution fails
        """
        try:
            escaped_model_id = self._escape_string(model_id)
            sql = f"""SELECT * FROM trainers WHERE model_id = '{escaped_model_id}'
                ORDER BY created_at DESC"""
            result = self._system_query(sql)
            return self._results_to_trainers(result)
        except Exception as e:
            raise DatabaseError(
                f"Failed to get trainers for model {model_id}: {e}"
            ) from e

    def create_trainer(self, trainer: Trainer) -> None:
        """Create a new trainer in the database.

        Args:
            trainer: Trainer object to create

        Raises:
            DatabaseError: If trainer creation fails
        """
        try:
            sql = self._build_trainer_insert_sql(trainer)
            self._system_execute(sql)
        except Exception as e:
            raise DatabaseError(f"Failed to create trainer {trainer.id}: {e}") from e

    def update_trainer(self, trainer: Trainer) -> None:
        """Update an existing trainer in the database.

        Args:
            trainer: Trainer object with updated data

        Raises:
            DatabaseError: If trainer update fails
        """
        try:
            sql = self._build_trainer_update_sql(trainer)
            self._system_execute(sql)
        except Exception as e:
            raise DatabaseError(f"Failed to update trainer {trainer.id}: {e}") from e

    def delete_trainer(self, id: str) -> None:
        """Delete a trainer by ID.

        Args:
            id: Trainer ID to delete

        Raises:
            DatabaseError: If trainer deletion fails
        """
        try:
            sql = f"DELETE FROM trainers WHERE id = '{self._escape_string(id)}'"
            self._system_execute(sql)
        except Exception as e:
            raise DatabaseError(f"Failed to delete trainer {id}: {e}") from e

    def trainer_exists(self, id: str) -> bool:
        """Check if a trainer exists by ID.

        Args:
            id: Trainer ID to check

        Returns:
            True if trainer exists, False otherwise

        Raises:
            DatabaseError: If query execution fails
        """
        return self.get_trainer_by_id(id) is not None

    def trainer_name_version_exists(self, name: str, version: int) -> bool:
        """Check if a trainer with specific name and version exists.

        Args:
            name: Trainer name
            version: Trainer version

        Returns:
            True if trainer exists, False otherwise

        Raises:
            DatabaseError: If query execution fails
        """
        return self.get_trainer_by_name_version(name, version) is not None

    def get_next_version_for_name(self, name: str) -> int:
        """Get the next version number for a trainer name.

        Args:
            name: Trainer name

        Returns:
            Next version number (max_version + 1)

        Raises:
            DatabaseError: If version calculation fails
        """
        try:
            trainers = self.get_trainers_by_name(name)
            if not trainers:
                return 1

            max_version = max(trainer.version for trainer in trainers)
            return max_version + 1
        except Exception as e:
            raise DatabaseError(f"Failed to get next version for {name}: {e}") from e

    def _result_to_trainer(self, row: dict[str, Any]) -> Trainer:
        """Convert a database row to a Trainer object.

        Args:
            row: Database row as dictionary

        Returns:
            Trainer object created from row data

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

            return Trainer(
                id=str(row["id"]),
                name=str(row["name"]),
                version=int(row["version"]),
                model_id=str(row["model_id"]),
                model_version=int(row["model_version"]),
                spec=str(row["spec"]),
                description=str(row["description"]),
                plan_id=str(row["plan_id"]) if row.get("plan_id") else None,
                created_at=created_at,
                updated_at=updated_at,
            )
        except (KeyError, ValueError, TypeError) as e:
            raise DatabaseError(f"Failed to convert row to Trainer: {e}") from e

    def _results_to_trainers(self, result) -> list[Trainer]:
        """Convert query results to list of Trainer objects.

        Args:
            result: QueryResult object from database query

        Returns:
            List of Trainer objects

        Raises:
            DatabaseError: If conversion fails
        """
        try:
            return [self._result_to_trainer(row) for row in result.rows]
        except Exception as e:
            raise DatabaseError(f"Failed to convert results to trainers: {e}") from e

    def _build_trainer_insert_sql(self, trainer: Trainer) -> str:
        """Build INSERT SQL statement for a trainer.

        Args:
            trainer: Trainer object to insert

        Returns:
            SQL INSERT statement string

        Raises:
            DatabaseError: If SQL building fails
        """
        try:
            # Format timestamps for SQL
            created_at_str = trainer.created_at.isoformat()
            updated_at_str = trainer.updated_at.isoformat()

            plan_id_value = f"'{self._escape_string(trainer.plan_id)}'" if trainer.plan_id else "NULL"

            sql = f"""INSERT INTO trainers (
                id, name, version, model_id, model_version,
                spec, description, plan_id, created_at, updated_at
            ) VALUES (
                '{self._escape_string(trainer.id)}',
                '{self._escape_string(trainer.name)}',
                {trainer.version},
                '{self._escape_string(trainer.model_id)}',
                {trainer.model_version},
                '{self._escape_string(trainer.spec)}',
                '{self._escape_string(trainer.description)}',
                {plan_id_value},
                '{created_at_str}',
                '{updated_at_str}'
            )"""

            return sql
        except Exception as e:
            raise DatabaseError(f"Failed to build insert SQL: {e}") from e

    def _build_trainer_update_sql(self, trainer: Trainer) -> str:
        """Build UPDATE SQL statement for a trainer.

        Args:
            trainer: Trainer object with updated data

        Returns:
            SQL UPDATE statement string

        Raises:
            DatabaseError: If SQL building fails
        """
        try:
            # Format timestamps for SQL
            updated_at_str = trainer.updated_at.isoformat()
            plan_id_value = f"'{self._escape_string(trainer.plan_id)}'" if trainer.plan_id else "NULL"

            sql = f"""UPDATE trainers SET
                name = '{self._escape_string(trainer.name)}',
                version = {trainer.version},
                model_id = '{self._escape_string(trainer.model_id)}',
                model_version = {trainer.model_version},
                spec = '{self._escape_string(trainer.spec)}',
                description = '{self._escape_string(trainer.description)}',
                plan_id = {plan_id_value},
                updated_at = '{updated_at_str}'
            WHERE id = '{self._escape_string(trainer.id)}'"""

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
