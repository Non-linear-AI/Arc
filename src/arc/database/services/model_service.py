"""Model service for managing Arc model metadata."""

from datetime import datetime
from typing import Any

from arc.database.base import DatabaseError
from arc.database.models.model import Model
from arc.database.services.base import BaseService


class ModelService(BaseService):
    """Service for managing model metadata in the system database.

    Handles operations on the models table including:
    - Model registration and versioning
    - Model inheritance relationships
    - Model lifecycle management
    - CRUD operations with proper SQL escaping
    """

    def __init__(self, db_manager):
        """Initialize ModelService.

        Args:
            db_manager: DatabaseManager instance
        """
        super().__init__(db_manager)

    def list_all_models(self) -> list[Model]:
        """List all models ordered by creation date (newest first).

        Returns:
            List of Model objects ordered by created_at DESC

        Raises:
            DatabaseError: If query execution fails
        """
        try:
            result = self._system_query("SELECT * FROM models ORDER BY created_at DESC")
            return self._results_to_models(result)
        except Exception as e:
            raise DatabaseError(f"Failed to list models: {e}") from e

    def get_model_by_id(self, id: str) -> Model | None:
        """Get a model by its ID.

        Args:
            id: Model ID to search for

        Returns:
            Model object if found, None otherwise

        Raises:
            DatabaseError: If query execution fails
        """
        try:
            sql = f"SELECT * FROM models WHERE id = '{self._escape_string(id)}'"
            result = self._system_query(sql)
            if result.empty():
                return None
            return self._result_to_model(result.first())
        except Exception as e:
            raise DatabaseError(f"Failed to get model by id {id}: {e}") from e

    def get_model_by_name_version(self, name: str, version: int) -> Model | None:
        """Get a model by name and version.

        Args:
            name: Model name
            version: Model version

        Returns:
            Model object if found, None otherwise

        Raises:
            DatabaseError: If query execution fails
        """
        try:
            escaped_name = self._escape_string(name)
            sql = f"""SELECT * FROM models WHERE name = '{escaped_name}'
                AND version = {version}"""
            result = self._system_query(sql)
            if result.empty():
                return None
            return self._result_to_model(result.first())
        except Exception as e:
            msg = f"Failed to get model {name} version {version}: {e}"
            raise DatabaseError(msg) from e

    def get_latest_model_by_name(self, name: str) -> Model | None:
        """Get the latest version of a model by name.

        Args:
            name: Model name

        Returns:
            Latest Model object if found, None otherwise

        Raises:
            DatabaseError: If query execution fails
        """
        try:
            escaped_name = self._escape_string(name)
            sql = f"""SELECT * FROM models WHERE name = '{escaped_name}'
                ORDER BY version DESC LIMIT 1"""
            result = self._system_query(sql)
            if result.empty():
                return None
            return self._result_to_model(result.first())
        except Exception as e:
            raise DatabaseError(f"Failed to get latest model for {name}: {e}") from e

    def get_models_by_name(self, name: str) -> list[Model]:
        """Get all versions of a model by name.

        Args:
            name: Model name

        Returns:
            List of Model objects ordered by version DESC

        Raises:
            DatabaseError: If query execution fails
        """
        try:
            escaped_name = self._escape_string(name)
            sql = f"""SELECT * FROM models WHERE name = '{escaped_name}'
                ORDER BY version DESC"""
            result = self._system_query(sql)
            return self._results_to_models(result)
        except Exception as e:
            raise DatabaseError(f"Failed to get models by name {name}: {e}") from e

    def create_model(self, model: Model) -> None:
        """Create a new model in the database.

        Args:
            model: Model object to create

        Raises:
            DatabaseError: If model creation fails
        """
        try:
            sql = self._build_model_insert_sql(model)
            self._system_execute(sql)
        except Exception as e:
            raise DatabaseError(f"Failed to create model {model.id}: {e}") from e

    def update_model(self, model: Model) -> None:
        """Update an existing model in the database.

        Args:
            model: Model object with updated data

        Raises:
            DatabaseError: If model update fails
        """
        try:
            sql = self._build_model_update_sql(model)
            self._system_execute(sql)
        except Exception as e:
            raise DatabaseError(f"Failed to update model {model.id}: {e}") from e

    def delete_model(self, id: str) -> None:
        """Delete a model by ID.

        Args:
            id: Model ID to delete

        Raises:
            DatabaseError: If model deletion fails
        """
        try:
            sql = f"DELETE FROM models WHERE id = '{self._escape_string(id)}'"
            self._system_execute(sql)
        except Exception as e:
            raise DatabaseError(f"Failed to delete model {id}: {e}") from e

    def model_exists(self, id: str) -> bool:
        """Check if a model exists by ID.

        Args:
            id: Model ID to check

        Returns:
            True if model exists, False otherwise

        Raises:
            DatabaseError: If query execution fails
        """
        return self.get_model_by_id(id) is not None

    def model_name_version_exists(self, name: str, version: int) -> bool:
        """Check if a model with specific name and version exists.

        Args:
            name: Model name
            version: Model version

        Returns:
            True if model exists, False otherwise

        Raises:
            DatabaseError: If query execution fails
        """
        return self.get_model_by_name_version(name, version) is not None

    def generate_next_model_id(self) -> str:
        """Generate the next sequential model ID.

        Returns:
            Next available model ID as string

        Raises:
            DatabaseError: If ID generation fails
        """
        try:
            sql = """SELECT COALESCE(MAX(CAST(id AS INTEGER)), 0) + 1 AS next_id
                FROM models"""
            result = self._system_query(sql)
            if result.empty():
                return "1"
            next_id = result.first().get("next_id", 1)
            return str(next_id)
        except Exception as e:
            raise DatabaseError(f"Failed to generate next model ID: {e}") from e

    def get_next_version_for_name(self, name: str) -> int:
        """Get the next version number for a model name.

        Args:
            name: Model name

        Returns:
            Next version number (max_version + 1)

        Raises:
            DatabaseError: If version calculation fails
        """
        try:
            models = self.get_models_by_name(name)
            if not models:
                return 1

            max_version = max(model.version for model in models)
            return max_version + 1
        except Exception as e:
            raise DatabaseError(f"Failed to get next version for {name}: {e}") from e

    def _result_to_model(self, row: dict[str, Any]) -> Model:
        """Convert a database row to a Model object.

        Args:
            row: Database row as dictionary

        Returns:
            Model object created from row data

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

            return Model(
                id=str(row["id"]),
                type=str(row["type"]),
                name=str(row["name"]),
                version=int(row["version"]),
                description=str(row["description"]),
                base_model_id=row.get("base_model_id"),  # Can be None
                spec=str(row["spec"]),
                arc_graph=str(row["arc_graph"]),
                created_at=created_at,
                updated_at=updated_at,
            )
        except (KeyError, ValueError, TypeError) as e:
            raise DatabaseError(f"Failed to convert row to Model: {e}") from e

    def _results_to_models(self, result) -> list[Model]:
        """Convert query results to list of Model objects.

        Args:
            result: QueryResult object from database query

        Returns:
            List of Model objects

        Raises:
            DatabaseError: If conversion fails
        """
        try:
            return [self._result_to_model(row) for row in result.rows]
        except Exception as e:
            raise DatabaseError(f"Failed to convert results to models: {e}") from e

    def _build_model_insert_sql(self, model: Model) -> str:
        """Build INSERT SQL statement for a model.

        Args:
            model: Model object to insert

        Returns:
            SQL INSERT statement string

        Raises:
            DatabaseError: If SQL building fails
        """
        try:
            # Format timestamps for SQL
            created_at_str = model.created_at.isoformat()
            updated_at_str = model.updated_at.isoformat()

            # Handle optional base_model_id
            base_model_id_sql = (
                f"'{self._escape_string(model.base_model_id)}'"
                if model.base_model_id is not None
                else "NULL"
            )

            sql = f"""INSERT INTO models (
                id, type, name, version, description, base_model_id,
                spec, arc_graph, created_at, updated_at
            ) VALUES (
                '{self._escape_string(model.id)}',
                '{self._escape_string(model.type)}',
                '{self._escape_string(model.name)}',
                {model.version},
                '{self._escape_string(model.description)}',
                {base_model_id_sql},
                '{self._escape_string(model.spec)}',
                '{self._escape_string(model.arc_graph)}',
                '{created_at_str}',
                '{updated_at_str}'
            )"""

            return sql
        except Exception as e:
            raise DatabaseError(f"Failed to build insert SQL: {e}") from e

    def _build_model_update_sql(self, model: Model) -> str:
        """Build UPDATE SQL statement for a model.

        Args:
            model: Model object with updated data

        Returns:
            SQL UPDATE statement string

        Raises:
            DatabaseError: If SQL building fails
        """
        try:
            # Format timestamps for SQL
            updated_at_str = model.updated_at.isoformat()

            # Handle optional base_model_id
            base_model_id_sql = (
                f"'{self._escape_string(model.base_model_id)}'"
                if model.base_model_id is not None
                else "NULL"
            )

            sql = f"""UPDATE models SET
                type = '{self._escape_string(model.type)}',
                name = '{self._escape_string(model.name)}',
                version = {model.version},
                description = '{self._escape_string(model.description)}',
                base_model_id = {base_model_id_sql},
                spec = '{self._escape_string(model.spec)}',
                arc_graph = '{self._escape_string(model.arc_graph)}',
                updated_at = '{updated_at_str}'
            WHERE id = '{self._escape_string(model.id)}'"""

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
