"""Data processor service for managing Arc data processor metadata."""

from datetime import datetime
from typing import Any

from arc.database.base import DatabaseError
from arc.database.models.data_processor import DataProcessor
from arc.database.services.base import BaseService


class DataProcessorService(BaseService):
    """Service for managing data processor metadata in the system database.

    Handles operations on the data_processors table including:
    - Data processor registration and versioning
    - CRUD operations with proper SQL escaping
    """

    def __init__(self, db_manager):
        """Initialize DataProcessorService.

        Args:
            db_manager: DatabaseManager instance
        """
        super().__init__(db_manager)

    def get_data_processor_by_id(self, id: str) -> DataProcessor | None:
        """Get a data processor by its ID.

        Args:
            id: Data processor ID to search for

        Returns:
            DataProcessor object if found, None otherwise

        Raises:
            DatabaseError: If query execution fails
        """
        try:
            escaped_id = self._escape_string(id)
            sql = f"SELECT * FROM data_processors WHERE id = '{escaped_id}'"
            result = self._system_query(sql)
            if result.empty():
                return None
            return self._result_to_data_processor(result.first())
        except Exception as e:
            raise DatabaseError(f"Failed to get data processor by id {id}: {e}") from e

    def get_data_processor_by_name_version(
        self, name: str, version: int
    ) -> DataProcessor | None:
        """Get a data processor by name and version.

        Args:
            name: Data processor name
            version: Data processor version

        Returns:
            DataProcessor object if found, None otherwise

        Raises:
            DatabaseError: If query execution fails
        """
        try:
            escaped_name = self._escape_string(name)
            sql = f"""SELECT * FROM data_processors WHERE name = '{escaped_name}'
                AND version = {version}"""
            result = self._system_query(sql)
            if result.empty():
                return None
            return self._result_to_data_processor(result.first())
        except Exception as e:
            msg = f"Failed to get data processor {name} version {version}: {e}"
            raise DatabaseError(msg) from e

    def get_latest_data_processor_by_name(self, name: str) -> DataProcessor | None:
        """Get the latest version of a data processor by name.

        Args:
            name: Data processor name

        Returns:
            Latest DataProcessor object if found, None otherwise

        Raises:
            DatabaseError: If query execution fails
        """
        try:
            escaped_name = self._escape_string(name)
            sql = f"""SELECT * FROM data_processors WHERE name = '{escaped_name}'
                ORDER BY version DESC LIMIT 1"""
            result = self._system_query(sql)
            if result.empty():
                return None
            return self._result_to_data_processor(result.first())
        except Exception as e:
            raise DatabaseError(
                f"Failed to get latest data processor for {name}: {e}"
            ) from e

    def get_data_processors_by_name(self, name: str) -> list[DataProcessor]:
        """Get all versions of a data processor by name.

        Args:
            name: Data processor name

        Returns:
            List of DataProcessor objects ordered by version DESC

        Raises:
            DatabaseError: If query execution fails
        """
        try:
            escaped_name = self._escape_string(name)
            sql = f"""SELECT * FROM data_processors WHERE name = '{escaped_name}'
                ORDER BY version DESC"""
            result = self._system_query(sql)
            return self._results_to_data_processors(result)
        except Exception as e:
            raise DatabaseError(
                f"Failed to get data processors by name {name}: {e}"
            ) from e

    def create_data_processor(self, data_processor: DataProcessor) -> None:
        """Create a new data processor in the database.

        Args:
            data_processor: DataProcessor object to create

        Raises:
            DatabaseError: If data processor creation fails
        """
        try:
            sql = self._build_data_processor_insert_sql(data_processor)
            self._system_execute(sql)
        except Exception as e:
            raise DatabaseError(
                f"Failed to create data processor {data_processor.id}: {e}"
            ) from e

    def get_next_version_for_name(self, name: str) -> int:
        """Get the next version number for a data processor name.

        Args:
            name: Data processor name

        Returns:
            Next version number (max_version + 1)

        Raises:
            DatabaseError: If version calculation fails
        """
        try:
            processors = self.get_data_processors_by_name(name)
            if not processors:
                return 1

            max_version = max(processor.version for processor in processors)
            return max_version + 1
        except Exception as e:
            raise DatabaseError(f"Failed to get next version for {name}: {e}") from e

    def _result_to_data_processor(self, row: dict[str, Any]) -> DataProcessor:
        """Convert a database row to a DataProcessor object.

        Args:
            row: Database row as dictionary

        Returns:
            DataProcessor object created from row data

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

            return DataProcessor(
                id=str(row["id"]),
                name=str(row["name"]),
                version=int(row["version"]),
                spec=str(row["spec"]),
                description=str(row["description"]),
                created_at=created_at,
                updated_at=updated_at,
            )
        except (KeyError, ValueError, TypeError) as e:
            raise DatabaseError(f"Failed to convert row to DataProcessor: {e}") from e

    def _results_to_data_processors(self, result) -> list[DataProcessor]:
        """Convert query results to list of DataProcessor objects.

        Args:
            result: QueryResult object from database query

        Returns:
            List of DataProcessor objects

        Raises:
            DatabaseError: If conversion fails
        """
        try:
            return [self._result_to_data_processor(row) for row in result.rows]
        except Exception as e:
            raise DatabaseError(
                f"Failed to convert results to data processors: {e}"
            ) from e

    def _build_data_processor_insert_sql(self, processor: DataProcessor) -> str:
        """Build INSERT SQL statement for a data processor.

        Args:
            processor: DataProcessor object to insert

        Returns:
            SQL INSERT statement string

        Raises:
            DatabaseError: If SQL building fails
        """
        try:
            # Format timestamps for SQL
            created_at_str = processor.created_at.isoformat()
            updated_at_str = processor.updated_at.isoformat()

            sql = f"""INSERT INTO data_processors (
                id, name, version, spec, description, created_at, updated_at
            ) VALUES (
                '{self._escape_string(processor.id)}',
                '{self._escape_string(processor.name)}',
                {processor.version},
                '{self._escape_string(processor.spec)}',
                '{self._escape_string(processor.description)}',
                '{created_at_str}',
                '{updated_at_str}'
            )"""

            return sql
        except Exception as e:
            raise DatabaseError(f"Failed to build insert SQL: {e}") from e

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
