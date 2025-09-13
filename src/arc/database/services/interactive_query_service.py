"""Interactive query service for Arc query processing."""

import time
from typing import Any

from ..base import DatabaseError, QueryResult, QueryValidationError, TimedQueryResult
from .base import BaseService


class InteractiveQueryService(BaseService):
    """Service for interactive query processing and execution.

    Handles complex query operations including:
    - Interactive SQL query execution
    - Query validation and safety checks
    - Database targeting (system vs user)
    - Query result formatting
    """

    def __init__(self, db_manager):
        """Initialize InteractiveQueryService.

        Args:
            db_manager: DatabaseManager instance
        """
        super().__init__(db_manager)

    def execute_query(self, query: str, target_db: str = "system") -> QueryResult:
        """Execute SQL query against the specified database.

        Args:
            query: SQL query to execute
            target_db: Target database ("system" or "user")

        Returns:
            QueryResult with execution time tracking

        Raises:
            QueryValidationError: If query is invalid or target database is invalid
            DatabaseError: If database operation fails
        """
        start_time = time.time()

        # Validate target database
        if target_db not in ["system", "user"]:
            raise QueryValidationError(
                f"Invalid target database: {target_db}. Must be 'system' or 'user'."
            )

        # Validate query for safety
        self._validate_query(query, target_db)

        # Check if user database is configured when needed
        if target_db == "user" and not self.db_manager.has_user_database():
            raise QueryValidationError(
                "User database is not configured. Set user database path in settings."
            )

        try:
            # Execute query based on target database
            if target_db == "system":
                result = self._system_query(query)
            else:  # user
                result = self._user_query(query)

            execution_time = time.time() - start_time

            # Return a TimedQueryResult with execution time
            return TimedQueryResult(result.rows, execution_time)

        except Exception as e:
            execution_time = time.time() - start_time
            if isinstance(e, DatabaseError):
                raise
            else:
                raise DatabaseError(f"Unexpected error: {str(e)}") from e

    def _validate_query(self, query: str, target_db: str) -> None:
        """Validate query for safety and correctness based on target database.

        Args:
            query: SQL query to validate
            target_db: Target database ("system" or "user")

        Raises:
            QueryValidationError: If query is invalid
        """
        if not query or not query.strip():
            raise QueryValidationError("Empty SQL query provided.")

        query_upper = query.upper().strip()

        if target_db == "system":
            # System database: read-only access (SELECT only)
            if not query_upper.startswith("SELECT"):
                raise QueryValidationError(
                    "System database is read-only. Only SELECT queries are allowed. "
                    "Supported: SELECT statements. "
                    "Not supported: INSERT, UPDATE, DELETE, CREATE, DROP, etc."
                )
        elif target_db == "user":
            # User database: full SQL access allowed
            # We could add specific validations here if needed in the future
            # For now, allow all SQL operations on user database
            pass

    def get_available_tables(self, target_db: str = "system") -> list[dict[str, str]]:
        """Get list of available tables in the specified database.

        Args:
            target_db: Target database ("system" or "user")

        Returns:
            List of tables with name and type

        Raises:
            QueryValidationError: If target database is invalid or not configured
            DatabaseError: If database operation fails
        """
        if target_db not in ["system", "user"]:
            raise QueryValidationError(f"Invalid target database: {target_db}")

        if target_db == "user" and not self.db_manager.has_user_database():
            raise QueryValidationError("User database is not configured.")

        if target_db == "system":
            result = self._system_query("""
                SELECT name, type FROM sqlite_master
                WHERE type='table' AND name NOT LIKE 'sqlite_%'
                ORDER BY name
            """)
        else:  # user
            result = self._user_query("""
                SELECT name, type FROM sqlite_master
                WHERE type='table' AND name NOT LIKE 'sqlite_%'
                ORDER BY name
            """)

        return [{"name": row["name"], "type": row["type"]} for row in result]

    def get_table_schema(
        self, table_name: str, target_db: str = "system"
    ) -> list[dict[str, Any]]:
        """Get schema information for a specific table.

        Args:
            table_name: Name of the table
            target_db: Target database ("system" or "user")

        Returns:
            List of column information dictionaries

        Raises:
            QueryValidationError: If target database is invalid, not configured,
                or table not found
            DatabaseError: If database operation fails
        """
        if target_db not in ["system", "user"]:
            raise QueryValidationError(f"Invalid target database: {target_db}")

        if target_db == "user" and not self.db_manager.has_user_database():
            raise QueryValidationError("User database is not configured.")

        if target_db == "system":
            result = self._system_query(f"PRAGMA table_info({table_name})")
        else:  # user
            result = self._user_query(f"PRAGMA table_info({table_name})")

        if result.empty():
            raise QueryValidationError(
                f"Table '{table_name}' not found in {target_db} database."
            )

        columns = []
        for row in result:
            columns.append(
                {
                    "name": row["name"],
                    "type": row["type"],
                    "nullable": not row["notnull"],
                    "primary_key": bool(row["pk"]),
                }
            )

        return columns
