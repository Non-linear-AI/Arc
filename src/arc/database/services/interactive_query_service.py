"""Interactive query service for Arc query processing."""

import time

from arc.database.base import (
    DatabaseError,
    QueryResult,
    QueryValidationError,
    TimedQueryResult,
)
from arc.database.services.base import BaseService


class InteractiveQueryService(BaseService):
    """Service for interactive query processing and execution.

    Handles complex query operations including:
    - Interactive SQL query execution
    - Query validation and safety checks
    - Database targeting (system vs user)
    - Query result formatting
    - Schema cache invalidation for DDL operations
    """

    def __init__(self, db_manager, schema_service=None):
        """Initialize InteractiveQueryService.

        Args:
            db_manager: DatabaseManager instance
            schema_service: SchemaService instance for cache invalidation
        """
        super().__init__(db_manager)
        self._schema_service = schema_service

    def execute_query(
        self, query: str, target_db: str = "system", read_only: bool = False
    ) -> QueryResult:
        """Execute SQL query against the specified database.

        Args:
            query: SQL query to execute
            target_db: Target database ("system" or "user")
            read_only: If True, only SELECT queries are allowed (default: False)

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
        self._validate_query(query, target_db, read_only=read_only)

        # Check if user database is configured when needed
        if target_db == "user" and not self.db_manager.has_user_database():
            raise QueryValidationError(
                "User database is not configured. Set user database path in settings."
            )

        try:
            # Execute query based on target database
            result = (
                self._system_query(query)
                if target_db == "system"
                else self._user_query(query)
            )

            execution_time = time.time() - start_time

            # Check if this was a DDL operation and invalidate schema cache if needed
            if self._schema_service and self._schema_service.is_ddl_statement(query):
                self._schema_service.invalidate_cache(target_db)

            # Return a TimedQueryResult with execution time
            return TimedQueryResult(result.rows, execution_time)

        except Exception as e:
            execution_time = time.time() - start_time
            if isinstance(e, DatabaseError):
                raise
            else:
                raise DatabaseError(f"Unexpected error: {str(e)}") from e

    def _validate_query(
        self, query: str, target_db: str, read_only: bool = False
    ) -> None:
        """Validate query for safety and correctness based on target database.

        Args:
            query: SQL query to validate
            target_db: Target database ("system" or "user")
            read_only: If True, only SELECT queries are allowed

        Raises:
            QueryValidationError: If query is invalid
        """
        if not query or not query.strip():
            raise QueryValidationError("Empty SQL query provided.")

        query_upper = query.upper().strip()

        # System database is always read-only
        if target_db == "system" or read_only:
            # Block modification SQL commands
            modification_commands = {
                "INSERT",
                "UPDATE",
                "DELETE",
                "CREATE",
                "DROP",
                "ALTER",
                "TRUNCATE",
            }
            first_word = query_upper.split()[0] if query_upper.split() else ""
            if first_word in modification_commands:
                if target_db == "system":
                    msg = (
                        "System database is read-only. "
                        "Supported: SELECT, SHOW, DESCRIBE. "
                        "Not supported: INSERT, UPDATE, DELETE, CREATE, DROP, etc."
                    )
                else:
                    msg = (
                        "Read-only mode enforced. "
                        "Supported: SELECT, SHOW, DESCRIBE. "
                        "For data loading or transformations, use ml_data tool instead."
                    )
                raise QueryValidationError(msg)
        elif target_db == "user":
            # User database: full SQL access allowed when not in read-only mode
            pass
