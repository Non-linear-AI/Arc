"""Database manager for Arc - handles system and user database separation."""

import threading
from pathlib import Path
from typing import Any

from .base import Database, DatabaseError, QueryResult
from .duckdb import DuckDBDatabase


class DatabaseManager:
    """Manages system and user database separation for Arc.

    - System database: stores Arc metadata (models, jobs, plugins)
    - User database: stores training/prediction data

    Provides centralized database access and abstracts direct DB queries
    from core system components.
    """

    def __init__(
        self,
        system_db_path: str | Path,
        user_db_path: str | Path | None = None,
        shared_connections_for_tests: bool = False,
    ):
        """Initialize database manager with system and optional user database paths.

        Args:
            system_db_path: Path to system database (Arc metadata)
            user_db_path: Optional path to user database (training data)
            shared_connections_for_tests: If True, use shared connections instead of
                thread-local ones. Only use for integration tests that need data
                sharing between threads.
        """
        self.system_db_path = str(system_db_path)
        self.user_db_path = str(user_db_path) if user_db_path else None
        self.shared_connections_for_tests = shared_connections_for_tests

        if shared_connections_for_tests:
            # Test mode: use shared connections (NOT thread-safe)
            self._system_db: Database | None = None
            self._user_db: Database | None = None
        else:
            # Production mode: use thread-local storage for database connections
            # Each thread gets its own Database instances to avoid connection sharing
            self._thread_local = threading.local()

        # Service instances (lazy-loaded)
        self._services_initialized = False

    def _get_system_db(self) -> Database:
        """Get or create system database connection."""
        if self.shared_connections_for_tests:
            # Test mode: use shared connection
            if self._system_db is None:
                self._system_db = DuckDBDatabase(self.system_db_path)
                # Initialize system schema on first access
                self._system_db.init_schema()
            return self._system_db
        else:
            # Production mode: use thread-local connection
            if not hasattr(self._thread_local, "system_db"):
                self._thread_local.system_db = DuckDBDatabase(self.system_db_path)
                # Initialize system schema on first access
                self._thread_local.system_db.init_schema()
            return self._thread_local.system_db

    def _get_user_db(self) -> Database:
        """Get or create user database connection."""
        if self.user_db_path is None:
            raise DatabaseError("No user database configured")

        if self.shared_connections_for_tests:
            # Test mode: use shared connection
            if self._user_db is None:
                self._user_db = DuckDBDatabase(self.user_db_path)
                # User database doesn't need Arc system schema
            return self._user_db
        else:
            # Production mode: use thread-local connection
            if not hasattr(self._thread_local, "user_db"):
                self._thread_local.user_db = DuckDBDatabase(self.user_db_path)
                # User database doesn't need Arc system schema
            return self._thread_local.user_db

    def system_query(self, sql: str, params: list | None = None) -> QueryResult:
        """Execute a query against the system database.

        Args:
            sql: SQL SELECT statement
            params: Optional list of parameters for the query

        Returns:
            QueryResult containing the results

        Raises:
            DatabaseError: If query execution fails
        """
        return self._get_system_db().query(sql, params)

    def system_execute(self, sql: str, params: list | None = None) -> None:
        """Execute a statement against the system database.

        Args:
            sql: SQL statement (DDL/DML)
            params: Optional list of parameters for the statement

        Raises:
            DatabaseError: If statement execution fails
        """
        self._get_system_db().execute(sql, params)

    def user_query(self, sql: str, params: list | None = None) -> QueryResult:
        """Execute a query against the user database.

        Args:
            sql: SQL SELECT statement
            params: Optional list of parameters for the query

        Returns:
            QueryResult containing the results

        Raises:
            DatabaseError: If query execution fails or no user DB configured
        """
        return self._get_user_db().query(sql, params)

    def user_execute(self, sql: str, params: list | None = None) -> None:
        """Execute a statement against the user database.

        Args:
            sql: SQL statement (DDL/DML)
            params: Optional list of parameters for the statement

        Raises:
            DatabaseError: If statement execution fails or no user DB configured
        """
        self._get_user_db().execute(sql, params)

    def set_user_database(self, db_path: str | Path) -> None:
        """Switch to a different user database.

        Args:
            db_path: Path to the new user database
        """
        if self.shared_connections_for_tests:
            # Test mode: close shared connection
            if self._user_db is not None:
                self._user_db.close()
                self._user_db = None
        else:
            # Production mode: close thread-local connection
            if hasattr(self._thread_local, "user_db"):
                self._thread_local.user_db.close()
                delattr(self._thread_local, "user_db")

        self.user_db_path = str(db_path)

    def get_system_db_path(self) -> str:
        """Get the system database path."""
        return self.system_db_path

    def get_user_db_path(self) -> str | None:
        """Get the current user database path."""
        return self.user_db_path

    def has_user_database(self) -> bool:
        """Check if a user database is configured."""
        return self.user_db_path is not None

    def close(self) -> None:
        """Close all database connections."""
        if self.shared_connections_for_tests:
            # Test mode: close shared connections
            if self._system_db is not None:
                self._system_db.close()
                self._system_db = None

            if self._user_db is not None:
                self._user_db.close()
                self._user_db = None
        else:
            # Production mode: close thread-local connections
            if hasattr(self._thread_local, "system_db"):
                self._thread_local.system_db.close()
                delattr(self._thread_local, "system_db")

            if hasattr(self._thread_local, "user_db"):
                self._thread_local.user_db.close()
                delattr(self._thread_local, "user_db")

    def __enter__(self) -> "DatabaseManager":
        """Context manager entry."""
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        """Context manager exit - ensure connections are closed."""
        self.close()

    def __del__(self) -> None:
        """Destructor - ensure connections are closed."""
        self.close()
