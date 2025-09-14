"""Database manager for Arc - handles system and user database separation."""

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
        self, system_db_path: str | Path, user_db_path: str | Path | None = None
    ):
        """Initialize database manager with system and optional user database paths.

        Args:
            system_db_path: Path to system database (Arc metadata)
            user_db_path: Optional path to user database (training data)
        """
        self.system_db_path = str(system_db_path)
        self.user_db_path = str(user_db_path) if user_db_path else None

        # Lazy-loaded database connections
        self._system_db: Database | None = None
        self._user_db: Database | None = None

        # Service instances (lazy-loaded)
        self._services_initialized = False

    def _get_system_db(self) -> Database:
        """Get or create system database connection."""
        if self._system_db is None:
            self._system_db = DuckDBDatabase(self.system_db_path)
            # Initialize system schema on first access
            self._system_db.init_schema()
        return self._system_db

    def _get_user_db(self) -> Database:
        """Get or create user database connection."""
        if self.user_db_path is None:
            raise DatabaseError("No user database configured")

        if self._user_db is None:
            self._user_db = DuckDBDatabase(self.user_db_path)
            # User database doesn't need Arc system schema
        return self._user_db

    def system_query(self, sql: str) -> QueryResult:
        """Execute a query against the system database.

        Args:
            sql: SQL SELECT statement

        Returns:
            QueryResult containing the results

        Raises:
            DatabaseError: If query execution fails
        """
        return self._get_system_db().query(sql)

    def system_execute(self, sql: str) -> None:
        """Execute a statement against the system database.

        Args:
            sql: SQL statement (DDL/DML)

        Raises:
            DatabaseError: If statement execution fails
        """
        self._get_system_db().execute(sql)

    def user_query(self, sql: str) -> QueryResult:
        """Execute a query against the user database.

        Args:
            sql: SQL SELECT statement

        Returns:
            QueryResult containing the results

        Raises:
            DatabaseError: If query execution fails or no user DB configured
        """
        return self._get_user_db().query(sql)

    def user_execute(self, sql: str) -> None:
        """Execute a statement against the user database.

        Args:
            sql: SQL statement (DDL/DML)

        Raises:
            DatabaseError: If statement execution fails or no user DB configured
        """
        self._get_user_db().execute(sql)

    def set_user_database(self, db_path: str | Path) -> None:
        """Switch to a different user database.

        Args:
            db_path: Path to the new user database
        """
        # Close existing user database connection
        if self._user_db is not None:
            self._user_db.close()
            self._user_db = None

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
        if self._system_db is not None:
            self._system_db.close()
            self._system_db = None

        if self._user_db is not None:
            self._user_db.close()
            self._user_db = None

    def __enter__(self) -> "DatabaseManager":
        """Context manager entry."""
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        """Context manager exit - ensure connections are closed."""
        self.close()

    def __del__(self) -> None:
        """Destructor - ensure connections are closed."""
        self.close()
