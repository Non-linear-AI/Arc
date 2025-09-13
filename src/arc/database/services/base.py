"""Base service class for database operations."""

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ..manager import DatabaseManager


class BaseService:
    """Base class for all database services.

    Provides common functionality and database access patterns
    for domain-specific service classes.
    """

    def __init__(self, db_manager: "DatabaseManager"):
        """Initialize service with database manager.

        Args:
            db_manager: DatabaseManager instance for database access
        """
        self.db_manager = db_manager

    def _system_query(self, sql: str):
        """Execute query against system database."""
        return self.db_manager.system_query(sql)

    def _system_execute(self, sql: str) -> None:
        """Execute statement against system database."""
        self.db_manager.system_execute(sql)

    def _user_query(self, sql: str):
        """Execute query against user database."""
        return self.db_manager.user_query(sql)

    def _user_execute(self, sql: str) -> None:
        """Execute statement against user database."""
        self.db_manager.user_execute(sql)
