"""Base classes for database implementations."""

from abc import ABC, abstractmethod
from collections.abc import Iterator
from dataclasses import dataclass
from typing import Any


@dataclass
class QueryResult:
    """Result of a database query operation."""

    rows: list[dict[str, Any]]
    execution_time: float | None = None

    def empty(self) -> bool:
        """Check if the result set is empty."""
        return len(self.rows) == 0

    def count(self) -> int:
        """Get the number of rows in the result set."""
        return len(self.rows)

    def first(self) -> dict[str, Any] | None:
        """Get the first row, or None if empty."""
        return self.rows[0] if self.rows else None

    def last(self) -> dict[str, Any] | None:
        """Get the last row, or None if empty."""
        return self.rows[-1] if self.rows else None

    def to_list(self) -> list[dict[str, Any]]:
        """Get the raw list of rows."""
        return self.rows

    def __iter__(self) -> Iterator[dict[str, Any]]:
        """Allow iteration over rows."""
        return iter(self.rows)

    def __len__(self) -> int:
        """Get the number of rows."""
        return len(self.rows)

    def __bool__(self) -> bool:
        """Check if the result set has any rows."""
        return not self.empty()


class Database(ABC):
    """Abstract base class for database implementations."""

    @abstractmethod
    def query(self, sql: str) -> QueryResult:
        """Execute a SELECT query and return results.

        Args:
            sql: SQL SELECT statement to execute

        Returns:
            QueryResult containing the query results

        Raises:
            DatabaseError: If query execution fails
        """
        pass

    @abstractmethod
    def execute(self, sql: str) -> None:
        """Execute a DDL or DML statement (CREATE, INSERT, UPDATE, DELETE).

        Args:
            sql: SQL statement to execute

        Raises:
            DatabaseError: If statement execution fails
        """
        pass

    @abstractmethod
    def close(self) -> None:
        """Close the database connection and clean up resources."""
        pass

    @abstractmethod
    def init_schema(self) -> None:
        """Initialize the database schema with required tables and indexes.

        Creates all necessary tables for Arc's data model including:
        - models: Core model definitions with versioning
        - jobs: Long-running training processes
        - trained_models: Immutable training artifacts
        - deployments: Models served for inference
        - plugin_schemas: Plugin metadata for validation
        - plugins: Plugin system metadata
        - plugin_components: Component specifications

        This method is idempotent and can be called multiple times safely.

        Raises:
            DatabaseError: If schema creation fails
        """
        pass


class DatabaseError(Exception):
    """Base exception for database-related errors."""

    pass
