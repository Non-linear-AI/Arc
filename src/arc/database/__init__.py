"""Database module for Arc."""

from arc.database.base import (
    Database,
    DatabaseError,
    QueryResult,
    QueryValidationError,
    TimedQueryResult,
)
from arc.database.duckdb import DuckDBDatabase
from arc.database.manager import DatabaseManager

__all__ = [
    "Database",
    "DatabaseError",
    "QueryResult",
    "QueryValidationError",
    "TimedQueryResult",
    "DuckDBDatabase",
    "DatabaseManager",
]
