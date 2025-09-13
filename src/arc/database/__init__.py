"""Database module for Arc."""

from .base import (
    Database,
    DatabaseError,
    QueryResult,
    QueryValidationError,
    TimedQueryResult,
)
from .duckdb import DuckDBDatabase
from .manager import DatabaseManager

__all__ = [
    "Database",
    "DatabaseError",
    "QueryResult",
    "QueryValidationError",
    "TimedQueryResult",
    "DuckDBDatabase",
    "DatabaseManager",
]
