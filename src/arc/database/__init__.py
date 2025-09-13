"""Database module for Arc."""

from .base import Database, DatabaseError, QueryResult
from .duckdb import DuckDBDatabase
from .manager import DatabaseManager

__all__ = [
    "Database",
    "DatabaseError",
    "QueryResult",
    "DuckDBDatabase",
    "DatabaseManager",
]
