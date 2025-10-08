"""Simple tests for DuckDB database implementation."""

import pytest

from arc.database import DatabaseError, DuckDBDatabase


def test_duckdb_basic_operations():
    """Test basic DuckDB operations with in-memory database."""
    with DuckDBDatabase(":memory:") as db:
        # Test execute - create table
        db.execute("CREATE TABLE test (id INTEGER, name VARCHAR)")

        # Test execute - insert data
        db.execute("INSERT INTO test VALUES (1, 'Alice'), (2, 'Bob')")

        # Test query - select data
        result = db.query("SELECT * FROM test ORDER BY id")

        assert result.count() == 2
        assert not result.empty()
        assert result.first()["id"] == 1
        assert result.first()["name"] == "Alice"
        assert len(result) == 2


def test_query_result_methods():
    """Test QueryResult convenience methods."""
    with DuckDBDatabase(":memory:") as db:
        db.execute("CREATE TABLE test (id INTEGER)")

        # Test empty result
        empty_result = db.query("SELECT * FROM test")
        assert empty_result.empty()
        assert empty_result.count() == 0
        assert empty_result.first() is None
        assert not empty_result

        # Test with data
        db.execute("INSERT INTO test VALUES (1), (2), (3)")
        result = db.query("SELECT * FROM test ORDER BY id")

        assert not result.empty()
        assert result.count() == 3
        assert result.first()["id"] == 1
        assert result.last()["id"] == 3
        assert bool(result)


def test_database_error_handling():
    """Test error handling for invalid SQL."""
    with DuckDBDatabase(":memory:") as db:
        with pytest.raises(DatabaseError):
            db.query("INVALID SQL")

        with pytest.raises(DatabaseError):
            db.execute("INVALID SQL")


def test_init_schema():
    """Test schema initialization creates all required tables."""
    with DuckDBDatabase(":memory:") as db:
        # Initialize schema
        db.init_schema()

        # Verify core tables exist by querying them
        expected_tables = [
            "models",
            "jobs",
            "plugin_schemas",
            "plugins",
            "plugin_components",
            "trainers",
            "plans",
            "training_runs",
            "training_metrics",
            "training_checkpoints",
        ]

        for table in expected_tables:
            # This should not raise an error if table exists
            result = db.query(f"SELECT COUNT(*) as count FROM {table}")
            assert result.first()["count"] == 0

        # Test idempotent behavior - should not fail when called again
        db.init_schema()

        # Verify we can insert and query data
        db.execute("""
            INSERT INTO models (id, name, version, type)
            VALUES ('test-1', 'test_model', 1, 'classification')
        """)

        result = db.query("SELECT * FROM models WHERE id = 'test-1'")
        assert result.count() == 1
        assert result.first()["name"] == "test_model"
