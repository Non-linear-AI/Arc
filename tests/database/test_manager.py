"""Tests for DatabaseManager."""

from pathlib import Path

import pytest

from arc.database import DatabaseError, DatabaseManager


def test_database_manager_initialization():
    """Test DatabaseManager initialization."""
    system_db = ":memory:"
    user_db = ":memory:"

    with DatabaseManager(system_db, user_db) as manager:
        assert manager.get_system_db_path() == system_db
        assert manager.get_user_db_path() == user_db
        assert manager.has_user_database()


def test_database_manager_no_user_db():
    """Test DatabaseManager with no user database."""
    system_db = ":memory:"

    with DatabaseManager(system_db) as manager:
        assert manager.get_system_db_path() == system_db
        assert manager.get_user_db_path() is None
        assert not manager.has_user_database()


def test_system_database_operations():
    """Test system database operations."""
    with DatabaseManager(":memory:") as manager:
        # System database should have Arc schema initialized
        result = manager.system_query("SELECT COUNT(*) as count FROM models")
        assert result.first()["count"] == 0

        # Insert a test model
        manager.system_execute("""
            INSERT INTO models (id, name, version, type)
            VALUES ('test-1', 'test_model', 1, 'classification')
        """)

        result = manager.system_query("SELECT * FROM models WHERE id = 'test-1'")
        assert result.count() == 1
        assert result.first()["name"] == "test_model"


def test_user_database_operations():
    """Test user database operations."""
    with DatabaseManager(":memory:", ":memory:") as manager:
        # Create a test table in user database
        manager.user_execute("CREATE TABLE test_data (id INTEGER, value TEXT)")

        # Insert test data
        manager.user_execute("INSERT INTO test_data VALUES (1, 'test')")

        result = manager.user_query("SELECT * FROM test_data")
        assert result.count() == 1
        assert result.first()["value"] == "test"


def test_user_database_not_configured():
    """Test operations when user database is not configured."""
    with DatabaseManager(":memory:") as manager:
        with pytest.raises(DatabaseError, match="No user database configured"):
            manager.user_query("SELECT 1")

        with pytest.raises(DatabaseError, match="No user database configured"):
            manager.user_execute("CREATE TABLE test (id INTEGER)")


def test_set_user_database():
    """Test switching user databases."""
    with DatabaseManager(":memory:") as manager:
        assert not manager.has_user_database()

        # Set user database
        manager.set_user_database(":memory:")
        assert manager.has_user_database()
        assert manager.get_user_db_path() == ":memory:"

        # Should now work
        manager.user_execute("CREATE TABLE test (id INTEGER)")
        result = manager.user_query("SELECT COUNT(*) as count FROM test")
        assert result.first()["count"] == 0


def test_pathlib_paths():
    """Test that Path objects are handled correctly."""
    system_path = Path(":memory:")
    user_path = Path(":memory:")

    with DatabaseManager(system_path, user_path) as manager:
        assert manager.get_system_db_path() == str(system_path)
        assert manager.get_user_db_path() == str(user_path)
