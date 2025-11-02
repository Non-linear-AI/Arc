"""Tests for InteractiveQueryService READ-ONLY enforcement."""

from __future__ import annotations

import pytest

from arc.database import DatabaseManager
from arc.database.base import QueryValidationError
from arc.database.services.interactive_query_service import InteractiveQueryService


@pytest.fixture
def db_manager(tmp_path):
    """Create a database manager with temporary databases."""
    system_db = tmp_path / "system.db"
    user_db = tmp_path / "user.db"
    manager = DatabaseManager(str(system_db), str(user_db))

    # Initialize connections
    manager.system_query("SELECT 1")
    manager.user_execute("SELECT 1")

    yield manager

    manager.close()


@pytest.fixture
def query_service(db_manager):
    """Create a query service instance."""
    return InteractiveQueryService(db_manager)


def test_system_db_blocks_create_table(query_service):
    """Test that system database blocks CREATE TABLE operations."""
    with pytest.raises(
        QueryValidationError, match="System database is read-only.*CREATE"
    ):
        query_service.execute_query(
            "CREATE TABLE test (id INTEGER)", target_db="system"
        )


def test_system_db_blocks_insert(query_service):
    """Test that system database blocks INSERT operations."""
    with pytest.raises(QueryValidationError, match="System database is read-only.*"):
        query_service.execute_query("INSERT INTO test VALUES (1)", target_db="system")


def test_system_db_blocks_update(query_service):
    """Test that system database blocks UPDATE operations."""
    with pytest.raises(QueryValidationError, match="System database is read-only.*"):
        query_service.execute_query("UPDATE test SET value = 1", target_db="system")


def test_system_db_blocks_delete(query_service):
    """Test that system database blocks DELETE operations."""
    with pytest.raises(QueryValidationError, match="System database is read-only.*"):
        query_service.execute_query("DELETE FROM test", target_db="system")


def test_system_db_blocks_drop(query_service):
    """Test that system database blocks DROP operations."""
    with pytest.raises(QueryValidationError, match="System database is read-only.*"):
        query_service.execute_query("DROP TABLE test", target_db="system")


def test_system_db_blocks_alter(query_service):
    """Test that system database blocks ALTER operations."""
    with pytest.raises(QueryValidationError, match="System database is read-only.*"):
        query_service.execute_query(
            "ALTER TABLE test ADD COLUMN value INTEGER", target_db="system"
        )


def test_system_db_blocks_truncate(query_service):
    """Test that system database blocks TRUNCATE operations."""
    with pytest.raises(QueryValidationError, match="System database is read-only.*"):
        query_service.execute_query("TRUNCATE TABLE test", target_db="system")


def test_system_db_allows_select(query_service):
    """Test that system database allows SELECT operations."""
    # Should not raise an exception
    result = query_service.execute_query("SELECT 1 as test", target_db="system")
    assert not result.empty()


def test_read_only_mode_blocks_create_on_user_db(query_service):
    """Test that read_only mode blocks CREATE TABLE on user database."""
    with pytest.raises(QueryValidationError, match="Read-only mode enforced.*ml_data"):
        query_service.execute_query(
            "CREATE TABLE test (id INTEGER)", target_db="user", read_only=True
        )


def test_read_only_mode_blocks_insert_on_user_db(query_service):
    """Test that read_only mode blocks INSERT on user database."""
    with pytest.raises(QueryValidationError, match="Read-only mode enforced.*ml_data"):
        query_service.execute_query(
            "INSERT INTO test VALUES (1)", target_db="user", read_only=True
        )


def test_read_only_mode_blocks_update_on_user_db(query_service):
    """Test that read_only mode blocks UPDATE on user database."""
    with pytest.raises(QueryValidationError, match="Read-only mode enforced.*ml_data"):
        query_service.execute_query(
            "UPDATE test SET value = 1", target_db="user", read_only=True
        )


def test_read_only_mode_blocks_delete_on_user_db(query_service):
    """Test that read_only mode blocks DELETE on user database."""
    with pytest.raises(QueryValidationError, match="Read-only mode enforced.*ml_data"):
        query_service.execute_query(
            "DELETE FROM test", target_db="user", read_only=True
        )


def test_read_only_mode_blocks_drop_on_user_db(query_service):
    """Test that read_only mode blocks DROP on user database."""
    with pytest.raises(QueryValidationError, match="Read-only mode enforced.*ml_data"):
        query_service.execute_query("DROP TABLE test", target_db="user", read_only=True)


def test_read_only_mode_allows_select_on_user_db(query_service):
    """Test that read_only mode allows SELECT on user database."""
    # Should not raise an exception
    result = query_service.execute_query(
        "SELECT 1 as test", target_db="user", read_only=True
    )
    assert not result.empty()


def test_user_db_allows_write_without_read_only(query_service):
    """Test that user database allows write operations when read_only=False."""
    # Create a test table
    query_service.execute_query(
        "CREATE TABLE test_write (id INTEGER, value TEXT)", target_db="user"
    )

    # Insert data
    query_service.execute_query(
        "INSERT INTO test_write VALUES (1, 'test')", target_db="user"
    )

    # Verify data was inserted
    result = query_service.execute_query("SELECT * FROM test_write", target_db="user")
    assert not result.empty()
    assert result.count() == 1

    # Cleanup
    query_service.execute_query("DROP TABLE test_write", target_db="user")


def test_case_insensitive_command_detection(query_service):
    """Test that command detection is case-insensitive."""
    # Should block lowercase create
    with pytest.raises(QueryValidationError):
        query_service.execute_query(
            "create table test (id INTEGER)", target_db="system"
        )

    # Should block mixed case CREATE
    with pytest.raises(QueryValidationError):
        query_service.execute_query(
            "CrEaTe TaBlE test (id INTEGER)", target_db="system"
        )


def test_empty_query_validation(query_service):
    """Test that empty queries are rejected."""
    with pytest.raises(QueryValidationError, match="Empty SQL query"):
        query_service.execute_query("", target_db="user")

    with pytest.raises(QueryValidationError, match="Empty SQL query"):
        query_service.execute_query("   ", target_db="user")
