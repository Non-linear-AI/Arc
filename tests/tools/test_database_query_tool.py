"""Tests for DatabaseQueryTool READ-ONLY enforcement."""

from __future__ import annotations

import pytest

from arc.database import DatabaseManager
from arc.database.services.container import ServiceContainer
from arc.tools.base import ToolResult
from arc.tools.database_query import DatabaseQueryTool


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
def services(tmp_path, db_manager):
    """Create service container for tests."""
    artifacts_dir = tmp_path / "artifacts"
    artifacts_dir.mkdir(exist_ok=True)
    return ServiceContainer(db_manager, artifacts_dir=str(artifacts_dir))


@pytest.fixture
def database_query_tool(services):
    """Create a DatabaseQueryTool instance."""
    return DatabaseQueryTool(services)


@pytest.mark.asyncio
async def test_tool_blocks_create_table_on_user_db(database_query_tool):
    """Test that database_query tool blocks CREATE TABLE on user database."""
    result = await database_query_tool.execute(
        query="CREATE TABLE test (id INTEGER)", target_db="user"
    )

    assert isinstance(result, ToolResult)
    assert not result.success
    assert result.error is not None
    assert "Read-only mode enforced" in result.error
    assert "ml_data" in result.error


@pytest.mark.asyncio
async def test_tool_blocks_insert_on_user_db(database_query_tool):
    """Test that database_query tool blocks INSERT on user database."""
    result = await database_query_tool.execute(
        query="INSERT INTO test VALUES (1)", target_db="user"
    )

    assert isinstance(result, ToolResult)
    assert not result.success
    assert result.error is not None
    assert "Read-only mode enforced" in result.error


@pytest.mark.asyncio
async def test_tool_blocks_update_on_user_db(database_query_tool):
    """Test that database_query tool blocks UPDATE on user database."""
    result = await database_query_tool.execute(
        query="UPDATE test SET value = 1", target_db="user"
    )

    assert isinstance(result, ToolResult)
    assert not result.success
    assert result.error is not None
    assert "Read-only mode enforced" in result.error


@pytest.mark.asyncio
async def test_tool_blocks_delete_on_user_db(database_query_tool):
    """Test that database_query tool blocks DELETE on user database."""
    result = await database_query_tool.execute(
        query="DELETE FROM test", target_db="user"
    )

    assert isinstance(result, ToolResult)
    assert not result.success
    assert result.error is not None
    assert "Read-only mode enforced" in result.error


@pytest.mark.asyncio
async def test_tool_blocks_drop_on_user_db(database_query_tool):
    """Test that database_query tool blocks DROP on user database."""
    result = await database_query_tool.execute(
        query="DROP TABLE test", target_db="user"
    )

    assert isinstance(result, ToolResult)
    assert not result.success
    assert result.error is not None
    assert "Read-only mode enforced" in result.error


@pytest.mark.asyncio
async def test_tool_blocks_alter_on_user_db(database_query_tool):
    """Test that database_query tool blocks ALTER on user database."""
    result = await database_query_tool.execute(
        query="ALTER TABLE test ADD COLUMN value INTEGER", target_db="user"
    )

    assert isinstance(result, ToolResult)
    assert not result.success
    assert result.error is not None
    assert "Read-only mode enforced" in result.error


@pytest.mark.asyncio
async def test_tool_blocks_truncate_on_user_db(database_query_tool):
    """Test that database_query tool blocks TRUNCATE on user database."""
    result = await database_query_tool.execute(
        query="TRUNCATE TABLE test", target_db="user"
    )

    assert isinstance(result, ToolResult)
    assert not result.success
    assert result.error is not None
    assert "Read-only mode enforced" in result.error


@pytest.mark.asyncio
async def test_tool_allows_select_on_user_db(database_query_tool):
    """Test that database_query tool allows SELECT on user database."""
    result = await database_query_tool.execute(
        query="SELECT 1 as test", target_db="user"
    )

    assert isinstance(result, ToolResult)
    assert result.success
    # The result should contain the query output as JSON
    assert "test" in result.output or "1" in result.output


@pytest.mark.asyncio
async def test_tool_blocks_create_on_system_db(database_query_tool):
    """Test that database_query tool blocks CREATE on system database."""
    result = await database_query_tool.execute(
        query="CREATE TABLE test (id INTEGER)", target_db="system"
    )

    assert isinstance(result, ToolResult)
    assert not result.success
    assert result.error is not None
    assert "System database is read-only" in result.error


@pytest.mark.asyncio
async def test_tool_allows_select_on_system_db(database_query_tool):
    """Test that database_query tool allows SELECT on system database."""
    result = await database_query_tool.execute(
        query="SELECT 1 as test", target_db="system"
    )

    assert isinstance(result, ToolResult)
    assert result.success


@pytest.mark.asyncio
async def test_tool_rejects_invalid_target_db(database_query_tool):
    """Test that database_query tool rejects invalid target database."""
    result = await database_query_tool.execute(query="SELECT 1", target_db="invalid")

    assert isinstance(result, ToolResult)
    assert not result.success
    assert result.error is not None
    assert "Invalid target database" in result.error


@pytest.mark.asyncio
async def test_tool_suggests_ml_data_for_mutations(database_query_tool):
    """Test that error messages suggest using ml_data for mutations."""
    result = await database_query_tool.execute(
        query="CREATE TABLE test AS SELECT * FROM read_csv('data.csv')",
        target_db="user",
    )

    assert isinstance(result, ToolResult)
    assert not result.success
    assert result.error is not None
    assert "ml_data" in result.error.lower()
