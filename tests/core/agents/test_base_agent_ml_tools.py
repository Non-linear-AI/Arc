"""Test BaseAgent ML tools infrastructure."""

from pathlib import Path

import pytest

from arc.core.agents.shared.base_agent import BaseAgent
from arc.database import DatabaseManager
from arc.database.services import ServiceContainer


class TestMLAgentForTesting(BaseAgent):
    """Concrete BaseAgent subclass for testing."""

    def get_template_directory(self):
        """Return a temporary directory for template rendering."""
        return Path("/tmp")


@pytest.fixture
def test_agent():
    """Create a test agent with in-memory databases."""
    db_manager = DatabaseManager(":memory:", ":memory:")
    services = ServiceContainer(db_manager)
    return TestMLAgentForTesting(services, "test_api_key")


class TestMLTools:
    """Test ML tools infrastructure in BaseAgent."""

    def test_get_ml_tools(self, test_agent):
        """Test that _get_ml_tools returns all 3 tools."""
        tools = test_agent._get_ml_tools()
        assert len(tools) == 3

        tool_names = {tool.name for tool in tools}
        assert "database_query" in tool_names
        assert "list_available_knowledge" in tool_names
        assert "read_knowledge_content" in tool_names

    def test_ml_tools_have_descriptions(self, test_agent):
        """Test that all ML tools have descriptions."""
        tools = test_agent._get_ml_tools()
        for tool in tools:
            assert tool.description
            assert len(tool.description) > 0

    def test_ml_tools_have_parameters(self, test_agent):
        """Test that all ML tools have parameters."""
        tools = test_agent._get_ml_tools()
        for tool in tools:
            assert tool.parameters
            assert "type" in tool.parameters
            assert tool.parameters["type"] == "object"


class TestListKnowledgeTool:
    """Test list_available_knowledge tool."""

    def test_handle_list_knowledge(self, test_agent):
        """Test listing knowledge works."""
        result = test_agent._handle_list_knowledge()
        assert result is not None
        assert isinstance(result, str)
        assert "Available" in result
        assert "knowledge" in result.lower()

    def test_list_knowledge_includes_builtin(self, test_agent):
        """Test that builtin knowledge is listed."""
        result = test_agent._handle_list_knowledge()
        # Should include some builtin knowledge like DCN
        assert "dcn" in result.lower() or "deep" in result.lower()


class TestDatabaseQueryTool:
    """Test database_query tool."""

    @pytest.mark.asyncio
    async def test_handle_database_query_invalid_drop(self, test_agent):
        """Test that DROP queries are rejected."""
        result = await test_agent._handle_database_query("DROP TABLE users")
        assert "Error" in result
        assert "read-only" in result.lower()

    @pytest.mark.asyncio
    async def test_handle_database_query_invalid_delete(self, test_agent):
        """Test that DELETE queries are rejected."""
        result = await test_agent._handle_database_query("DELETE FROM users")
        assert "Error" in result
        assert "read-only" in result.lower()

    @pytest.mark.asyncio
    async def test_handle_database_query_invalid_insert(self, test_agent):
        """Test that INSERT queries are rejected."""
        result = await test_agent._handle_database_query(
            "INSERT INTO users VALUES (1, 'test')"
        )
        assert "Error" in result
        assert "read-only" in result.lower()

    @pytest.mark.asyncio
    async def test_handle_database_query_invalid_update(self, test_agent):
        """Test that UPDATE queries are rejected."""
        result = await test_agent._handle_database_query("UPDATE users SET name='test'")
        assert "Error" in result
        assert "read-only" in result.lower()

    @pytest.mark.asyncio
    async def test_handle_database_query_valid_select(self, test_agent):
        """Test that SELECT queries are allowed."""
        # Create a simple table first
        test_agent.services.query.execute_query(
            "CREATE TABLE test_table AS SELECT 1 as col1", "user"
        )

        result = await test_agent._handle_database_query("SELECT * FROM test_table")
        # Should succeed
        assert "Error" not in result or "read-only" not in result.lower()

    @pytest.mark.asyncio
    async def test_handle_database_query_simple_select(self, test_agent):
        """Test simple SELECT without table."""
        result = await test_agent._handle_database_query("SELECT 1 AS num")
        # Should execute successfully
        assert "Error" not in result or "read-only" not in result.lower()


class TestReadKnowledgeTool:
    """Test read_knowledge_content tool."""

    def test_handle_read_knowledge_missing_id(self, test_agent):
        """Test reading knowledge with missing ID."""
        result = test_agent._handle_read_knowledge("", "model")
        assert "Error" in result

    def test_handle_read_knowledge_invalid_id(self, test_agent):
        """Test reading knowledge with invalid ID."""
        result = test_agent._handle_read_knowledge("nonexistent_architecture", "model")
        assert "Error" in result or "not found" in result.lower()

    def test_handle_read_knowledge_valid_builtin(self, test_agent):
        """Test reading valid builtin knowledge."""
        # DCN is a builtin knowledge
        result = test_agent._handle_read_knowledge("dcn", "model")
        assert result is not None
        assert isinstance(result, str)
        assert len(result) > 0
        # Should not be an error
        assert "Error" not in result or "Deep & Cross" in result

    def test_handle_read_knowledge_different_phases(self, test_agent):
        """Test reading knowledge with different phases."""
        # Try reading with different valid phases
        for phase in ["model", "train", "evaluate", "data"]:
            result = test_agent._handle_read_knowledge("dcn", phase)
            # Should return something, may be knowledge or error if phase not available
            assert result is not None
            assert isinstance(result, str)


class TestExecuteMLTool:
    """Test _execute_ml_tool dispatcher."""

    @pytest.mark.asyncio
    async def test_execute_ml_tool_list_knowledge(self, test_agent):
        """Test executing list_knowledge tool via dispatcher."""

        result = await test_agent._execute_ml_tool("list_available_knowledge", "{}")
        assert result is not None
        assert isinstance(result, str)
        assert "Available" in result

    @pytest.mark.asyncio
    async def test_execute_ml_tool_database_query(self, test_agent):
        """Test executing database_query tool via dispatcher."""
        import json

        result = await test_agent._execute_ml_tool(
            "database_query", json.dumps({"query": "SELECT 1"})
        )
        assert result is not None
        assert isinstance(result, str)

    @pytest.mark.asyncio
    async def test_execute_ml_tool_read_knowledge(self, test_agent):
        """Test executing read_knowledge tool via dispatcher."""
        import json

        result = await test_agent._execute_ml_tool(
            "read_knowledge_content",
            json.dumps({"knowledge_id": "dcn", "phase": "model"}),
        )
        assert result is not None
        assert isinstance(result, str)

    @pytest.mark.asyncio
    async def test_execute_ml_tool_unknown(self, test_agent):
        """Test executing unknown tool raises error."""
        result = await test_agent._execute_ml_tool("unknown_tool", "{}")
        assert "Unknown" in result or "not found" in result.lower()
