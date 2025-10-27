"""Tests for ML Plan agent with tool support."""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from arc.core.agents.ml_plan import MLPlanAgent


class TestMLPlanAgent:
    """Test ML Plan agent functionality."""

    @pytest.fixture
    def mock_services(self):
        """Mock services container."""
        services_mock = MagicMock()

        # Mock ML data service
        mock_dataset_info = MagicMock()
        mock_dataset_info.name = "test_table"
        mock_dataset_info.columns = [
            {"name": "feature1", "type": "REAL"},
            {"name": "feature2", "type": "REAL"},
            {"name": "target", "type": "INTEGER"},
        ]
        services_mock.ml_data.get_dataset_info.return_value = mock_dataset_info

        return services_mock

    @pytest.fixture
    def ml_plan_agent(self, mock_services):
        """ML Plan agent instance."""
        return MLPlanAgent(mock_services, "test_api_key")

    def test_get_tools(self, ml_plan_agent):
        """Test that agent provides all three tools."""
        tools = ml_plan_agent._get_ml_tools()

        assert len(tools) == 3
        tool_names = [tool.name for tool in tools]
        assert "list_available_knowledge" in tool_names
        assert "read_knowledge_content" in tool_names
        assert "database_query" in tool_names

    def test_handle_list_knowledge(self, ml_plan_agent):
        """Test listing available knowledge."""
        import json

        result = ml_plan_agent._handle_list_knowledge()

        assert isinstance(result, str)
        # Should be valid JSON
        knowledge_list = json.loads(result)
        assert isinstance(knowledge_list, list)

    def test_handle_read_knowledge_valid(self, ml_plan_agent):
        """Test reading valid knowledge document."""
        # Mock knowledge loader - now returns tuple (content, actual_phase)
        ml_plan_agent.knowledge_loader.load_knowledge = MagicMock(
            return_value=("# MLP Guide\n\nBest practices for MLPs...", "model")
        )
        ml_plan_agent.knowledge_loader.scan_metadata = MagicMock(
            return_value={"mlp": MagicMock(name="Multi-Layer Perceptron Guide")}
        )

        result = ml_plan_agent._handle_read_knowledge("mlp", "model")

        assert "MLP Guide" in result
        assert "Best practices" in result

    def test_handle_read_knowledge_invalid(self, ml_plan_agent):
        """Test reading non-existent knowledge."""
        # Mock returns tuple (None, None) for not found
        ml_plan_agent.knowledge_loader.load_knowledge = MagicMock(
            return_value=(None, None)
        )

        result = ml_plan_agent._handle_read_knowledge("invalid", "model")

        assert "Error" in result
        assert "not found" in result

    @pytest.mark.asyncio
    async def test_handle_database_query_validation(self, ml_plan_agent):
        """Test database query validation (read-only)."""
        # Test INSERT (should fail)
        result = await ml_plan_agent._handle_database_query(
            "INSERT INTO table VALUES (1, 2, 3)"
        )
        assert "Error" in result
        assert "read-only" in result

        # Test UPDATE (should fail)
        result = await ml_plan_agent._handle_database_query("UPDATE table SET col = 1")
        assert "Error" in result
        assert "read-only" in result

        # Test DELETE (should fail)
        result = await ml_plan_agent._handle_database_query("DELETE FROM table")
        assert "Error" in result
        assert "read-only" in result

    @pytest.mark.asyncio
    async def test_handle_database_query_allowed(self, ml_plan_agent):
        """Test that SELECT queries are allowed."""
        # Mock the database tool
        with patch("arc.tools.database_query.DatabaseQueryTool") as mock_tool_class:
            mock_tool = MagicMock()
            mock_result = MagicMock()
            mock_result.success = True
            mock_result.output = "Query result: 10 rows"
            mock_tool.execute = AsyncMock(return_value=mock_result)
            mock_tool_class.return_value = mock_tool

            result = await ml_plan_agent._handle_database_query(
                "SELECT * FROM table LIMIT 10"
            )

            assert "Query result" in result
            assert "10 rows" in result

    @pytest.mark.asyncio
    async def test_execute_tool_with_progress(self, ml_plan_agent):
        """Test tool execution (progress reporting handled by BaseAgent)."""
        # Test list_available_knowledge
        result = await ml_plan_agent._execute_ml_tool("list_available_knowledge", "{}")
        assert result is not None
        assert isinstance(result, str)

        # Test read_knowledge_content
        result = await ml_plan_agent._execute_ml_tool(
            "read_knowledge_content", '{"knowledge_id": "mlp", "phase": "model"}'
        )
        assert result is not None
        assert isinstance(result, str)
        assert "mlp" in result.lower() or "feedforward" in result.lower()

    @pytest.mark.asyncio
    async def test_execute_tool_database_query_progress(self, ml_plan_agent):
        """Test database query execution (progress handled by BaseAgent)."""
        # Mock the database tool to avoid actual query
        with patch("arc.tools.database_query.DatabaseQueryTool") as mock_tool_class:
            mock_tool = MagicMock()
            mock_result = MagicMock()
            mock_result.success = True
            mock_result.output = "Query result: 10 rows"
            mock_tool.execute = AsyncMock(return_value=mock_result)
            mock_tool_class.return_value = mock_tool

            query = "SELECT target, COUNT(*) FROM table GROUP BY target"
            result = await ml_plan_agent._execute_ml_tool(
                "database_query", f'{{"query": "{query}"}}'
            )

            # Check that tool executed successfully
            assert result is not None
            assert "Query result: 10 rows" in result
