"""Tests for ReadKnowledgeTool."""

import pytest

from arc.core.agents.shared.knowledge_loader import KnowledgeLoader
from arc.tools.knowledge import ReadKnowledgeTool


def get_bundled_knowledge_path():
    """Get path to bundled knowledge in package."""
    loader = KnowledgeLoader()
    return loader.builtin_path


class TestReadKnowledgeTool:
    """Tests for ReadKnowledgeTool."""

    @pytest.fixture
    def temp_knowledge_dir(self, tmp_path):
        """Create temporary knowledge directory."""
        knowledge_dir = tmp_path / "knowledge"
        knowledge_dir.mkdir()

        # Create test knowledge
        test_dir = knowledge_dir / "test"
        test_dir.mkdir()

        metadata = """id: test
name: "Test Knowledge"
type: pattern
description: "Test pattern"
keywords: [test]
phases: [model]
"""
        (test_dir / "metadata.yaml").write_text(metadata)
        (test_dir / "model-guide.md").write_text("# Test Guide\n\nTest content")

        return knowledge_dir

    def test_tool_initialization(self, temp_knowledge_dir):
        """Test tool initialization."""
        tool = ReadKnowledgeTool(builtin_path=temp_knowledge_dir, user_path=None)
        assert tool.knowledge_loader.builtin_path == temp_knowledge_dir

    @pytest.mark.asyncio
    async def test_execute_success(self, temp_knowledge_dir):
        """Test successful knowledge read."""
        tool = ReadKnowledgeTool(builtin_path=temp_knowledge_dir, user_path=None)
        result = await tool.execute(knowledge_id="test", phase="model")

        assert result.success is True
        assert "Test Guide" in result.output
        assert "Test content" in result.output
        assert result.metadata["knowledge_id"] == "test"
        assert result.metadata["phase"] == "model"

    @pytest.mark.asyncio
    async def test_execute_missing_knowledge_id(self, temp_knowledge_dir):
        """Test execution without knowledge_id."""
        tool = ReadKnowledgeTool(builtin_path=temp_knowledge_dir, user_path=None)
        result = await tool.execute()

        assert result.success is False
        assert "Missing required parameter" in result.error
        assert result.recovery_actions is not None

    @pytest.mark.asyncio
    async def test_execute_nonexistent_knowledge(self, temp_knowledge_dir):
        """Test reading nonexistent knowledge."""
        tool = ReadKnowledgeTool(builtin_path=temp_knowledge_dir, user_path=None)
        result = await tool.execute(knowledge_id="nonexistent")

        assert result.success is False
        assert "not found" in result.error
        assert "Available knowledge" in result.recovery_actions

    @pytest.mark.asyncio
    async def test_execute_with_metadata_header(self, temp_knowledge_dir):
        """Test that output includes metadata header."""
        tool = ReadKnowledgeTool(builtin_path=temp_knowledge_dir, user_path=None)
        result = await tool.execute(knowledge_id="test")

        assert result.success is True
        assert "Test Knowledge" in result.output  # Name from metadata
        assert "pattern" in result.output  # Type from metadata
        assert "Test pattern" in result.output  # Description from metadata

    @pytest.mark.asyncio
    async def test_execute_default_phase(self, temp_knowledge_dir):
        """Test that default phase is 'model'."""
        tool = ReadKnowledgeTool(builtin_path=temp_knowledge_dir, user_path=None)
        result = await tool.execute(knowledge_id="test")

        assert result.success is True
        assert result.metadata["phase"] == "model"


class TestReadKnowledgeToolWithRealData:
    """Tests using real bundled knowledge."""

    @pytest.mark.asyncio
    async def test_read_dcn_knowledge(self):
        """Test reading real DCN knowledge."""
        bundled_path = get_bundled_knowledge_path()
        tool = ReadKnowledgeTool(bundled_path)

        result = await tool.execute(knowledge_id="dcn", phase="model")

        assert result.success is True
        assert "Deep & Cross" in result.output or "DCN" in result.output
        assert result.metadata["knowledge_id"] == "dcn"

    @pytest.mark.asyncio
    async def test_read_mlp_knowledge(self):
        """Test reading MLP knowledge."""
        bundled_path = get_bundled_knowledge_path()
        tool = ReadKnowledgeTool(bundled_path)

        result = await tool.execute(knowledge_id="mlp")

        assert result.success is True
        assert "mlp" in result.output.lower() or "perceptron" in result.output.lower()

    @pytest.mark.asyncio
    async def test_list_available_knowledge_in_error(self):
        """Test that error message lists available knowledge."""
        bundled_path = get_bundled_knowledge_path()
        tool = ReadKnowledgeTool(bundled_path)

        result = await tool.execute(knowledge_id="nonexistent")

        assert result.success is False
        # Should list dcn and mlp as available
        assert "dcn" in result.recovery_actions
        assert "mlp" in result.recovery_actions
