"""Tests for knowledge loader."""

from pathlib import Path

import pytest

from arc.core.agents.shared.knowledge_loader import KnowledgeLoader, KnowledgeMetadata


def get_bundled_knowledge_path() -> Path:
    """Get path to bundled knowledge in package."""
    # This is the builtin knowledge path

    loader = KnowledgeLoader()
    return loader.builtin_path


class TestKnowledgeMetadata:
    """Tests for KnowledgeMetadata class."""

    def test_metadata_creation(self):
        """Test creating metadata from dict."""
        data = {
            "id": "dcn",
            "name": "Deep & Cross Network",
            "type": "architecture",
            "description": "Test description",
            "keywords": ["dcn", "feature-crossing"],
            "problem_type": "classification",
            "complexity": "intermediate",
            "domain": "recommendation",
        }

        metadata = KnowledgeMetadata(data)

        assert metadata.id == "dcn"
        assert metadata.name == "Deep & Cross Network"
        assert metadata.type == "architecture"
        assert metadata.keywords == ["dcn", "feature-crossing"]
        assert metadata.complexity == "intermediate"
        assert metadata.domain == "recommendation"

    def test_metadata_to_dict(self):
        """Test converting metadata to dict."""
        data = {
            "id": "test",
            "name": "Test Knowledge",
            "type": "pattern",
            "description": "Test",
            "keywords": ["test"],
            "problem_type": "classification",
            "phases": ["model"],
            "complexity": "basic",
            "domain": "general",
        }

        metadata = KnowledgeMetadata(data)
        result = metadata.to_dict()

        assert result["id"] == "test"
        assert result["type"] == "pattern"
        assert result["keywords"] == ["test"]

    def test_metadata_str(self):
        """Test string representation."""
        data = {
            "id": "test",
            "name": "Test",
            "type": "pattern",
            "description": "Test description",
            "keywords": ["k1", "k2"],
        }

        metadata = KnowledgeMetadata(data)
        str_repr = str(metadata)

        assert "test" in str_repr
        assert "pattern" in str_repr
        assert "k1, k2" in str_repr
        assert "Test description" in str_repr


class TestKnowledgeBuiltin:
    """Tests for builtin knowledge."""

    def test_get_bundled_knowledge_path(self):
        """Test getting bundled knowledge path."""
        path = get_bundled_knowledge_path()
        assert path.exists()
        assert path.name == "knowledge"
        assert (path / "dcn").exists()

    def test_builtin_knowledge_loads(self):
        """Test that builtin knowledge loads correctly."""
        loader = KnowledgeLoader()

        # Should load from builtin path
        metadata_map = loader.scan_metadata()

        assert "dcn" in metadata_map
        assert "mlp" in metadata_map

    def test_load_builtin_knowledge_content(self):
        """Test loading builtin knowledge content."""
        loader = KnowledgeLoader()

        content, actual_phase = loader.load_knowledge("dcn", phase="model")

        assert content is not None
        assert actual_phase is not None
        assert "Deep & Cross" in content or "DCN" in content


class TestKnowledgeLoader:
    """Tests for KnowledgeLoader class."""

    @pytest.fixture
    def temp_knowledge_dir(self, tmp_path):
        """Create temporary knowledge directory with test data."""
        knowledge_dir = tmp_path / "knowledge"
        knowledge_dir.mkdir()

        # Create test knowledge
        test_knowledge = knowledge_dir / "test_arch"
        test_knowledge.mkdir()

        # Create metadata
        metadata = """id: test_arch
name: "Test Architecture"
type: architecture
description: "Test architecture description"
keywords:
  - test
  - architecture
problem_type: classification
phases:
  - model
complexity: basic
domain: general
"""
        (test_knowledge / "metadata.yaml").write_text(metadata)

        # Create guide
        guide = "# Test Architecture\n\nThis is a test guide."
        (test_knowledge / "model-guide.md").write_text(guide)

        return knowledge_dir

    def test_loader_initialization(self, temp_knowledge_dir):
        """Test loader initialization."""
        loader = KnowledgeLoader(builtin_path=temp_knowledge_dir, user_path=None)
        assert loader.builtin_path == temp_knowledge_dir

    def test_scan_metadata(self, temp_knowledge_dir):
        """Test scanning metadata."""
        loader = KnowledgeLoader(builtin_path=temp_knowledge_dir, user_path=None)
        metadata_map = loader.scan_metadata()

        assert "test_arch" in metadata_map
        assert metadata_map["test_arch"].name == "Test Architecture"
        assert metadata_map["test_arch"].type == "architecture"

    def test_get_metadata_list(self, temp_knowledge_dir):
        """Test getting metadata list."""
        loader = KnowledgeLoader(builtin_path=temp_knowledge_dir, user_path=None)
        metadata_list = loader.get_metadata_list()

        assert len(metadata_list) >= 1
        test_arch = [m for m in metadata_list if m.id == "test_arch"]
        assert len(test_arch) == 1

    def test_load_knowledge(self, temp_knowledge_dir):
        """Test loading knowledge content."""
        loader = KnowledgeLoader(builtin_path=temp_knowledge_dir, user_path=None)
        content, actual_phase = loader.load_knowledge("test_arch", phase="model")

        assert content is not None
        assert actual_phase == "model"
        assert "Test Architecture" in content
        assert "test guide" in content

    def test_load_nonexistent_knowledge(self, temp_knowledge_dir):
        """Test loading nonexistent knowledge."""
        # Use non-existent path for user, temp for builtin
        loader = KnowledgeLoader(
            builtin_path=temp_knowledge_dir,
            user_path=temp_knowledge_dir / "nonexistent",
        )
        content, actual_phase = loader.load_knowledge("nonexistent")

        assert content is None
        assert actual_phase is None

    def test_format_metadata_for_llm(self, temp_knowledge_dir):
        """Test formatting metadata for LLM."""
        loader = KnowledgeLoader(builtin_path=temp_knowledge_dir, user_path=None)
        formatted = loader.format_metadata_for_llm()

        assert "test_arch" in formatted
        assert "architecture" in formatted.lower()
        assert "Test architecture description" in formatted

    def test_metadata_caching(self, temp_knowledge_dir):
        """Test that metadata is cached."""
        loader = KnowledgeLoader(builtin_path=temp_knowledge_dir, user_path=None)

        # First scan
        metadata1 = loader.scan_metadata()

        # Second scan should return cached
        metadata2 = loader.scan_metadata()

        assert metadata1 is metadata2  # Same object due to caching

    def test_user_overrides_builtin(self, tmp_path):
        """Test that user knowledge overrides builtin."""
        # Create builtin knowledge
        builtin_dir = tmp_path / "builtin"
        builtin_dir.mkdir()
        builtin_knowledge = builtin_dir / "test"
        builtin_knowledge.mkdir()
        (builtin_knowledge / "metadata.yaml").write_text(
            "id: test\nname: Builtin\ntype: architecture\n"
            "description: Builtin version\nkeywords: [builtin]\nphases: [model]"
        )
        (builtin_knowledge / "model-guide.md").write_text("# Builtin Guide")

        # Create user knowledge (override)
        user_dir = tmp_path / "user"
        user_dir.mkdir()
        user_knowledge = user_dir / "test"
        user_knowledge.mkdir()
        (user_knowledge / "metadata.yaml").write_text(
            "id: test\nname: User Override\ntype: architecture\n"
            "description: User version\nkeywords: [user]\nphases: [model]"
        )
        (user_knowledge / "model-guide.md").write_text("# User Guide")

        # Load with both paths
        loader = KnowledgeLoader(builtin_path=builtin_dir, user_path=user_dir)

        # Metadata should be from user
        metadata_map = loader.scan_metadata()
        assert metadata_map["test"].name == "User Override"
        assert metadata_map["test"].description == "User version"

        # Content should be from user
        content, actual_phase = loader.load_knowledge("test", phase="model")
        assert content is not None
        assert actual_phase == "model"
        assert "User Guide" in content
        assert "Builtin Guide" not in content


class TestKnowledgeLoaderWithRealData:
    """Tests using real bundled knowledge data."""

    def test_load_dcn_knowledge(self):
        """Test loading real DCN knowledge."""
        bundled_path = get_bundled_knowledge_path()
        loader = KnowledgeLoader(bundled_path)

        # Load DCN metadata
        metadata_map = loader.scan_metadata()
        assert "dcn" in metadata_map

        dcn_metadata = metadata_map["dcn"]
        assert dcn_metadata.type == "architecture"
        assert "dcn" in [k.lower() for k in dcn_metadata.keywords]

        # Load DCN guide
        content, actual_phase = loader.load_knowledge("dcn", phase="model")
        assert content is not None
        assert actual_phase is not None
        assert "Deep & Cross" in content or "DCN" in content
        assert "cross" in content.lower()

    def test_load_mlp_knowledge(self):
        """Test loading MLP knowledge."""
        bundled_path = get_bundled_knowledge_path()
        loader = KnowledgeLoader(bundled_path)

        content, actual_phase = loader.load_knowledge("mlp", phase="model")
        assert content is not None
        assert actual_phase is not None
        assert "mlp" in content.lower() or "multilayer perceptron" in content.lower()
