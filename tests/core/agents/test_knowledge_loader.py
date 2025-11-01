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
            "name": "Deep & Cross Network",
            "description": "Test description",
            "phases": ["model"],
        }

        metadata = KnowledgeMetadata("dcn", data)

        assert metadata.id == "dcn"
        assert metadata.name == "Deep & Cross Network"
        assert metadata.description == "Test description"
        assert metadata.phases == ["model"]

    def test_metadata_to_dict(self):
        """Test converting metadata to dict."""
        data = {
            "name": "Test Knowledge",
            "description": "Test",
            "phases": ["model"],
        }

        metadata = KnowledgeMetadata("test", data)
        result = metadata.to_dict()

        assert result["id"] == "test"
        assert result["name"] == "Test Knowledge"
        assert result["description"] == "Test"
        assert result["phases"] == ["model"]

    def test_metadata_str(self):
        """Test string representation."""
        data = {
            "name": "Test",
            "description": "Test description",
            "phases": ["data", "model"],
        }

        metadata = KnowledgeMetadata("test", data)
        str_repr = str(metadata)

        assert "test" in str_repr
        assert "Test" in str_repr
        assert "Test description" in str_repr
        assert "data, model" in str_repr


class TestKnowledgeBuiltin:
    """Tests for builtin knowledge."""

    def test_get_bundled_knowledge_path(self):
        """Test getting bundled knowledge path."""
        path = get_bundled_knowledge_path()
        assert path.exists()
        assert path.name == "knowledge"
        # Check for flat structure
        assert (path / "metadata.yaml").exists()
        assert (path / "dcn.md").exists()
        assert (path / "mlp.md").exists()

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

        content = loader.load_knowledge("dcn")

        assert content is not None
        assert "Deep & Cross" in content or "DCN" in content


class TestKnowledgeLoader:
    """Tests for KnowledgeLoader class."""

    @pytest.fixture
    def temp_knowledge_dir(self, tmp_path):
        """Create temporary knowledge directory with test data."""
        knowledge_dir = tmp_path / "knowledge"
        knowledge_dir.mkdir()

        # Create metadata.yaml (flat structure)
        metadata = """test_arch:
  name: "Test Architecture"
  description: "Test architecture description"
  phases:
    - model
"""
        (knowledge_dir / "metadata.yaml").write_text(metadata)

        # Create knowledge content file (flat structure)
        guide = "# Test Architecture\n\nThis is a test guide."
        (knowledge_dir / "test_arch.md").write_text(guide)

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
        assert metadata_map["test_arch"].phases == ["model"]

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
        content = loader.load_knowledge("test_arch")

        assert content is not None
        assert "Test Architecture" in content
        assert "test guide" in content

    def test_load_nonexistent_knowledge(self, temp_knowledge_dir):
        """Test loading nonexistent knowledge."""
        # Use non-existent path for user, temp for builtin
        loader = KnowledgeLoader(
            builtin_path=temp_knowledge_dir,
            user_path=temp_knowledge_dir / "nonexistent",
        )
        content = loader.load_knowledge("nonexistent")

        assert content is None

    def test_format_metadata_for_llm(self, temp_knowledge_dir):
        """Test formatting metadata for LLM."""
        loader = KnowledgeLoader(builtin_path=temp_knowledge_dir, user_path=None)
        formatted = loader.format_metadata_for_llm()

        assert "test_arch" in formatted
        assert "Test Architecture" in formatted
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
        # Create builtin knowledge (flat structure)
        builtin_dir = tmp_path / "builtin"
        builtin_dir.mkdir()
        (builtin_dir / "metadata.yaml").write_text(
            "test:\n  name: Builtin\n  description: Builtin version\n  phases: [model]"
        )
        (builtin_dir / "test.md").write_text("# Builtin Guide")

        # Create user knowledge (override, flat structure)
        user_dir = tmp_path / "user"
        user_dir.mkdir()
        (user_dir / "metadata.yaml").write_text(
            "test:\n  name: User Override\n  description: User version\n  phases: [model]"
        )
        (user_dir / "test.md").write_text("# User Guide")

        # Load with both paths
        loader = KnowledgeLoader(builtin_path=builtin_dir, user_path=user_dir)

        # Metadata should be from user
        metadata_map = loader.scan_metadata()
        assert metadata_map["test"].name == "User Override"
        assert metadata_map["test"].description == "User version"

        # Content should be from user
        content = loader.load_knowledge("test")
        assert content is not None
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
        assert dcn_metadata.name == "Deep & Cross Network"
        assert "model" in dcn_metadata.phases

        # Load DCN guide
        content = loader.load_knowledge("dcn")
        assert content is not None
        assert "Deep & Cross" in content or "DCN" in content
        assert "cross" in content.lower()

    def test_load_mlp_knowledge(self):
        """Test loading MLP knowledge."""
        bundled_path = get_bundled_knowledge_path()
        loader = KnowledgeLoader(bundled_path)

        content = loader.load_knowledge("mlp")
        assert content is not None
        assert "mlp" in content.lower() or "multilayer perceptron" in content.lower()
