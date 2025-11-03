"""Tests for vocabulary persistence with model artifacts."""

import json

import torch.nn as nn

from arc.ml.artifacts import ModelArtifact, ModelArtifactManager


class SimpleModel(nn.Module):
    """Simple test model."""

    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(10, 1)

    def forward(self, x):
        return self.linear(x)


class TestVocabularySaveLoad:
    """Test saving and loading vocabularies with model artifacts."""

    def test_save_single_vocabulary(self, tmp_path):
        """Test saving a single vocabulary with model artifact."""
        manager = ModelArtifactManager(tmp_path)
        model = SimpleModel()

        # Create vocabulary for a categorical feature
        vocabularies = {
            "user_id": {
                "vocabulary": {"user_1": 0, "user_2": 1, "user_3": 2},
                "vocab_size": 3,
            }
        }

        artifact = ModelArtifact(
            model_id="test_model", model_name="Test Model", version=1
        )

        # Save model with vocabularies
        artifact_dir = manager.save_model_artifact(
            model=model, artifact=artifact, vocabularies=vocabularies
        )

        # Check that vocabularies file was created
        vocab_file = artifact_dir / "vocabularies.json"
        assert vocab_file.exists()

        # Verify contents
        with open(vocab_file) as f:
            saved_vocabs = json.load(f)

        assert "user_id" in saved_vocabs
        assert saved_vocabs["user_id"]["vocab_size"] == 3
        assert "user_1" in saved_vocabs["user_id"]["vocabulary"]
        assert saved_vocabs["user_id"]["vocabulary"]["user_1"] == 0

    def test_save_multiple_vocabularies(self, tmp_path):
        """Test saving multiple vocabularies for different categorical features."""
        manager = ModelArtifactManager(tmp_path)
        model = SimpleModel()

        vocabularies = {
            "user_id": {
                "vocabulary": {"alice": 0, "bob": 1, "charlie": 2},
                "vocab_size": 3,
            },
            "item_id": {
                "vocabulary": {"item_a": 0, "item_b": 1, "item_c": 2, "item_d": 3},
                "vocab_size": 4,
            },
            "category": {
                "vocabulary": {"electronics": 0, "books": 1},
                "vocab_size": 2,
            },
        }

        artifact = ModelArtifact(
            model_id="multi_vocab_model", model_name="Multi Vocab Model", version=1
        )

        artifact_dir = manager.save_model_artifact(
            model=model, artifact=artifact, vocabularies=vocabularies
        )

        # Load and verify
        vocab_file = artifact_dir / "vocabularies.json"
        with open(vocab_file) as f:
            saved_vocabs = json.load(f)

        assert len(saved_vocabs) == 3
        assert "user_id" in saved_vocabs
        assert "item_id" in saved_vocabs
        assert "category" in saved_vocabs

        # Verify specific vocabularies
        assert saved_vocabs["user_id"]["vocab_size"] == 3
        assert saved_vocabs["item_id"]["vocab_size"] == 4
        assert saved_vocabs["category"]["vocab_size"] == 2

    def test_load_vocabularies_with_artifact(self, tmp_path):
        """Test loading vocabularies along with model artifact."""
        manager = ModelArtifactManager(tmp_path)
        model = SimpleModel()

        vocabularies = {
            "genre": {
                "vocabulary": {"action": 0, "comedy": 1, "drama": 2, "sci-fi": 3},
                "vocab_size": 4,
            }
        }

        artifact = ModelArtifact(
            model_id="vocab_load_test", model_name="Vocab Load Test", version=1
        )

        # Save
        manager.save_model_artifact(
            model=model, artifact=artifact, vocabularies=vocabularies
        )

        # Load vocabularies
        loaded_vocabs = manager.load_vocabularies("vocab_load_test", version=1)

        # Verify
        assert loaded_vocabs is not None
        assert "genre" in loaded_vocabs
        assert loaded_vocabs["genre"]["vocab_size"] == 4
        assert loaded_vocabs["genre"]["vocabulary"]["action"] == 0
        assert loaded_vocabs["genre"]["vocabulary"]["sci-fi"] == 3

    def test_load_vocabularies_latest_version(self, tmp_path):
        """Test loading vocabularies from latest version without specifying."""
        manager = ModelArtifactManager(tmp_path)
        model = SimpleModel()

        # Save version 1
        vocab_v1 = {"cat": {"vocabulary": {"a": 0}, "vocab_size": 1}}
        artifact_v1 = ModelArtifact(model_id="versioned", model_name="V1", version=1)
        manager.save_model_artifact(
            model=model, artifact=artifact_v1, vocabularies=vocab_v1
        )

        # Save version 2 with different vocabulary
        vocab_v2 = {"cat": {"vocabulary": {"a": 0, "b": 1}, "vocab_size": 2}}
        artifact_v2 = ModelArtifact(model_id="versioned", model_name="V2", version=2)
        manager.save_model_artifact(
            model=model, artifact=artifact_v2, vocabularies=vocab_v2
        )

        # Load latest (should be v2)
        loaded_vocabs = manager.load_vocabularies("versioned")

        assert loaded_vocabs["cat"]["vocab_size"] == 2
        assert "b" in loaded_vocabs["cat"]["vocabulary"]

    def test_save_without_vocabularies(self, tmp_path):
        """Test saving model without vocabularies (for non-categorical models)."""
        manager = ModelArtifactManager(tmp_path)
        model = SimpleModel()

        artifact = ModelArtifact(
            model_id="no_vocab_model", model_name="No Vocab Model", version=1
        )

        # Save without vocabularies parameter
        artifact_dir = manager.save_model_artifact(model=model, artifact=artifact)

        # Vocabularies file should not exist
        vocab_file = artifact_dir / "vocabularies.json"
        assert not vocab_file.exists()

    def test_load_nonexistent_vocabularies(self, tmp_path):
        """Test loading vocabularies when none were saved."""
        manager = ModelArtifactManager(tmp_path)
        model = SimpleModel()

        artifact = ModelArtifact(model_id="no_vocab", model_name="No Vocab", version=1)

        # Save without vocabularies
        manager.save_model_artifact(model=model, artifact=artifact)

        # Load vocabularies (should return None or empty dict)
        loaded_vocabs = manager.load_vocabularies("no_vocab", version=1)

        assert loaded_vocabs is None or loaded_vocabs == {}

    def test_vocabulary_with_none_values(self, tmp_path):
        """Test vocabulary that includes None (null) values."""
        manager = ModelArtifactManager(tmp_path)
        model = SimpleModel()

        vocabularies = {
            "status": {
                "vocabulary": {"active": 0, "inactive": 1, None: 2},
                "vocab_size": 3,
            }
        }

        artifact = ModelArtifact(
            model_id="null_vocab", model_name="Null Vocab", version=1
        )

        # Save with None in vocabulary
        manager.save_model_artifact(
            model=model, artifact=artifact, vocabularies=vocabularies
        )

        # Load and verify None is preserved
        loaded_vocabs = manager.load_vocabularies("null_vocab", version=1)

        assert loaded_vocabs is not None
        # JSON serializes None as "null" (string), so check for either
        vocab_dict = loaded_vocabs["status"]["vocabulary"]
        assert None in vocab_dict or "null" in vocab_dict

    def test_overwrite_vocabularies(self, tmp_path):
        """Test overwriting vocabularies when updating model."""
        manager = ModelArtifactManager(tmp_path)
        model = SimpleModel()

        # Save initial version
        vocab_initial = {"field": {"vocabulary": {"x": 0}, "vocab_size": 1}}
        artifact = ModelArtifact(
            model_id="overwrite_test", model_name="Overwrite", version=1
        )
        manager.save_model_artifact(
            model=model, artifact=artifact, vocabularies=vocab_initial
        )

        # Overwrite with new vocabulary
        vocab_new = {"field": {"vocabulary": {"x": 0, "y": 1, "z": 2}, "vocab_size": 3}}
        manager.save_model_artifact(
            model=model, artifact=artifact, vocabularies=vocab_new, overwrite=True
        )

        # Load and verify new vocabulary
        loaded_vocabs = manager.load_vocabularies("overwrite_test", version=1)
        assert loaded_vocabs["field"]["vocab_size"] == 3
        assert "z" in loaded_vocabs["field"]["vocabulary"]

    def test_vocabulary_persistence_across_loads(self, tmp_path):
        """Test that vocabularies persist correctly across multiple loads."""
        manager = ModelArtifactManager(tmp_path)
        model = SimpleModel()

        vocabularies = {
            "color": {
                "vocabulary": {"red": 0, "green": 1, "blue": 2, "yellow": 3},
                "vocab_size": 4,
            }
        }

        artifact = ModelArtifact(
            model_id="persist_test", model_name="Persist Test", version=1
        )

        manager.save_model_artifact(
            model=model, artifact=artifact, vocabularies=vocabularies
        )

        # Load multiple times
        loaded_1 = manager.load_vocabularies("persist_test", version=1)
        loaded_2 = manager.load_vocabularies("persist_test", version=1)
        loaded_3 = manager.load_vocabularies("persist_test", version=1)

        # All loads should be identical
        assert loaded_1 == loaded_2 == loaded_3
        assert loaded_1["color"]["vocabulary"]["blue"] == 2

    def test_large_vocabulary(self, tmp_path):
        """Test handling of large vocabularies."""
        manager = ModelArtifactManager(tmp_path)
        model = SimpleModel()

        # Create large vocabulary (1000 unique values)
        large_vocab = {f"item_{i}": i for i in range(1000)}

        vocabularies = {
            "large_category": {"vocabulary": large_vocab, "vocab_size": 1000}
        }

        artifact = ModelArtifact(
            model_id="large_vocab", model_name="Large Vocab", version=1
        )

        # Save and load
        manager.save_model_artifact(
            model=model, artifact=artifact, vocabularies=vocabularies
        )
        loaded_vocabs = manager.load_vocabularies("large_vocab", version=1)

        # Verify size and sample entries
        assert loaded_vocabs["large_category"]["vocab_size"] == 1000
        assert len(loaded_vocabs["large_category"]["vocabulary"]) == 1000
        assert loaded_vocabs["large_category"]["vocabulary"]["item_0"] == 0
        assert loaded_vocabs["large_category"]["vocabulary"]["item_999"] == 999
