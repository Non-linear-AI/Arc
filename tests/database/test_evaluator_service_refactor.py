"""Tests for evaluator service refactoring (trainer_id -> model_id)."""

import tempfile
from datetime import datetime
from pathlib import Path

import pytest

from arc.database.duckdb import DuckDBDatabase
from arc.database.manager import DatabaseManager
from arc.database.models.evaluator import Evaluator
from arc.database.services.evaluator_service import EvaluatorService


@pytest.fixture
def temp_db():
    """Create a temporary database for testing."""
    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = Path(tmpdir) / "test.db"
        db = DuckDBDatabase(str(db_path))
        yield db
        db.close()


@pytest.fixture
def db_manager(temp_db):
    """Create a database manager with temporary database."""
    return DatabaseManager(temp_db)


@pytest.fixture
def evaluator_service(db_manager):
    """Create an evaluator service."""
    return EvaluatorService(db_manager)


@pytest.fixture
def sample_evaluator():
    """Create a sample evaluator for testing."""
    return Evaluator(
        id="eval-001",
        name="test-evaluator",
        version=1,
        model_id="test-model",
        model_version=1,
        spec="evaluator_spec_yaml",
        description="Test evaluator",
        plan_id=None,
        created_at=datetime.now(),
        updated_at=datetime.now(),
    )


class TestEvaluatorServiceRefactor:
    """Tests for evaluator service after trainer -> model refactoring."""

    def test_create_evaluator_with_model_id(self, evaluator_service, sample_evaluator):
        """Test creating an evaluator with model_id."""
        evaluator_service.create_evaluator(sample_evaluator)

        # Retrieve and verify
        retrieved = evaluator_service.get_evaluator_by_id(sample_evaluator.id)
        assert retrieved is not None
        assert retrieved.id == sample_evaluator.id
        assert retrieved.model_id == "test-model"
        assert retrieved.model_version == 1

    def test_get_evaluators_by_model(self, evaluator_service):
        """Test getting evaluators by model_id."""
        # Create multiple evaluators for the same model
        eval1 = Evaluator(
            id="eval-001",
            name="evaluator-1",
            version=1,
            model_id="model-123",
            model_version=1,
            spec="spec1",
            description="First evaluator",
            created_at=datetime.now(),
            updated_at=datetime.now(),
        )
        eval2 = Evaluator(
            id="eval-002",
            name="evaluator-2",
            version=1,
            model_id="model-123",
            model_version=2,
            spec="spec2",
            description="Second evaluator",
            created_at=datetime.now(),
            updated_at=datetime.now(),
        )
        eval3 = Evaluator(
            id="eval-003",
            name="evaluator-3",
            version=1,
            model_id="model-456",
            model_version=1,
            spec="spec3",
            description="Different model",
            created_at=datetime.now(),
            updated_at=datetime.now(),
        )

        evaluator_service.create_evaluator(eval1)
        evaluator_service.create_evaluator(eval2)
        evaluator_service.create_evaluator(eval3)

        # Get evaluators for model-123
        evaluators = evaluator_service.get_evaluators_by_model("model-123")
        assert len(evaluators) == 2
        assert all(e.model_id == "model-123" for e in evaluators)

        # Get evaluators for model-456
        evaluators = evaluator_service.get_evaluators_by_model("model-456")
        assert len(evaluators) == 1
        assert evaluators[0].model_id == "model-456"

    def test_get_evaluators_by_model_empty(self, evaluator_service):
        """Test getting evaluators for non-existent model."""
        evaluators = evaluator_service.get_evaluators_by_model("nonexistent-model")
        assert len(evaluators) == 0

    def test_update_evaluator_model_id(self, evaluator_service, sample_evaluator):
        """Test updating an evaluator's model_id."""
        evaluator_service.create_evaluator(sample_evaluator)

        # Update model_id
        sample_evaluator.model_id = "new-model-id"
        sample_evaluator.model_version = 2
        sample_evaluator.updated_at = datetime.now()
        evaluator_service.update_evaluator(sample_evaluator)

        # Verify update
        retrieved = evaluator_service.get_evaluator_by_id(sample_evaluator.id)
        assert retrieved.model_id == "new-model-id"
        assert retrieved.model_version == 2

    def test_list_all_evaluators_contains_model_id(self, evaluator_service, sample_evaluator):
        """Test that listing evaluators includes model_id field."""
        evaluator_service.create_evaluator(sample_evaluator)

        evaluators = evaluator_service.list_all_evaluators()
        assert len(evaluators) >= 1

        # Find our evaluator
        our_eval = next((e for e in evaluators if e.id == sample_evaluator.id), None)
        assert our_eval is not None
        assert hasattr(our_eval, "model_id")
        assert hasattr(our_eval, "model_version")
        assert our_eval.model_id == "test-model"
        assert our_eval.model_version == 1

    def test_get_evaluator_by_name_version_with_model_id(self, evaluator_service):
        """Test getting evaluator by name and version includes model_id."""
        evaluator = Evaluator(
            id="eval-unique",
            name="unique-evaluator",
            version=5,
            model_id="model-xyz",
            model_version=3,
            spec="spec_content",
            description="Unique evaluator",
            created_at=datetime.now(),
            updated_at=datetime.now(),
        )
        evaluator_service.create_evaluator(evaluator)

        # Retrieve by name and version
        retrieved = evaluator_service.get_evaluator_by_name_version("unique-evaluator", 5)
        assert retrieved is not None
        assert retrieved.model_id == "model-xyz"
        assert retrieved.model_version == 3

    def test_get_latest_evaluator_with_model_id(self, evaluator_service):
        """Test getting latest evaluator includes model_id."""
        # Create multiple versions
        for i in range(1, 4):
            evaluator = Evaluator(
                id=f"eval-v{i}",
                name="versioned-evaluator",
                version=i,
                model_id=f"model-v{i}",
                model_version=i,
                spec=f"spec_v{i}",
                description=f"Version {i}",
                created_at=datetime.now(),
                updated_at=datetime.now(),
            )
            evaluator_service.create_evaluator(evaluator)

        # Get latest version
        latest = evaluator_service.get_latest_evaluator_by_name("versioned-evaluator")
        assert latest is not None
        assert latest.version == 3
        assert latest.model_id == "model-v3"
        assert latest.model_version == 3

    def test_evaluator_versioning_with_models(self, evaluator_service):
        """Test evaluator versioning works correctly with model references."""
        # Create v1 evaluator
        eval_v1 = Evaluator(
            id="eval-v1",
            name="my-evaluator",
            version=1,
            model_id="model-a",
            model_version=1,
            spec="spec_v1",
            description="Version 1",
            created_at=datetime.now(),
            updated_at=datetime.now(),
        )
        evaluator_service.create_evaluator(eval_v1)

        # Get next version number
        next_version = evaluator_service.get_next_version_for_name("my-evaluator")
        assert next_version == 2

        # Create v2 evaluator pointing to different model
        eval_v2 = Evaluator(
            id="eval-v2",
            name="my-evaluator",
            version=next_version,
            model_id="model-b",
            model_version=1,
            spec="spec_v2",
            description="Version 2",
            created_at=datetime.now(),
            updated_at=datetime.now(),
        )
        evaluator_service.create_evaluator(eval_v2)

        # Verify both versions exist
        all_versions = evaluator_service.get_evaluators_by_name("my-evaluator")
        assert len(all_versions) == 2
        assert all_versions[0].version == 2  # Ordered DESC
        assert all_versions[0].model_id == "model-b"
        assert all_versions[1].version == 1
        assert all_versions[1].model_id == "model-a"

    def test_no_trainer_id_in_evaluator(self, evaluator_service, sample_evaluator):
        """Test that evaluator objects don't have trainer_id attribute."""
        evaluator_service.create_evaluator(sample_evaluator)

        retrieved = evaluator_service.get_evaluator_by_id(sample_evaluator.id)
        assert retrieved is not None

        # Verify no trainer_id attribute
        assert not hasattr(retrieved, "trainer_id")
        assert not hasattr(retrieved, "trainer_version")

        # Verify model_id attributes exist
        assert hasattr(retrieved, "model_id")
        assert hasattr(retrieved, "model_version")

    def test_delete_evaluator_by_model(self, evaluator_service):
        """Test deleting evaluators and verifying by model query."""
        # Create evaluators
        eval1 = Evaluator(
            id="eval-to-delete",
            name="deletable",
            version=1,
            model_id="model-delete",
            model_version=1,
            spec="spec",
            description="Will be deleted",
            created_at=datetime.now(),
            updated_at=datetime.now(),
        )
        evaluator_service.create_evaluator(eval1)

        # Verify it exists
        assert evaluator_service.evaluator_exists("eval-to-delete")
        evaluators = evaluator_service.get_evaluators_by_model("model-delete")
        assert len(evaluators) == 1

        # Delete it
        evaluator_service.delete_evaluator("eval-to-delete")

        # Verify deletion
        assert not evaluator_service.evaluator_exists("eval-to-delete")
        evaluators = evaluator_service.get_evaluators_by_model("model-delete")
        assert len(evaluators) == 0

    def test_sql_injection_protection_model_id(self, evaluator_service):
        """Test SQL injection protection for model_id field."""
        # Try to create evaluator with SQL injection attempt in model_id
        malicious_evaluator = Evaluator(
            id="eval-malicious",
            name="safe-name",
            version=1,
            model_id="'; DROP TABLE evaluators; --",
            model_version=1,
            spec="spec",
            description="Testing SQL injection",
            created_at=datetime.now(),
            updated_at=datetime.now(),
        )

        # This should not raise an error (string escaping should handle it)
        evaluator_service.create_evaluator(malicious_evaluator)

        # Verify the evaluator was created with escaped string
        retrieved = evaluator_service.get_evaluator_by_id("eval-malicious")
        assert retrieved is not None
        assert "DROP TABLE" in retrieved.model_id  # String is stored as-is

        # Verify table still exists (query should work)
        all_evals = evaluator_service.list_all_evaluators()
        assert len(all_evals) >= 1
