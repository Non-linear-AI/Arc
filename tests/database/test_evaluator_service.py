"""Tests for EvaluatorService."""

from datetime import UTC, datetime

import pytest

from arc.database import DatabaseManager
from arc.database.models.evaluator import Evaluator
from arc.database.services.evaluator_service import EvaluatorService


@pytest.fixture
def db_manager(tmp_path):
    """Create temporary file database manager for testing."""
    system_db = tmp_path / "system.db"
    user_db = tmp_path / "user.db"
    with DatabaseManager(str(system_db), str(user_db)) as manager:
        yield manager


@pytest.fixture
def evaluator_service(db_manager):
    """Create an EvaluatorService instance for testing."""
    return EvaluatorService(db_manager)


@pytest.fixture
def sample_evaluator():
    """Create a sample evaluator for testing."""
    now = datetime.now(UTC)
    return Evaluator(
        id="test-eval-1",
        name="diabetes_eval",
        version=1,
        model_id="trainer-123-v1",
        spec="""name: diabetes_eval
trainer_ref: diabetes_trainer
dataset: test_diabetes_data
target_column: outcome
metrics:
  - accuracy
  - precision
""",
        description="Diabetes model evaluator",
        created_at=now,
        updated_at=now,
    )


@pytest.fixture
def sample_evaluators():
    """Create multiple sample evaluators for testing."""
    now = datetime.now(UTC)
    return [
        Evaluator(
            id="eval-1",
            name="diabetes_eval",
            version=1,
            model_id="trainer-123-v1",
            spec="name: diabetes_eval\ntrainer_ref: diabetes_trainer\n",
            description="First version",
            created_at=now,
            updated_at=now,
        ),
        Evaluator(
            id="eval-2",
            name="diabetes_eval",
            version=2,
            model_id="trainer-123-v2",
            spec="name: diabetes_eval\ntrainer_ref: diabetes_trainer_v2\n",
            description="Second version",
            created_at=now,
            updated_at=now,
        ),
        Evaluator(
            id="eval-3",
            name="iris_eval",
            version=1,
            model_id="trainer-456-v1",
            spec="name: iris_eval\ntrainer_ref: iris_trainer\n",
            description="Iris evaluator",
            created_at=now,
            updated_at=now,
        ),
    ]


class TestEvaluatorDataClass:
    """Test the Evaluator dataclass."""

    def test_evaluator_creation(self, sample_evaluator):
        """Test creating an Evaluator instance."""
        assert sample_evaluator.id == "test-eval-1"
        assert sample_evaluator.name == "diabetes_eval"
        assert sample_evaluator.version == 1
        assert sample_evaluator.model_id == "trainer-123-v1"
        assert isinstance(sample_evaluator.created_at, datetime)
        assert isinstance(sample_evaluator.updated_at, datetime)


class TestEvaluatorServiceCreation:
    """Test evaluator creation operations."""

    def test_create_evaluator(self, evaluator_service, sample_evaluator):
        """Test creating a new evaluator."""
        evaluator_service.create_evaluator(sample_evaluator)
        result = evaluator_service.get_evaluator_by_id(sample_evaluator.id)

        assert result is not None
        assert result.id == sample_evaluator.id
        assert result.name == sample_evaluator.name
        assert result.version == sample_evaluator.version

    def test_create_multiple_evaluators(self, evaluator_service, sample_evaluators):
        """Test creating multiple evaluators."""
        for evaluator in sample_evaluators:
            evaluator_service.create_evaluator(evaluator)

        all_evaluators = evaluator_service.list_all_evaluators()
        assert len(all_evaluators) == 3


class TestEvaluatorServiceRetrieval:
    """Test evaluator retrieval operations."""

    def test_get_evaluator_by_id_exists(self, evaluator_service, sample_evaluator):
        """Test retrieving an evaluator by ID when it exists."""
        evaluator_service.create_evaluator(sample_evaluator)
        result = evaluator_service.get_evaluator_by_id(sample_evaluator.id)

        assert result is not None
        assert result.id == sample_evaluator.id
        assert result.name == sample_evaluator.name

    def test_get_evaluator_by_id_not_exists(self, evaluator_service):
        """Test retrieving an evaluator by ID when it doesn't exist."""
        result = evaluator_service.get_evaluator_by_id("nonexistent")
        assert result is None

    def test_get_evaluator_by_name_version(self, evaluator_service, sample_evaluators):
        """Test retrieving an evaluator by name and version."""
        for evaluator in sample_evaluators:
            evaluator_service.create_evaluator(evaluator)

        result = evaluator_service.get_evaluator_by_name_version("diabetes_eval", 2)

        assert result is not None
        assert result.name == "diabetes_eval"
        assert result.version == 2
        assert result.id == "eval-2"

    def test_get_evaluator_by_name_version_not_exists(self, evaluator_service):
        """Test retrieving nonexistent name/version combo."""
        result = evaluator_service.get_evaluator_by_name_version("nonexistent", 1)
        assert result is None

    def test_get_latest_evaluator_by_name(self, evaluator_service, sample_evaluators):
        """Test retrieving the latest version of an evaluator."""
        for evaluator in sample_evaluators:
            evaluator_service.create_evaluator(evaluator)

        result = evaluator_service.get_latest_evaluator_by_name("diabetes_eval")

        assert result is not None
        assert result.name == "diabetes_eval"
        assert result.version == 2
        assert result.id == "eval-2"

    def test_get_latest_evaluator_by_name_not_exists(self, evaluator_service):
        """Test getting latest evaluator for nonexistent name."""
        result = evaluator_service.get_latest_evaluator_by_name("nonexistent")
        assert result is None

    def test_get_evaluators_by_name(self, evaluator_service, sample_evaluators):
        """Test retrieving all evaluators with a specific name."""
        for evaluator in sample_evaluators:
            evaluator_service.create_evaluator(evaluator)

        results = evaluator_service.get_evaluators_by_name("diabetes_eval")

        assert len(results) == 2
        # Should be ordered by version DESC
        assert results[0].version == 2
        assert results[1].version == 1

    def test_get_evaluators_by_name_not_exists(self, evaluator_service):
        """Test getting evaluators for nonexistent name."""
        results = evaluator_service.get_evaluators_by_name("nonexistent")
        assert len(results) == 0

    def test_get_evaluators_by_model(self, evaluator_service, sample_evaluators):
        """Test retrieving evaluators by model ID."""
        for evaluator in sample_evaluators:
            evaluator_service.create_evaluator(evaluator)

        results = evaluator_service.get_evaluators_by_model("trainer-123-v1")

        assert len(results) == 1
        # Should belong to the specific model version
        assert results[0].model_id == "trainer-123-v1"

    def test_list_all_evaluators(self, evaluator_service, sample_evaluators):
        """Test listing all evaluators."""
        for evaluator in sample_evaluators:
            evaluator_service.create_evaluator(evaluator)

        results = evaluator_service.list_all_evaluators()

        assert len(results) == 3
        # Should contain all evaluator names
        names = {r.name for r in results}
        assert names == {"diabetes_eval", "iris_eval"}


class TestEvaluatorServiceUpdate:
    """Test evaluator update operations."""

    def test_update_evaluator(self, evaluator_service, sample_evaluator):
        """Test updating an existing evaluator."""
        evaluator_service.create_evaluator(sample_evaluator)

        sample_evaluator.description = "Updated description"
        sample_evaluator.updated_at = datetime.now(UTC)

        evaluator_service.update_evaluator(sample_evaluator)

        result = evaluator_service.get_evaluator_by_id(sample_evaluator.id)
        assert result is not None
        assert result.description == "Updated description"


class TestEvaluatorServiceDeletion:
    """Test evaluator deletion operations."""

    def test_delete_evaluator(self, evaluator_service, sample_evaluator):
        """Test deleting an evaluator."""
        evaluator_service.create_evaluator(sample_evaluator)

        evaluator_service.delete_evaluator(sample_evaluator.id)

        result = evaluator_service.get_evaluator_by_id(sample_evaluator.id)
        assert result is None

    def test_delete_nonexistent_evaluator(self, evaluator_service):
        """Test deleting a nonexistent evaluator doesn't raise error."""
        # Should not raise an error
        evaluator_service.delete_evaluator("nonexistent")


class TestEvaluatorServiceExistence:
    """Test evaluator existence check operations."""

    def test_evaluator_exists(self, evaluator_service, sample_evaluator):
        """Test checking if an evaluator exists."""
        assert not evaluator_service.evaluator_exists(sample_evaluator.id)

        evaluator_service.create_evaluator(sample_evaluator)

        assert evaluator_service.evaluator_exists(sample_evaluator.id)

    def test_evaluator_name_version_exists(self, evaluator_service, sample_evaluator):
        """Test checking if an evaluator with specific name and version exists."""
        assert not evaluator_service.evaluator_name_version_exists(
            sample_evaluator.name, sample_evaluator.version
        )

        evaluator_service.create_evaluator(sample_evaluator)

        assert evaluator_service.evaluator_name_version_exists(
            sample_evaluator.name, sample_evaluator.version
        )


class TestEvaluatorServiceVersioning:
    """Test evaluator versioning operations."""

    def test_get_next_version_for_name_empty(self, evaluator_service):
        """Test getting next version when no evaluators exist."""
        next_version = evaluator_service.get_next_version_for_name("new_eval")
        assert next_version == 1

    def test_get_next_version_for_name_existing(
        self, evaluator_service, sample_evaluators
    ):
        """Test getting next version when evaluators exist."""
        for evaluator in sample_evaluators:
            evaluator_service.create_evaluator(evaluator)

        next_version = evaluator_service.get_next_version_for_name("diabetes_eval")
        assert next_version == 3  # Max version is 2, so next is 3
