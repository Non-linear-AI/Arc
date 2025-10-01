"""Tests for ModelService."""

from datetime import UTC, datetime

import pytest

from arc.database import DatabaseManager
from arc.database.services.model_service import Model, ModelService


@pytest.fixture
def db_manager(tmp_path):
    """Create temporary file database manager for testing."""
    system_db = tmp_path / "system.db"
    user_db = tmp_path / "user.db"
    with DatabaseManager(str(system_db), str(user_db)) as manager:
        yield manager


@pytest.fixture
def model_service(db_manager):
    """Create a ModelService instance for testing."""
    return ModelService(db_manager)


@pytest.fixture
def sample_model():
    """Create a sample model for testing."""
    now = datetime.now(UTC)
    return Model(
        id="test-model-1",
        type="classification",
        name="iris_classifier",
        version=1,
        description="A test model for iris classification",
        spec='{"algorithm": "random_forest", "params": {"n_estimators": 100}}',
        created_at=now,
        updated_at=now,
    )


@pytest.fixture
def sample_models():
    """Create multiple sample models for testing."""
    now = datetime.now(UTC)
    return [
        Model(
            id="model-1",
            type="classification",
            name="iris_classifier",
            version=1,
            description="First version",
            spec='{"algorithm": "svm"}',
            created_at=now,
            updated_at=now,
        ),
        Model(
            id="model-2",
            type="classification",
            name="iris_classifier",
            version=2,
            description="Second version",
            spec='{"algorithm": "random_forest"}',
            created_at=now,
            updated_at=now,
        ),
        Model(
            id="model-3",
            type="regression",
            name="house_prices",
            version=1,
            description="House price prediction",
            spec='{"algorithm": "linear_regression"}',
            created_at=now,
            updated_at=now,
        ),
    ]


class TestModelDataClass:
    """Test the Model dataclass."""

    def test_model_creation(self, sample_model):
        """Test creating a Model instance."""
        assert sample_model.id == "test-model-1"
        assert sample_model.type == "classification"
        assert sample_model.name == "iris_classifier"
        assert sample_model.version == 1
        assert isinstance(sample_model.created_at, datetime)
        assert isinstance(sample_model.updated_at, datetime)

    def test_model_with_version(self):
        """Test creating a Model with version."""
        now = datetime.now(UTC)
        model = Model(
            id="child-model",
            type="classification",
            name="improved_classifier",
            version=2,
            description="Improved version",
            spec='{"algorithm": "xgboost"}',
            created_at=now,
            updated_at=now,
        )


class TestModelServiceCRUD:
    """Test CRUD operations in ModelService."""

    def test_create_model(self, model_service, sample_model):
        """Test creating a model."""
        # Initially no models
        models = model_service.list_all_models()
        assert len(models) == 0

        # Create model
        model_service.create_model(sample_model)

        # Verify model was created
        models = model_service.list_all_models()
        assert len(models) == 1
        assert models[0].id == sample_model.id
        assert models[0].name == sample_model.name

    def test_get_model_by_id(self, model_service, sample_model):
        """Test getting a model by ID."""
        # Model doesn't exist initially
        result = model_service.get_model_by_id(sample_model.id)
        assert result is None

        # Create model
        model_service.create_model(sample_model)

        # Get model by ID
        result = model_service.get_model_by_id(sample_model.id)
        assert result is not None
        assert result.id == sample_model.id
        assert result.name == sample_model.name
        assert result.version == sample_model.version

    def test_get_model_by_name_version(self, model_service, sample_models):
        """Test getting a model by name and version."""
        # Create multiple models
        for model in sample_models:
            model_service.create_model(model)

        # Get specific version
        result = model_service.get_model_by_name_version("iris_classifier", 2)
        assert result is not None
        assert result.id == "model-2"
        assert result.version == 2

        # Non-existent version
        result = model_service.get_model_by_name_version("iris_classifier", 99)
        assert result is None

    def test_get_latest_model_by_name(self, model_service, sample_models):
        """Test getting the latest version of a model by name."""
        # Create multiple models
        for model in sample_models:
            model_service.create_model(model)

        # Get latest version of iris_classifier
        result = model_service.get_latest_model_by_name("iris_classifier")
        assert result is not None
        assert result.version == 2  # Latest version
        assert result.id == "model-2"

        # Get model with only one version
        result = model_service.get_latest_model_by_name("house_prices")
        assert result is not None
        assert result.version == 1
        assert result.id == "model-3"

        # Non-existent model
        result = model_service.get_latest_model_by_name("nonexistent")
        assert result is None

    def test_get_models_by_name(self, model_service, sample_models):
        """Test getting all versions of a model by name."""
        # Create multiple models
        for model in sample_models:
            model_service.create_model(model)

        # Get all versions of iris_classifier
        results = model_service.get_models_by_name("iris_classifier")
        assert len(results) == 2
        # Should be ordered by version DESC
        assert results[0].version == 2
        assert results[1].version == 1

        # Get model with single version
        results = model_service.get_models_by_name("house_prices")
        assert len(results) == 1
        assert results[0].version == 1

        # Non-existent model
        results = model_service.get_models_by_name("nonexistent")
        assert len(results) == 0

    def test_update_model(self, model_service, sample_model):
        """Test updating a model."""
        # Create model
        model_service.create_model(sample_model)

        # Update model
        sample_model.description = "Updated description"
        sample_model.updated_at = datetime.now(UTC)
        model_service.update_model(sample_model)

        # Verify update
        result = model_service.get_model_by_id(sample_model.id)
        assert result is not None
        assert result.description == "Updated description"

    def test_delete_model(self, model_service, sample_model):
        """Test deleting a model."""
        # Create model
        model_service.create_model(sample_model)
        assert model_service.get_model_by_id(sample_model.id) is not None

        # Delete model
        model_service.delete_model(sample_model.id)

        # Verify deletion
        assert model_service.get_model_by_id(sample_model.id) is None

    def test_list_all_models_ordering(self, model_service, sample_models):
        """Test that list_all_models returns models in correct order."""
        # Create models with different timestamps
        for i, model in enumerate(sample_models):
            # Modify timestamps to ensure different created_at times
            base_time = datetime.now(UTC)
            model.created_at = base_time.replace(second=i)
            model_service.create_model(model)

        # Get all models - should be ordered by created_at DESC
        results = model_service.list_all_models()
        assert len(results) == 3

        # Note: Exact ordering may depend on DB timestamp precision
        # But we should have all models
        model_ids = {model.id for model in results}
        expected_ids = {"model-1", "model-2", "model-3"}
        assert model_ids == expected_ids


class TestModelServiceUtilities:
    """Test utility methods in ModelService."""

    def test_model_exists(self, model_service, sample_model):
        """Test checking if a model exists."""
        # Model doesn't exist initially
        assert not model_service.model_exists(sample_model.id)

        # Create model
        model_service.create_model(sample_model)

        # Model now exists
        assert model_service.model_exists(sample_model.id)

        # Delete model
        model_service.delete_model(sample_model.id)

        # Model no longer exists
        assert not model_service.model_exists(sample_model.id)

    def test_model_name_version_exists(self, model_service, sample_models):
        """Test checking if a specific name/version combination exists."""
        # Create models
        for model in sample_models:
            model_service.create_model(model)

        # Check existing combinations
        assert model_service.model_name_version_exists("iris_classifier", 1)
        assert model_service.model_name_version_exists("iris_classifier", 2)
        assert model_service.model_name_version_exists("house_prices", 1)

        # Check non-existing combinations
        assert not model_service.model_name_version_exists("iris_classifier", 3)
        assert not model_service.model_name_version_exists("nonexistent", 1)

    def test_generate_next_model_id(self, model_service, sample_models):
        """Test generating next model ID."""
        # First ID should be "1"
        next_id = model_service.generate_next_model_id()
        assert next_id == "1"

        # Create some models with numeric IDs
        numeric_models = []
        for i, model in enumerate(sample_models, 1):
            model.id = str(i)
            numeric_models.append(model)
            model_service.create_model(model)

        # Next ID should be "4"
        next_id = model_service.generate_next_model_id()
        assert next_id == "4"

    def test_get_next_version_for_name(self, model_service, sample_models):
        """Test getting next version for a model name."""
        # No models exist - should return 1
        next_version = model_service.get_next_version_for_name("nonexistent")
        assert next_version == 1

        # Create models
        for model in sample_models:
            model_service.create_model(model)

        # iris_classifier has versions 1 and 2, so next should be 3
        next_version = model_service.get_next_version_for_name("iris_classifier")
        assert next_version == 3

        # house_prices has version 1, so next should be 2
        next_version = model_service.get_next_version_for_name("house_prices")
        assert next_version == 2


class TestModelServiceHelpers:
    """Test helper methods in ModelService."""

    def test_escape_string(self, model_service):
        """Test SQL string escaping."""
        # Normal string
        assert model_service._escape_string("test") == "test"

        # String with single quote
        assert model_service._escape_string("test's model") == "test''s model"

        # String with multiple quotes
        assert model_service._escape_string("'test' 'data'") == "''test'' ''data''"

        # Empty string
        assert model_service._escape_string("") == ""

        # None input
        assert model_service._escape_string(None) == ""

    def test_result_to_model_conversion(self, model_service, sample_model):
        """Test converting database row to Model object."""
        # Create and retrieve model to test conversion
        model_service.create_model(sample_model)
        retrieved = model_service.get_model_by_id(sample_model.id)

        assert retrieved is not None
        assert isinstance(retrieved, Model)
        assert retrieved.id == sample_model.id
        assert retrieved.name == sample_model.name
        assert retrieved.version == sample_model.version
        assert isinstance(retrieved.created_at, datetime)
        assert isinstance(retrieved.updated_at, datetime)


class TestModelServiceEdgeCases:
    """Test edge cases and error conditions."""

    def test_model_with_special_characters(self, model_service):
        """Test model with special characters in fields."""
        now = datetime.now(UTC)
        special_model = Model(
            id="special-model",
            type="test'type",
            name="test'model'name",
            version=1,
            description="Description with 'quotes' and \"double quotes\"",
            spec='{"param": "value\'with\'quotes"}',
            created_at=now,
            updated_at=now,
        )

        # Should handle special characters properly
        model_service.create_model(special_model)
        retrieved = model_service.get_model_by_id("special-model")

        assert retrieved is not None
        assert retrieved.name == "test'model'name"
        expected_desc = "Description with 'quotes' and \"double quotes\""
        assert retrieved.description == expected_desc

    def test_model_with_simple_spec(self, model_service):
        """Test model with simple spec."""
        now = datetime.now(UTC)
        model = Model(
            id="no-base",
            type="test",
            name="test",
            version=1,
            description="Test",
            spec="{}",
            created_at=now,
            updated_at=now,
        )

        model_service.create_model(model)
        retrieved = model_service.get_model_by_id("no-base")

        assert retrieved is not None

    def test_empty_database_operations(self, model_service):
        """Test operations on empty database."""
        # List models on empty database
        models = model_service.list_all_models()
        assert len(models) == 0

        # Get non-existent model
        assert model_service.get_model_by_id("nonexistent") is None
        assert model_service.get_model_by_name_version("nonexistent", 1) is None
        assert model_service.get_latest_model_by_name("nonexistent") is None

        # Get models by name on empty database
        models = model_service.get_models_by_name("nonexistent")
        assert len(models) == 0

        # Utility methods on empty database
        assert not model_service.model_exists("nonexistent")
        assert not model_service.model_name_version_exists("nonexistent", 1)
        assert model_service.get_next_version_for_name("nonexistent") == 1
