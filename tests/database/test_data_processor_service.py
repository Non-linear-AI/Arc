"""Tests for DataProcessorService."""

from datetime import UTC, datetime

import pytest

from arc.database import DatabaseManager
from arc.database.models.data_processor import DataProcessor
from arc.database.services.data_processor_service import DataProcessorService


@pytest.fixture
def db_manager(tmp_path):
    """Create temporary file database manager for testing."""
    system_db = tmp_path / "system.db"
    user_db = tmp_path / "user.db"
    with DatabaseManager(str(system_db), str(user_db)) as manager:
        yield manager


@pytest.fixture
def data_processor_service(db_manager):
    """Create a DataProcessorService instance for testing."""
    return DataProcessorService(db_manager)


@pytest.fixture
def sample_data_processor():
    """Create a sample data processor for testing."""
    now = datetime.now(UTC)
    spec_yaml = (
        "name: customer_features\n"
        "description: Process customer data\n"
        "steps: []\n"
        "outputs: []"
    )
    return DataProcessor(
        id="customer-features-v1",
        name="customer_features",
        version=1,
        spec=spec_yaml,
        description="A test data processor for customer features",
        created_at=now,
        updated_at=now,
    )


class TestDataProcessorService:
    """Test DataProcessorService functionality."""

    def test_create_and_retrieve_data_processor(
        self, data_processor_service, sample_data_processor
    ):
        """Test creating and retrieving a data processor."""
        # Create the data processor
        data_processor_service.create_data_processor(sample_data_processor)

        # Retrieve by name and version
        retrieved = data_processor_service.get_data_processor_by_name_version(
            "customer_features", 1
        )

        assert retrieved is not None
        assert retrieved.id == "customer-features-v1"
        assert retrieved.name == "customer_features"
        assert retrieved.version == 1
        assert "customer_features" in retrieved.spec

    def test_get_latest_data_processor(self, data_processor_service):
        """Test retrieving the latest version of a data processor."""
        now = datetime.now(UTC)

        # Create multiple versions
        v1 = DataProcessor(
            id="sales-pipeline-v1",
            name="sales_pipeline",
            version=1,
            spec="name: sales_pipeline\ndescription: v1\nsteps: []\noutputs: []",
            description="Version 1",
            created_at=now,
            updated_at=now,
        )
        v2 = DataProcessor(
            id="sales-pipeline-v2",
            name="sales_pipeline",
            version=2,
            spec="name: sales_pipeline\ndescription: v2\nsteps: []\noutputs: []",
            description="Version 2",
            created_at=now,
            updated_at=now,
        )

        data_processor_service.create_data_processor(v1)
        data_processor_service.create_data_processor(v2)

        # Get latest
        latest = data_processor_service.get_latest_data_processor_by_name(
            "sales_pipeline"
        )

        assert latest is not None
        assert latest.version == 2
        assert latest.id == "sales-pipeline-v2"

    def test_get_next_version(self, data_processor_service):
        """Test calculating the next version number."""
        now = datetime.now(UTC)

        # No existing versions
        next_version = data_processor_service.get_next_version_for_name("new_processor")
        assert next_version == 1

        # Create a processor
        processor = DataProcessor(
            id="test-processor-v1",
            name="test_processor",
            version=1,
            spec="name: test_processor\ndescription: test\nsteps: []\noutputs: []",
            description="Test",
            created_at=now,
            updated_at=now,
        )
        data_processor_service.create_data_processor(processor)

        # Next version should be 2
        next_version = data_processor_service.get_next_version_for_name(
            "test_processor"
        )
        assert next_version == 2
