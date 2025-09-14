"""Basic tests for PluginService."""

from datetime import UTC, datetime

import pytest

from arc.database import DatabaseManager
from arc.database.services.plugin_service import (
    ComponentMetadata,
    PluginMetadata,
    PluginService,
)


@pytest.fixture
def db_manager():
    """Create an in-memory database manager for testing."""
    with DatabaseManager(":memory:") as manager:
        yield manager


@pytest.fixture
def plugin_service(db_manager):
    """Create a PluginService instance for testing."""
    return PluginService(db_manager)


@pytest.fixture
def sample_plugin():
    """Create a sample plugin for testing."""
    now = datetime.now(UTC)
    return PluginMetadata(
        name="test-plugin",
        version="1.0.0",
        description="A test plugin",
        created_at=now,
        updated_at=now,
    )


@pytest.fixture
def sample_component():
    """Create a sample component for testing."""
    return ComponentMetadata(
        plugin_name="test-plugin",
        component_name="test-component",
        component_spec="test-spec",
        description="A test component",
    )


class TestPluginServiceBasic:
    """Basic tests for PluginService."""

    def test_register_and_get_plugin(self, plugin_service, sample_plugin):
        """Test registering and retrieving a plugin."""
        # Register plugin
        plugin_service.register_plugin(sample_plugin)

        # Get plugin
        retrieved = plugin_service.get_plugin(sample_plugin.name, sample_plugin.version)
        assert retrieved is not None
        assert retrieved.name == sample_plugin.name
        assert retrieved.version == sample_plugin.version
        assert retrieved.description == sample_plugin.description

    def test_list_plugins(self, plugin_service, sample_plugin):
        """Test listing plugins."""
        # Initially empty
        plugins = plugin_service.list_plugins()
        assert len(plugins) == 0

        # Register plugin
        plugin_service.register_plugin(sample_plugin)

        # List plugins
        plugins = plugin_service.list_plugins()
        assert len(plugins) == 1
        assert plugins[0].name == sample_plugin.name

    def test_unregister_plugin(self, plugin_service, sample_plugin):
        """Test unregistering a plugin."""
        # Register plugin
        plugin_service.register_plugin(sample_plugin)

        # Verify it exists
        assert plugin_service.get_plugin(sample_plugin.name, sample_plugin.version) is not None

        # Unregister plugin
        plugin_service.unregister_plugin(sample_plugin.name, sample_plugin.version)

        # Verify it's gone
        assert plugin_service.get_plugin(sample_plugin.name, sample_plugin.version) is None

    def test_register_and_list_components(self, plugin_service, sample_plugin, sample_component):
        """Test registering and listing components."""
        # Register plugin first
        plugin_service.register_plugin(sample_plugin)

        # Register component
        plugin_service.register_component(sample_component)

        # List all components
        components = plugin_service.list_components()
        assert len(components) == 1
        assert components[0].plugin_name == sample_component.plugin_name
        assert components[0].component_name == sample_component.component_name

        # Get components for plugin
        plugin_components = plugin_service.get_components_for_plugin(sample_plugin.name)
        assert len(plugin_components) == 1
        assert plugin_components[0].component_name == sample_component.component_name

    def test_schema_operations(self, plugin_service):
        """Test schema storage and retrieval."""
        algorithm_type = "test-algorithm"
        version = "1.0"
        schema_json = '{"type": "object", "properties": {}}'
        description = "Test schema"
        author = "Test Author"

        # Store schema
        plugin_service.store_plugin_schema(
            algorithm_type, version, schema_json, description, author
        )

        # Get schema
        retrieved = plugin_service.get_plugin_schema(algorithm_type, version)
        assert retrieved == schema_json

        # Remove schema
        plugin_service.remove_plugin_schema(algorithm_type, version)

        # Verify it's gone
        assert plugin_service.get_plugin_schema(algorithm_type, version) is None

    def test_get_nonexistent_plugin(self, plugin_service):
        """Test getting a non-existent plugin returns None."""
        result = plugin_service.get_plugin("nonexistent", "1.0")
        assert result is None

    def test_unregister_nonexistent_plugin(self, plugin_service):
        """Test unregistering a non-existent plugin doesn't error."""
        # Should not raise an exception
        plugin_service.unregister_plugin("nonexistent", "1.0")
