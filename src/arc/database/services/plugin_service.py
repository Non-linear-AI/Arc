"""Plugin service for managing Arc plugin system."""

from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any

from ..base import DatabaseError
from .base import BaseService


@dataclass
class PluginMetadata:
    """Data class representing a plugin in the Arc system.

    Mirrors the C++ PluginServiceMetadata struct with exact field mapping.
    """

    name: str
    version: str
    description: str
    created_at: datetime
    updated_at: datetime


@dataclass
class ComponentMetadata:
    """Data class representing a plugin component in the Arc system.

    Mirrors the C++ ComponentServiceMetadata struct with exact field mapping.
    """

    plugin_name: str
    component_name: str
    component_spec: str
    description: str


class PluginService(BaseService):
    """Service for managing plugin system in the system database.

    Handles operations on plugin-related tables including:
    - Plugin registration and versioning
    - Component specification management
    - Plugin schema validation
    - CRUD operations with proper SQL escaping
    """

    def __init__(self, db_manager):
        """Initialize PluginService.

        Args:
            db_manager: DatabaseManager instance
        """
        super().__init__(db_manager)

    def register_plugin(self, plugin: PluginMetadata) -> None:
        """Register a new plugin in the database.

        Args:
            plugin: PluginMetadata object to register

        Raises:
            DatabaseError: If plugin registration fails
        """
        try:
            sql = self._build_plugin_insert_sql(plugin)
            self._system_execute(sql)
        except Exception as e:
            raise DatabaseError(
                f"Failed to register plugin {plugin.name} v{plugin.version}: {e}"
            ) from e

    def unregister_plugin(self, name: str, version: str) -> None:
        """Unregister a plugin and all its components.

        Args:
            name: Plugin name
            version: Plugin version

        Raises:
            DatabaseError: If plugin unregistration fails
        """
        try:
            # First get the plugin ID to delete components
            plugin = self.get_plugin(name, version)
            if plugin is None:
                return  # Plugin doesn't exist, nothing to unregister

            # Get plugin ID for component deletion
            plugin_id = self._get_plugin_id(name, version)
            if plugin_id is not None:
                # Delete all components for this plugin
                comp_sql = f"DELETE FROM plugin_components WHERE plugin_id = {plugin_id}"
                self._system_execute(comp_sql)

            # Delete the plugin itself
            plugin_sql = (
                f"DELETE FROM plugins WHERE name = '{self._escape_string(name)}' "
                f"AND version = '{self._escape_string(version)}'"
            )
            self._system_execute(plugin_sql)
        except Exception as e:
            raise DatabaseError(f"Failed to unregister plugin {name} v{version}: {e}") from e

    def list_plugins(self) -> list[PluginMetadata]:
        """List all registered plugins.

        Returns:
            List of PluginMetadata objects ordered by name, version

        Raises:
            DatabaseError: If query execution fails
        """
        try:
            sql = "SELECT * FROM plugins ORDER BY name, version"
            result = self._system_query(sql)
            return self._results_to_plugins(result)
        except Exception as e:
            raise DatabaseError(f"Failed to list plugins: {e}") from e

    def get_plugin(self, name: str, version: str) -> PluginMetadata | None:
        """Get a specific plugin by name and version.

        Args:
            name: Plugin name
            version: Plugin version

        Returns:
            PluginMetadata object if found, None otherwise

        Raises:
            DatabaseError: If query execution fails
        """
        try:
            escaped_name = self._escape_string(name)
            escaped_version = self._escape_string(version)
            sql = (
                f"SELECT * FROM plugins WHERE name = '{escaped_name}' "
                f"AND version = '{escaped_version}'"
            )
            result = self._system_query(sql)
            if result.empty():
                return None
            return self._result_to_plugin(result.first())
        except Exception as e:
            raise DatabaseError(f"Failed to get plugin {name} v{version}: {e}") from e

    def register_component(self, component: ComponentMetadata) -> None:
        """Register a new component for a plugin.

        Args:
            component: ComponentMetadata object to register

        Raises:
            DatabaseError: If component registration fails
        """
        try:
            # First, get the plugin ID for the component's plugin
            plugin_id = self._get_plugin_id_for_component(component.plugin_name)
            if plugin_id is None:
                raise DatabaseError(
                    f"Plugin {component.plugin_name} not found for component registration"
                )

            sql = self._build_component_insert_sql(component, plugin_id)
            self._system_execute(sql)
        except Exception as e:
            msg = f"Failed to register component {component.component_name}: {e}"
            raise DatabaseError(msg) from e

    def unregister_components_for_plugin(self, plugin_name: str, version: str) -> None:
        """Unregister all components for a specific plugin version.

        Args:
            plugin_name: Plugin name
            version: Plugin version

        Raises:
            DatabaseError: If component unregistration fails
        """
        try:
            plugin_id = self._get_plugin_id(plugin_name, version)
            if plugin_id is None:
                return  # Plugin doesn't exist, no components to remove

            sql = f"DELETE FROM plugin_components WHERE plugin_id = {plugin_id}"
            self._system_execute(sql)
        except Exception as e:
            msg = f"Failed to unregister components for {plugin_name} v{version}: {e}"
            raise DatabaseError(msg) from e

    def list_components(self) -> list[ComponentMetadata]:
        """List all registered components.

        Returns:
            List of ComponentMetadata objects ordered by plugin_name, component_name

        Raises:
            DatabaseError: If query execution fails
        """
        try:
            sql = """
                SELECT p.name as plugin_name, pc.component_name,
                       pc.component_spec, pc.description
                FROM plugin_components pc
                JOIN plugins p ON pc.plugin_id = p.id
                ORDER BY p.name, pc.component_name
            """
            result = self._system_query(sql)
            return self._results_to_components(result)
        except Exception as e:
            raise DatabaseError(f"Failed to list components: {e}") from e

    def get_components_for_plugin(self, plugin_name: str) -> list[ComponentMetadata]:
        """Get all components for a specific plugin (across all versions).

        Args:
            plugin_name: Plugin name

        Returns:
            List of ComponentMetadata objects for the plugin

        Raises:
            DatabaseError: If query execution fails
        """
        try:
            escaped_name = self._escape_string(plugin_name)
            sql = f"""
                SELECT p.name as plugin_name, pc.component_name,
                       pc.component_spec, pc.description
                FROM plugin_components pc
                JOIN plugins p ON pc.plugin_id = p.id
                WHERE p.name = '{escaped_name}'
                ORDER BY pc.component_name
            """
            result = self._system_query(sql)
            return self._results_to_components(result)
        except Exception as e:
            msg = f"Failed to get components for plugin {plugin_name}: {e}"
            raise DatabaseError(msg) from e

    def store_plugin_schema(
        self,
        algorithm_type: str,
        version: str,
        schema_json: str,
        description: str,
        author: str,
    ) -> None:
        """Store a plugin schema in the database.

        Args:
            algorithm_type: Type of algorithm
            version: Schema version
            schema_json: JSON schema specification
            description: Schema description
            author: Schema author

        Raises:
            DatabaseError: If schema storage fails
        """
        try:
            escaped_type = self._escape_string(algorithm_type)
            escaped_version = self._escape_string(version)
            escaped_schema = self._escape_string(schema_json)
            escaped_desc = self._escape_string(description)
            escaped_author = self._escape_string(author)

            sql = f"""
                INSERT INTO plugin_schemas (
                    algorithm_type, version, schema_json, description, author
                ) VALUES (
                    '{escaped_type}', '{escaped_version}', '{escaped_schema}',
                    '{escaped_desc}', '{escaped_author}'
                )
            """
            self._system_execute(sql)
        except Exception as e:
            msg = f"Failed to store schema for {algorithm_type} v{version}: {e}"
            raise DatabaseError(msg) from e

    def get_plugin_schema(self, algorithm_type: str, version: str) -> str | None:
        """Get a plugin schema from the database.

        Args:
            algorithm_type: Type of algorithm
            version: Schema version

        Returns:
            JSON schema string if found, None otherwise

        Raises:
            DatabaseError: If query execution fails
        """
        try:
            escaped_type = self._escape_string(algorithm_type)
            escaped_version = self._escape_string(version)
            sql = (
                f"SELECT schema_json FROM plugin_schemas "
                f"WHERE algorithm_type = '{escaped_type}' "
                f"AND version = '{escaped_version}'"
            )
            result = self._system_query(sql)
            if result.empty():
                return None
            return result.first().get("schema_json")
        except Exception as e:
            msg = f"Failed to get schema for {algorithm_type} v{version}: {e}"
            raise DatabaseError(msg) from e

    def remove_plugin_schema(self, algorithm_type: str, version: str) -> None:
        """Remove a plugin schema from the database.

        Args:
            algorithm_type: Type of algorithm
            version: Schema version

        Raises:
            DatabaseError: If schema removal fails
        """
        try:
            escaped_type = self._escape_string(algorithm_type)
            escaped_version = self._escape_string(version)
            sql = (
                f"DELETE FROM plugin_schemas "
                f"WHERE algorithm_type = '{escaped_type}' "
                f"AND version = '{escaped_version}'"
            )
            self._system_execute(sql)
        except Exception as e:
            msg = f"Failed to remove schema for {algorithm_type} v{version}: {e}"
            raise DatabaseError(msg) from e

    def _get_plugin_id(self, name: str, version: str) -> int | None:
        """Get plugin ID for a specific plugin name and version.

        Args:
            name: Plugin name
            version: Plugin version

        Returns:
            Plugin ID if found, None otherwise

        Raises:
            DatabaseError: If query execution fails
        """
        try:
            escaped_name = self._escape_string(name)
            escaped_version = self._escape_string(version)
            sql = (
                f"SELECT id FROM plugins WHERE name = '{escaped_name}' "
                f"AND version = '{escaped_version}'"
            )
            result = self._system_query(sql)
            if result.empty():
                return None
            return result.first().get("id")
        except Exception as e:
            raise DatabaseError(f"Failed to get plugin ID for {name} v{version}: {e}") from e

    def _get_plugin_id_for_component(self, plugin_name: str) -> int | None:
        """Get plugin ID for component registration (latest version).

        Args:
            plugin_name: Plugin name

        Returns:
            Plugin ID of the latest version if found, None otherwise

        Raises:
            DatabaseError: If query execution fails
        """
        try:
            escaped_name = self._escape_string(plugin_name)
            sql = (
                f"SELECT id FROM plugins WHERE name = '{escaped_name}' "
                "ORDER BY version DESC LIMIT 1"
            )
            result = self._system_query(sql)
            if result.empty():
                return None
            return result.first().get("id")
        except Exception as e:
            msg = f"Failed to get plugin ID for component registration {plugin_name}: {e}"
            raise DatabaseError(msg) from e

    def _result_to_plugin(self, row: dict[str, Any]) -> PluginMetadata:
        """Convert a database row to a PluginMetadata object.

        Args:
            row: Database row as dictionary

        Returns:
            PluginMetadata object created from row data

        Raises:
            DatabaseError: If conversion fails
        """
        try:
            # Handle timestamp conversion
            created_at = row.get("created_at")
            updated_at = row.get("updated_at")

            if isinstance(created_at, str):
                created_at = datetime.fromisoformat(created_at)
            elif isinstance(created_at, datetime):
                # Database returns naive datetime, assume UTC
                if created_at.tzinfo is None:
                    created_at = created_at.replace(tzinfo=UTC)
            elif created_at is None:
                created_at = datetime.now(UTC)

            if isinstance(updated_at, str):
                updated_at = datetime.fromisoformat(updated_at)
            elif isinstance(updated_at, datetime):
                # Database returns naive datetime, assume UTC
                if updated_at.tzinfo is None:
                    updated_at = updated_at.replace(tzinfo=UTC)
            elif updated_at is None:
                updated_at = datetime.now(UTC)

            return PluginMetadata(
                name=str(row["name"]),
                version=str(row["version"]),
                description=str(row["description"]),
                created_at=created_at,
                updated_at=updated_at,
            )
        except (KeyError, ValueError, TypeError) as e:
            raise DatabaseError(f"Failed to convert row to PluginMetadata: {e}") from e

    def _results_to_plugins(self, result) -> list[PluginMetadata]:
        """Convert query results to list of PluginMetadata objects.

        Args:
            result: QueryResult object from database query

        Returns:
            List of PluginMetadata objects

        Raises:
            DatabaseError: If conversion fails
        """
        try:
            return [self._result_to_plugin(row) for row in result.rows]
        except Exception as e:
            raise DatabaseError(f"Failed to convert results to plugins: {e}") from e

    def _result_to_component(self, row: dict[str, Any]) -> ComponentMetadata:
        """Convert a database row to a ComponentMetadata object.

        Args:
            row: Database row as dictionary

        Returns:
            ComponentMetadata object created from row data

        Raises:
            DatabaseError: If conversion fails
        """
        try:
            return ComponentMetadata(
                plugin_name=str(row["plugin_name"]),
                component_name=str(row["component_name"]),
                component_spec=str(row["component_spec"]),
                description=str(row["description"]),
            )
        except (KeyError, ValueError, TypeError) as e:
            raise DatabaseError(f"Failed to convert row to ComponentMetadata: {e}") from e

    def _results_to_components(self, result) -> list[ComponentMetadata]:
        """Convert query results to list of ComponentMetadata objects.

        Args:
            result: QueryResult object from database query

        Returns:
            List of ComponentMetadata objects

        Raises:
            DatabaseError: If conversion fails
        """
        try:
            return [self._result_to_component(row) for row in result.rows]
        except Exception as e:
            raise DatabaseError(f"Failed to convert results to components: {e}") from e

    def _build_plugin_insert_sql(self, plugin: PluginMetadata) -> str:
        """Build INSERT SQL statement for a plugin.

        Args:
            plugin: PluginMetadata object to insert

        Returns:
            SQL INSERT statement string

        Raises:
            DatabaseError: If SQL building fails
        """
        try:
            # Format timestamps for SQL
            created_at_str = plugin.created_at.isoformat()
            updated_at_str = plugin.updated_at.isoformat()

            sql = f"""INSERT INTO plugins (
                name, version, description, created_at, updated_at
            ) VALUES (
                '{self._escape_string(plugin.name)}',
                '{self._escape_string(plugin.version)}',
                '{self._escape_string(plugin.description)}',
                '{created_at_str}',
                '{updated_at_str}'
            )"""

            return sql
        except Exception as e:
            raise DatabaseError(f"Failed to build plugin insert SQL: {e}") from e

    def _build_component_insert_sql(self, component: ComponentMetadata, plugin_id: int) -> str:
        """Build INSERT SQL statement for a component.

        Args:
            component: ComponentMetadata object to insert
            plugin_id: ID of the parent plugin

        Returns:
            SQL INSERT statement string

        Raises:
            DatabaseError: If SQL building fails
        """
        try:
            sql = f"""INSERT INTO plugin_components (
                plugin_id, component_name, component_spec, description
            ) VALUES (
                {plugin_id},
                '{self._escape_string(component.component_name)}',
                '{self._escape_string(component.component_spec)}',
                '{self._escape_string(component.description)}'
            )"""

            return sql
        except Exception as e:
            raise DatabaseError(f"Failed to build component insert SQL: {e}") from e

    def _escape_string(self, value: str) -> str:
        """Escape string values for SQL to prevent injection.

        Args:
            value: String value to escape

        Returns:
            Escaped string safe for SQL

        Raises:
            DatabaseError: If escaping fails
        """
        if value is None:
            return ""

        try:
            # Basic SQL string escaping - replace single quotes with double quotes
            return str(value).replace("'", "''")
        except Exception as e:
            raise DatabaseError(f"Failed to escape string '{value}': {e}") from e
