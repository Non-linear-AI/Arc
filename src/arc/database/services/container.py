"""Service container for centralized database service management."""

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ..manager import DatabaseManager

from .interactive_query_service import InteractiveQueryService
from .job_service import JobService
from .model_service import ModelService
from .plugin_service import PluginService
from .schema_service import SchemaService


class ServiceContainer:
    """Central container for all Arc database services.

    Provides centralized access to all database services with a single
    initialization point. Services are initialized lazily for better
    performance and resource management.
    """

    def __init__(self, db_manager: "DatabaseManager"):
        """Initialize ServiceContainer with database manager.

        Args:
            db_manager: DatabaseManager instance for database access
        """
        self.db_manager = db_manager

        # Services are initialized lazily via properties
        self._query_service = None
        self._model_service = None
        self._job_service = None
        self._plugin_service = None
        self._schema_service = None

    @property
    def query(self) -> InteractiveQueryService:
        """Get the interactive query service."""
        if self._query_service is None:
            self._query_service = InteractiveQueryService(self.db_manager, self.schema)
        return self._query_service

    @property
    def models(self) -> ModelService:
        """Get the model service."""
        if self._model_service is None:
            self._model_service = ModelService(self.db_manager)
        return self._model_service

    @property
    def jobs(self) -> JobService:
        """Get the job service."""
        if self._job_service is None:
            self._job_service = JobService(self.db_manager)
        return self._job_service

    @property
    def plugins(self) -> PluginService:
        """Get the plugin service."""
        if self._plugin_service is None:
            self._plugin_service = PluginService(self.db_manager)
        return self._plugin_service

    @property
    def schema(self) -> SchemaService:
        """Get the schema discovery service."""
        if self._schema_service is None:
            self._schema_service = SchemaService(self.db_manager)
        return self._schema_service

    def close(self) -> None:
        """Clean up resources and close database connections."""
        self.db_manager.close()

    def __enter__(self) -> "ServiceContainer":
        """Context manager entry."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        """Context manager exit - ensure connections are closed."""
        self.close()

    def __repr__(self) -> str:
        """String representation for debugging."""
        return f"ServiceContainer(db_manager={self.db_manager})"
