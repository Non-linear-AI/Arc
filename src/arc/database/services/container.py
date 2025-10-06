"""Service container for centralized database service management."""

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from arc.database.manager import DatabaseManager
    from arc.ml.runtime import MLRuntime

from arc.database.services.interactive_query_service import InteractiveQueryService
from arc.database.services.job_service import JobService
from arc.database.services.ml_data_service import MLDataService
from arc.database.services.ml_plan_service import MLPlanService
from arc.database.services.model_service import ModelService
from arc.database.services.plugin_service import PluginService
from arc.database.services.schema_service import SchemaService
from arc.database.services.trainer_service import TrainerService


class ServiceContainer:
    """Central container for all Arc database services.

    Provides centralized access to all database services with a single
    initialization point. Services are initialized lazily for better
    performance and resource management.
    """

    def __init__(self, db_manager: "DatabaseManager", artifacts_dir: str | None = None):
        """Initialize ServiceContainer with database manager.

        Args:
            db_manager: DatabaseManager instance for database access
            artifacts_dir: Directory for ML artifacts storage
        """
        self.db_manager = db_manager
        self.artifacts_dir = artifacts_dir

        # Services are initialized lazily via properties
        self._query_service = None
        self._model_service = None
        self._trainer_service = None
        self._job_service = None
        self._plugin_service = None
        self._schema_service = None
        self._ml_data_service = None
        self._ml_plan_service = None
        self._ml_runtime = None

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
    def trainers(self) -> TrainerService:
        """Get the trainer service."""
        if self._trainer_service is None:
            self._trainer_service = TrainerService(self.db_manager)
        return self._trainer_service

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

    @property
    def ml_data(self) -> MLDataService:
        """Get the ML data service."""
        if self._ml_data_service is None:
            self._ml_data_service = MLDataService(self.db_manager)
        return self._ml_data_service

    @property
    def ml_plans(self) -> MLPlanService:
        """Get the ML plan service."""
        if self._ml_plan_service is None:
            self._ml_plan_service = MLPlanService(self.db_manager.get_system_db())
        return self._ml_plan_service

    @property
    def ml_runtime(self) -> "MLRuntime":
        """Get the ML runtime service."""
        if self._ml_runtime is None:
            from arc.ml.runtime import MLRuntime

            self._ml_runtime = MLRuntime(self, self.artifacts_dir)
        return self._ml_runtime

    def shutdown(self) -> None:
        """Shutdown all services and clean up resources."""
        # Shutdown ML runtime if it was initialized
        if self._ml_runtime is not None:
            self._ml_runtime.shutdown()
        self.close()

    def close(self) -> None:
        """Clean up resources and close database connections."""
        self.db_manager.close()

    def __enter__(self) -> "ServiceContainer":
        """Context manager entry."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        """Context manager exit - ensure all resources are cleaned up."""
        self.shutdown()

    def __repr__(self) -> str:
        """String representation for debugging."""
        return f"ServiceContainer(db_manager={self.db_manager})"
