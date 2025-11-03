"""Reusable runtime helpers for ML operations."""

from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING

import torch

from arc.database import DatabaseError
from arc.database.models.model import Model
from arc.graph.model import ModelSpec
from arc.ml.artifacts import ModelArtifactManager
from arc.ml.predictor import ArcPredictor
from arc.ml.training_service import TrainingJobConfig, TrainingService

MODEL_ID_PATTERN = re.compile(r"^(?P<base>.+?)-v(?P<version>\d+)$")

if TYPE_CHECKING:  # pragma: no cover - import heavy modules only for typing
    from arc.database.services.container import ServiceContainer


class MLRuntimeError(Exception):
    """Raised when ML runtime operations fail."""

    def __init__(self, message: str, validation_report: dict | None = None):
        """Initialize MLRuntimeError.

        Args:
            message: Error message
            validation_report: Optional validation report dict for agent debugging
        """
        super().__init__(message)
        self.validation_report = validation_report


@dataclass
class PredictionSummary:
    """Summary information about a prediction run."""

    total_predictions: int
    outputs: list[str]
    saved_table: str | None = None


class MLRuntime:
    """Runtime utilities for ML operations shared across CLI and tools."""

    def __init__(self, services: ServiceContainer, artifacts_dir: Path | None = None):
        self.services = services
        self.model_service = services.models
        self.job_service = services.jobs
        self.ml_data_service = services.ml_data

        self.artifacts_root = Path(artifacts_dir or ".arc/artifacts")
        self.artifacts_root.mkdir(parents=True, exist_ok=True)

        self.artifact_manager = ModelArtifactManager(self.artifacts_root)
        self.training_service = TrainingService(
            self.job_service,
            artifacts_dir=self.artifacts_root,
            tracking_service=services.training_tracking,
        )

    def shutdown(self) -> None:
        """Ensure background resources are cleaned up."""
        self.training_service.shutdown()

    def create_model(
        self,
        *,
        name: str,
        schema_path: Path,
        description: str | None = None,
        model_type: str | None = None,
    ) -> Model:
        """Register a new Arc-Graph backed model."""
        _validate_model_name(name)

        schema_file = schema_path.expanduser()
        if not schema_file.exists():
            raise MLRuntimeError(f"Schema file not found: {schema_file}")

        try:
            schema_text = schema_file.read_text(encoding="utf-8")
        except OSError as exc:
            raise MLRuntimeError(f"Failed to read schema file: {exc}") from exc

        try:
            model_spec = ModelSpec.from_yaml(schema_text)
            # Basic validation - ensure model spec is valid
            _ = model_spec.get_input_names()
            _ = model_spec.get_output_names()
        except Exception as exc:  # noqa: BLE001 - propagate as runtime error
            raise MLRuntimeError(f"Invalid model schema: {exc}") from exc

        # Generate ID using validated name directly (no slugification needed)
        # Names are already restricted to [a-zA-Z0-9_-] by validation
        version = self.model_service.get_next_version_for_id_prefix(name)
        model_id = f"{name}-v{version}"

        now = datetime.now(UTC)
        model = Model(
            id=model_id,
            type=model_type or "ml.model_spec",
            name=name,
            version=version,
            description=description or "",
            spec=schema_text,
            created_at=now,
            updated_at=now,
        )

        try:
            self.model_service.create_model(model)
        except DatabaseError as exc:
            raise MLRuntimeError(f"Failed to register model: {exc}") from exc

        return model

    def register_data_processor(
        self,
        name: str,
        spec,  # DataSourceSpec
        description: str | None = None,
    ):  # -> DataProcessor
        """Register a data processor in the database.

        Always creates a new version for each registration to maintain
        full history of all generation attempts.

        Handles race conditions by retrying with updated version on
        duplicate key errors.

        Args:
            name: Data processor name
            spec: DataSourceSpec object (already parsed and validated)
            description: Optional description

        Returns:
            DataProcessor object with generated id and version

        Raises:
            MLRuntimeError: If registration fails after retries
        """
        from arc.database.models.data_processor import DataProcessor

        _validate_model_name(name)

        # Convert spec to YAML string for storage
        spec_yaml = spec.to_yaml()

        # Retry logic to handle race conditions where multiple processes
        # try to insert the same version number simultaneously
        max_retries = 5
        import logging

        logger = logging.getLogger(__name__)

        for attempt in range(max_retries):
            # Get next version using validated name directly (no slugification needed)
            # Names are already restricted to [a-zA-Z0-9_-] by validation
            next_version = self.services.data_processors.get_next_version_for_id_prefix(
                name
            )
            processor_id = f"{name}-v{next_version}"

            if attempt > 0:
                logger.debug(
                    f"Retry attempt {attempt + 1}/{max_retries} for data processor "
                    f"'{name}' with version {next_version}"
                )

            # Create DataProcessor
            now = datetime.now(UTC)
            processor = DataProcessor(
                id=processor_id,
                name=name,
                version=next_version,
                spec=spec_yaml,
                description=description or spec.description,
                created_at=now,
                updated_at=now,
            )

            # Try to save to database
            try:
                self.services.data_processors.create_data_processor(processor)
                # Success - return the processor
                if attempt > 0:
                    logger.info(
                        f"Successfully registered data processor '{name}' as {processor_id} "
                        f"after {attempt + 1} attempts"
                    )
                return processor
            except DatabaseError as exc:
                # Check if this is a duplicate key error
                error_msg = str(exc).lower()
                is_duplicate = (
                    "duplicate" in error_msg
                    or "unique constraint" in error_msg
                    or "primary key" in error_msg
                )

                if is_duplicate and attempt < max_retries - 1:
                    # Log the duplicate key error for debugging
                    logger.debug(
                        f"Duplicate key error for {processor_id}, will retry "
                        f"(attempt {attempt + 1}/{max_retries})"
                    )
                    # Retry with fresh version number
                    # Small delay to reduce contention
                    import time

                    time.sleep(0.01 * (attempt + 1))
                    continue
                else:
                    # Not a duplicate error or max retries exceeded
                    if is_duplicate:
                        raise MLRuntimeError(
                            f"Failed to register data processor '{name}': "
                            f"Version conflict persisted after {max_retries} retries. "
                            f"Last attempted version was {next_version}. "
                            f"This may indicate a database issue or high concurrency."
                        ) from exc
                    else:
                        raise MLRuntimeError(
                            f"Failed to register data processor: {exc}"
                        ) from exc

        # Should never reach here due to return/raise in loop
        raise MLRuntimeError(
            f"Failed to register data processor after {max_retries} attempts"
        )

    def load_data_processor(
        self,
        name: str,
        version: int | None = None,
    ):  # -> tuple[DataProcessor, DataSourceSpec]
        """Load data processor from database.

        Args:
            name: Data processor name
            version: Optional specific version (defaults to latest)

        Returns:
            Tuple of (DataProcessor, DataSourceSpec)

        Raises:
            MLRuntimeError: If data processor not found or spec parsing fails
        """
        from arc.graph.features.data_source import DataSourceSpec

        data_processor_service = self.services.data_processors

        # Fetch from database
        if version is not None:
            processor = data_processor_service.get_data_processor_by_name_version(
                name, version
            )
        else:
            processor = data_processor_service.get_latest_data_processor_by_name(name)

        if processor is None:
            version_str = f" version {version}" if version is not None else ""
            raise MLRuntimeError(f"Data processor '{name}'{version_str} not found")

        # Parse YAML to DataSourceSpec
        try:
            spec = DataSourceSpec.from_yaml(processor.spec)
        except Exception as exc:  # noqa: BLE001 - propagate as runtime error
            raise MLRuntimeError(f"Failed to parse data processor spec: {exc}") from exc

        return processor, spec

    def train_model(
        self,
        *,
        model_name: str,
        train_table: str,
        validation_table: str | None = None,
        validation_split: float | None = None,
        epochs: int | None = None,
        batch_size: int | None = None,
        learning_rate: float | None = None,
        checkpoint_dir: str | None = None,
        description: str | None = None,
        tags: list[str] | None = None,
        skip_validation: bool = False,
    ) -> str:
        """Submit a training job using a registered model with unified YAML spec.

        The model's YAML contains both architecture and training configuration.
        This method extracts the training config from the model spec and allows
        runtime parameter overrides.

        Args:
            model_name: Name of the registered model
            train_table: Training data table
            validation_table: Optional validation table
            validation_split: Optional override for validation split ratio
            epochs: Optional override for number of training epochs
            batch_size: Optional override for batch size
            learning_rate: Optional override for learning rate
            checkpoint_dir: Optional checkpoint directory
            description: Optional job description
            tags: Optional job tags
            skip_validation: Skip dry-run validation if already done (default: False)

        Returns:
            Job ID

        Raises:
            MLRuntimeError: If model not found or validation fails
        """
        _validate_model_name(model_name)

        train_table = str(train_table)
        validation_table = str(validation_table) if validation_table else None

        # Get registered model
        model_record = self.model_service.get_latest_model_by_name(model_name)
        if model_record is None:
            raise MLRuntimeError(
                f"Model '{model_name}' not found. Use /ml model first."
            )

        if not model_record.spec:
            raise MLRuntimeError("Stored model is missing specification; cannot train.")

        # Parse unified YAML to extract ModelSpec and training_config
        try:
            model_spec, training_config = self.model_service.parse_model_spec(
                model_record
            )
        except Exception as exc:  # noqa: BLE001
            raise MLRuntimeError(f"Failed to parse model spec: {exc}") from exc

        # Extract feature columns from model spec inputs
        feature_columns = []
        for input_spec in model_spec.inputs.values():
            if input_spec.columns:
                feature_columns.extend(input_spec.columns)

        if not feature_columns:
            raise MLRuntimeError(
                "Model spec must define input columns before training."
            )

        if not self.ml_data_service.dataset_exists(train_table):
            raise MLRuntimeError(
                f"Training table '{train_table}' does not exist in user DB"
            )

        # Extract target column from training config loss function
        loss_config = training_config.get("loss", {})
        if not loss_config or not loss_config.get("inputs"):
            raise MLRuntimeError(
                "Model spec must define a loss function with inputs "
                "to determine target column"
            )

        target_column = loss_config["inputs"].get("target")
        if not target_column:
            raise MLRuntimeError(
                "Model spec loss function must define 'target' input "
                "to specify target column"
            )

        columns_to_check = list(feature_columns)
        if target_column not in columns_to_check:
            columns_to_check.append(target_column)

        missing = [
            column
            for column, exists in self.ml_data_service.validate_columns(
                train_table, columns_to_check
            ).items()
            if not exists
        ]
        if missing:
            # Get actual columns in the table for comparison
            dataset_info = self.ml_data_service.get_dataset_info(train_table)
            actual_columns = dataset_info.column_names if dataset_info else []

            # Format missing columns for display
            missing_list = ", ".join(missing)
            actual_list = ", ".join(actual_columns[:10])  # Show first 10
            if len(actual_columns) > 10:
                actual_list += f", ... ({len(actual_columns) - 10} more)"

            raise MLRuntimeError(
                f"Model/data column mismatch for table '{train_table}'.\n\n"
                f"Model expects these columns (defined in model spec inputs):\n"
                f"  {missing_list}\n\n"
                f"Training table actually has:\n"
                f"  {actual_list}\n\n"
                f"Fix: Regenerate the model to match actual column names in "
                f"'{train_table}', or transform your data to include the "
                f"columns the model expects."
            )

        # Validate validation table if provided
        if validation_table and not self.ml_data_service.dataset_exists(
            validation_table
        ):
            raise MLRuntimeError(
                f"Validation table '{validation_table}' does not exist in user DB"
            )

        checkpoint_path: str | None = None
        if checkpoint_dir:
            checkpoint_path = str(Path(checkpoint_dir).expanduser())

        model_key, record_version = _split_model_identifier(model_record.id)
        if record_version is None:
            record_version = model_record.version

        # Apply runtime overrides or use training_config defaults
        final_validation_split = (
            validation_split
            if validation_split is not None
            else training_config.get("validation_split", 0.2)
        )
        final_epochs = (
            epochs if epochs is not None else training_config.get("epochs", 10)
        )
        final_batch_size = (
            batch_size
            if batch_size is not None
            else training_config.get("batch_size", 32)
        )
        final_learning_rate = (
            learning_rate
            if learning_rate is not None
            else training_config.get("learning_rate", 0.001)
        )

        # Create job config with training_config dict instead of trainer_spec
        job_config = TrainingJobConfig(
            model_id=model_key,
            model_version=record_version,
            model_name=model_record.name,
            train_table=train_table,
            target_column=target_column,
            model_spec=model_spec,
            training_config=training_config,
            feature_columns=list(feature_columns),
            validation_table=validation_table,
            validation_split=final_validation_split,
            epochs=final_epochs,
            batch_size=final_batch_size,
            learning_rate=final_learning_rate,
            artifacts_dir=str(self.artifacts_root),
            checkpoint_dir=checkpoint_path,
            description=description or f"Training {model_record.name}",
            tags=tags,
        )

        # Run validation synchronously BEFORE submitting job to catch errors
        # early. Validates model building, data loading, forward pass, loss.
        # Skip if validation was already done (e.g., during model registration)
        if not skip_validation:
            try:
                self.training_service.validate_job_config(job_config)
            except ValueError as validation_error:
                # Extract validation report if available
                from arc.ml.dry_run_validator import ValidationError

                validation_report = None
                if isinstance(validation_error, ValidationError):
                    validation_report = validation_error.validation_report.to_dict()

                # Convert validation errors to MLRuntimeError with report attached
                raise MLRuntimeError(
                    str(validation_error), validation_report=validation_report
                ) from validation_error

        job_id = self.training_service.submit_training_job(job_config)
        return job_id

    def predict(
        self,
        *,
        model_name: str,
        table_name: str,
        batch_size: int = 32,
        limit: int | None = None,
        output_table: str | None = None,
        device: str | None = None,
        streaming_prediction_threshold: int = 50000,
        streaming_prediction_chunk_size: int = 10000,
    ) -> PredictionSummary:
        """Run inference on a dataset and optionally persist predictions.

        Automatically uses streaming for large datasets (>50k rows) to prevent OOM.
        """
        table_name = str(table_name)
        output_table = str(output_table) if output_table else None

        if not self.ml_data_service.dataset_exists(table_name):
            raise MLRuntimeError(f"Table '{table_name}' does not exist")

        # Auto-detect whether to use streaming based on dataset size
        dataset_info = self.ml_data_service.get_dataset_info(table_name)
        use_streaming = False
        if dataset_info:
            effective_rows = dataset_info.row_count
            if limit is not None:
                effective_rows = min(effective_rows, limit)
            use_streaming = effective_rows > streaming_prediction_threshold

        predictor = self.load_predictor(model_name=model_name, device=device)

        try:
            if use_streaming:
                predictions = predictor.predict_from_table_streaming(
                    ml_data_service=self.ml_data_service,
                    table_name=table_name,
                    batch_size=batch_size,
                    chunk_size=streaming_prediction_chunk_size,
                    limit=limit,
                )
            else:
                predictions = predictor.predict_from_table(
                    ml_data_service=self.ml_data_service,
                    table_name=table_name,
                    batch_size=batch_size,
                    limit=limit,
                )
        except Exception as exc:  # noqa: BLE001
            raise MLRuntimeError(f"Prediction failed: {exc}") from exc

        output_names = list(predictions.keys())
        total_predictions = 0
        if output_names:
            total_predictions = predictions[output_names[0]].shape[0]

        saved_table_name = None
        if output_table:
            self.save_predictions_to_table(
                predictions=predictions,
                table_name=output_table,
                predictor=predictor,
                source_table=table_name,
                limit=limit,
            )
            saved_table_name = output_table

        return PredictionSummary(
            total_predictions=total_predictions,
            outputs=output_names,
            saved_table=saved_table_name,
        )

    def load_predictor(
        self, *, model_name: str, device: str | None = None
    ) -> ArcPredictor:
        """Load an ArcPredictor for the latest version of a model."""
        _validate_model_name(model_name)

        model_record = self.model_service.get_latest_model_by_name(model_name)
        if model_record is None:
            raise MLRuntimeError(f"Model '{model_name}' not found")

        model_key, _ = _split_model_identifier(model_record.id)

        try:
            return ArcPredictor.load_from_artifact(
                artifact_manager=self.artifact_manager,
                model_id=model_key,
                device=device or "cpu",
            )
        except Exception as exc:  # noqa: BLE001
            raise MLRuntimeError(f"Failed to load predictor: {exc}") from exc

    def save_predictions_to_table(
        self,
        *,
        predictions: dict[str, torch.Tensor],
        table_name: str,
        predictor: ArcPredictor,
        source_table: str,
        limit: int | None = None,
    ) -> None:
        """Persist predictions together with original feature columns."""
        # Extract feature columns from model spec
        feature_columns = []
        for input_spec in predictor.model_spec.inputs.values():
            if input_spec.columns:
                feature_columns.extend(input_spec.columns)

        if not feature_columns:
            raise MLRuntimeError("No feature columns found in model specification")

        try:
            self.ml_data_service.save_prediction_results(
                source_table=source_table,
                output_table=table_name,
                feature_columns=feature_columns,
                predictions=predictions,
                batch_size=1000,
                limit=limit,
            )
        except Exception as exc:  # noqa: BLE001
            raise MLRuntimeError(f"Failed to save predictions to table: {exc}") from exc


def _split_model_identifier(model_id: str) -> tuple[str, int | None]:
    match = MODEL_ID_PATTERN.match(model_id)
    if match:
        return match.group("base"), int(match.group("version"))
    return model_id, None


def _validate_model_name(name: str) -> None:
    """Validate model name follows proper naming conventions.

    Args:
        name: Model name to validate

    Raises:
        MLRuntimeError: If the model name is invalid
    """
    if not name or not name.strip():
        raise MLRuntimeError("Model name cannot be empty")

    # Check for spaces within the name (not just leading/trailing)
    if " " in name.strip():
        raise MLRuntimeError(
            "Model name cannot contain spaces. "
            "Use hyphens (-) or underscores (_) instead. "
            f"Example: '{name.replace(' ', '-')}'"
        )

    # Check for invalid characters that could cause issues
    if not re.match(r"^[a-zA-Z0-9_-]+$", name.strip()):
        raise MLRuntimeError(
            "Model name can only contain alphanumeric characters, "
            "hyphens, and underscores."
        )

    # Check length constraints
    if len(name.strip()) > 100:
        raise MLRuntimeError("Model name cannot exceed 100 characters")

    # Check that it doesn't start or end with hyphens/underscores (common convention)
    clean_name = name.strip()
    if clean_name.startswith(("-", "_")) or clean_name.endswith(("-", "_")):
        raise MLRuntimeError(
            "Model name should not start or end with hyphens or underscores"
        )


__all__ = [
    "MLRuntime",
    "MLRuntimeError",
    "PredictionSummary",
]
