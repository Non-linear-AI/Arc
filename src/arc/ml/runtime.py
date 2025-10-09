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
        self.trainer_service = services.trainers
        self.job_service = services.jobs
        self.ml_data_service = services.ml_data

        self.artifacts_root = Path(artifacts_dir or "artifacts")
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

        latest = self.model_service.get_latest_model_by_name(name)
        version = 1 if latest is None else latest.version + 1
        base_slug = _slugify_name(name)
        model_id = f"{base_slug}-v{version}"

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

    def create_trainer(
        self,
        *,
        name: str,
        model_id: str,
        schema_path: Path | None = None,
        schema_yaml: str | None = None,
        description: str | None = None,
    ):
        """Register a new Arc-Graph trainer specification.

        Args:
            name: Trainer name
            model_id: ID of the model this trainer is for (e.g., 'my_model-v1')
            schema_path: Path to trainer YAML (optional if schema_yaml provided)
            schema_yaml: Trainer YAML as string (optional if schema_yaml provided)
            description: Optional description

        Returns:
            Trainer object

        Raises:
            MLRuntimeError: If trainer registration fails
        """
        from arc.database.models.trainer import Trainer
        from arc.graph.trainer import TrainerSpec

        _validate_model_name(name)

        # Get schema text from either path or direct YAML
        if schema_yaml:
            schema_text = schema_yaml
        elif schema_path:
            schema_file = schema_path.expanduser()
            if not schema_file.exists():
                raise MLRuntimeError(f"Schema file not found: {schema_file}")

            try:
                schema_text = schema_file.read_text(encoding="utf-8")
            except OSError as exc:
                raise MLRuntimeError(f"Failed to read schema file: {exc}") from exc
        else:
            raise MLRuntimeError("Either schema_path or schema_yaml must be provided")

        # Parse and validate trainer spec
        try:
            trainer_spec = TrainerSpec.from_yaml(schema_text)
        except Exception as exc:  # noqa: BLE001
            raise MLRuntimeError(f"Invalid trainer schema: {exc}") from exc

        # Get the model this trainer references
        model_record = self.model_service.get_model_by_id(model_id)
        if model_record is None:
            raise MLRuntimeError(
                f"Model '{model_id}' not found. "
                f"Please check the model ID or create the model first."
            )

        # Validate that trainer's model_ref matches the model
        # The model_ref in the trainer spec should match the model ID
        expected_model_ref = model_record.id
        if trainer_spec.model_ref != expected_model_ref:
            raise MLRuntimeError(
                f"Trainer model_ref '{trainer_spec.model_ref}' does not match "
                f"model ID '{expected_model_ref}'. "
                f"Update the trainer YAML to use model_ref: {expected_model_ref}"
            )

        # Check version and generate new trainer ID
        latest = self.trainer_service.get_latest_trainer_by_name(name)
        version = 1 if latest is None else latest.version + 1
        base_slug = _slugify_name(name)
        trainer_id = f"{base_slug}-v{version}"

        now = datetime.now(UTC)
        trainer = Trainer(
            id=trainer_id,
            name=name,
            version=version,
            model_id=model_record.id,
            model_version=model_record.version,
            spec=schema_text,
            description=description or "",
            created_at=now,
            updated_at=now,
        )

        try:
            self.trainer_service.create_trainer(trainer)
        except DatabaseError as exc:
            raise MLRuntimeError(f"Failed to register trainer: {exc}") from exc

        return trainer

    def register_data_processor(
        self,
        name: str,
        spec,  # DataSourceSpec
        description: str | None = None,
    ):  # -> DataProcessor
        """Register a data processor in the database.

        Args:
            name: Data processor name
            spec: DataSourceSpec object (already parsed and validated)
            description: Optional description

        Returns:
            DataProcessor object with generated id and version

        Raises:
            MLRuntimeError: If registration fails
        """
        from arc.database.models.data_processor import DataProcessor

        _validate_model_name(name)

        # Convert spec to YAML string for storage
        spec_yaml = spec.to_yaml()

        # Get next version
        latest = self.services.data_processors.get_latest_data_processor_by_name(name)
        version = 1 if latest is None else latest.version + 1

        # Generate ID
        base_slug = _slugify_name(name)
        processor_id = f"{base_slug}-v{version}"

        # Create DataProcessor
        now = datetime.now(UTC)
        processor = DataProcessor(
            id=processor_id,
            name=name,
            version=version,
            spec=spec_yaml,
            description=description or spec.description,
            created_at=now,
            updated_at=now,
        )

        # Save to database
        try:
            self.services.data_processors.create_data_processor(processor)
        except DatabaseError as exc:
            raise MLRuntimeError(f"Failed to register data processor: {exc}") from exc

        return processor

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
        trainer_name: str,
        train_table: str,
        validation_table: str | None = None,
        checkpoint_dir: str | None = None,
        description: str | None = None,
        tags: list[str] | None = None,
    ) -> str:
        """Submit a training job using registered model and trainer.

        Args:
            model_name: Name of the registered model
            trainer_name: Name of the registered trainer
            train_table: Training data table
            validation_table: Optional validation table
            checkpoint_dir: Optional checkpoint directory
            description: Optional job description
            tags: Optional job tags

        Returns:
            Job ID

        Raises:
            MLRuntimeError: If model or trainer not found or validation fails
        """
        from arc.graph.trainer import TrainerSpec

        _validate_model_name(model_name)
        _validate_model_name(trainer_name)

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

        # Get registered trainer
        trainer_record = self.trainer_service.get_latest_trainer_by_name(trainer_name)
        if trainer_record is None:
            raise MLRuntimeError(
                f"Trainer '{trainer_name}' not found. Use /ml create-trainer first."
            )

        # Parse specs
        try:
            model_spec = ModelSpec.from_yaml(model_record.spec)
        except Exception as exc:  # noqa: BLE001
            raise MLRuntimeError(
                f"Failed to load model spec from stored record: {exc}"
            ) from exc

        try:
            trainer_spec = TrainerSpec.from_yaml(trainer_record.spec)
        except Exception as exc:  # noqa: BLE001
            raise MLRuntimeError(
                f"Failed to load trainer spec from stored record: {exc}"
            ) from exc

        # Validate trainer is for this model
        if trainer_spec.model_ref != model_record.id:
            raise MLRuntimeError(
                f"Trainer '{trainer_name}' (model_ref={trainer_spec.model_ref}) "
                f"is not compatible with model '{model_name}' (id={model_record.id})"
            )

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

        # Extract target column from model spec loss function
        if not model_spec.loss or not model_spec.loss.inputs:
            raise MLRuntimeError(
                "Model spec must define a loss function with inputs "
                "to determine target column"
            )

        target_column = model_spec.loss.inputs.get("target")
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
            missing_list = ", ".join(missing)
            raise MLRuntimeError(
                f"Training table is missing required column(s): {missing_list}"
            )

        # Use validation table if provided, otherwise use validation_split from trainer
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

        # Create job config using trainer spec values
        job_config = TrainingJobConfig(
            model_id=model_key,
            model_version=record_version,
            model_name=model_record.name,
            trainer_id=trainer_record.id,
            trainer_version=trainer_record.version,
            train_table=train_table,
            target_column=target_column,
            model_spec=model_spec,
            trainer_spec=trainer_spec,
            feature_columns=list(feature_columns),
            validation_table=validation_table,
            validation_split=trainer_spec.validation_split,
            epochs=trainer_spec.epochs,
            batch_size=trainer_spec.batch_size,
            learning_rate=trainer_spec.learning_rate,
            artifacts_dir=str(self.artifacts_root),
            checkpoint_dir=checkpoint_path,
            description=description
            or f"Training {model_record.name} with {trainer_record.name}",
            tags=tags,
        )

        job_id = self.training_service.submit_training_job(job_config)
        return job_id

    def train_with_trainer(
        self,
        *,
        trainer_name: str,
        train_table: str,
        target_column: str | None = None,
        validation_table: str | None = None,
        validation_split: float | None = None,
        epochs: int | None = None,
        batch_size: int | None = None,
        learning_rate: float | None = None,
        checkpoint_dir: str | None = None,
        description: str | None = None,
        tags: list[str] | None = None,
    ) -> str:
        """Submit a training job using a registered trainer (gets model from trainer).

        Args:
            trainer_name: Name of the registered trainer
            train_table: Training data table
            target_column: Optional override for target column
            validation_table: Optional validation table
            validation_split: Optional override for validation split
            epochs: Optional override for epochs
            batch_size: Optional override for batch size
            learning_rate: Optional override for learning rate
            checkpoint_dir: Optional checkpoint directory
            description: Optional job description
            tags: Optional job tags

        Returns:
            Job ID

        Raises:
            MLRuntimeError: If trainer/model not found or validation fails
        """
        from arc.graph.trainer import TrainerSpec

        _validate_model_name(trainer_name)

        train_table = str(train_table)
        validation_table = str(validation_table) if validation_table else None

        # Get registered trainer
        trainer_record = self.trainer_service.get_latest_trainer_by_name(trainer_name)
        if trainer_record is None:
            raise MLRuntimeError(
                f"Trainer '{trainer_name}' not found. Use /ml create-trainer first."
            )

        # Parse trainer spec
        try:
            trainer_spec = TrainerSpec.from_yaml(trainer_record.spec)
        except Exception as exc:  # noqa: BLE001
            raise MLRuntimeError(
                f"Failed to load trainer spec from stored record: {exc}"
            ) from exc

        # Get model using trainer's model_ref
        model_id = trainer_spec.model_ref
        if not model_id:
            raise MLRuntimeError(
                f"Trainer '{trainer_name}' does not have a model_ref. "
                "Trainer spec is invalid."
            )

        # Get model by ID (model_ref is the model ID like "pima-diabetes-v1")
        model_record = self.model_service.get_model_by_id(model_id)
        if model_record is None:
            raise MLRuntimeError(
                f"Model '{model_id}' referenced by trainer not found. "
                f"Trainer model_ref: {model_id}"
            )

        if not model_record.spec:
            raise MLRuntimeError("Stored model is missing specification; cannot train.")

        # Parse model spec
        try:
            model_spec = ModelSpec.from_yaml(model_record.spec)
        except Exception as exc:  # noqa: BLE001
            raise MLRuntimeError(
                f"Failed to load model spec from stored record: {exc}"
            ) from exc

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

        # Extract target column from model spec loss function or use override
        if target_column:
            # Use provided target column
            actual_target_column = target_column
        else:
            # Extract from model spec
            if not model_spec.loss or not model_spec.loss.inputs:
                raise MLRuntimeError(
                    "Model spec must define a loss function with inputs "
                    "to determine target column"
                )

            actual_target_column = model_spec.loss.inputs.get("target")
            if not actual_target_column:
                raise MLRuntimeError(
                    "Model spec loss function must define 'target' input "
                    "to specify target column"
                )

        columns_to_check = list(feature_columns)
        if actual_target_column not in columns_to_check:
            columns_to_check.append(actual_target_column)

        missing = [
            column
            for column, exists in self.ml_data_service.validate_columns(
                train_table, columns_to_check
            ).items()
            if not exists
        ]
        if missing:
            missing_list = ", ".join(missing)
            raise MLRuntimeError(
                f"Training table is missing required column(s): {missing_list}"
            )

        # Use validation table if provided
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

        # Apply overrides from parameters, otherwise use trainer spec values
        final_validation_split = (
            validation_split
            if validation_split is not None
            else trainer_spec.validation_split
        )
        final_epochs = epochs if epochs is not None else trainer_spec.epochs
        final_batch_size = (
            batch_size if batch_size is not None else trainer_spec.batch_size
        )
        final_learning_rate = (
            learning_rate if learning_rate is not None else trainer_spec.learning_rate
        )

        # Create job config
        job_config = TrainingJobConfig(
            model_id=model_key,
            model_version=record_version,
            model_name=model_record.name,
            trainer_id=trainer_record.id,
            trainer_version=trainer_record.version,
            train_table=train_table,
            target_column=actual_target_column,
            model_spec=model_spec,
            trainer_spec=trainer_spec,
            feature_columns=list(feature_columns),
            validation_table=validation_table,
            validation_split=final_validation_split,
            epochs=final_epochs,
            batch_size=final_batch_size,
            learning_rate=final_learning_rate,
            artifacts_dir=str(self.artifacts_root),
            checkpoint_dir=checkpoint_path,
            description=description
            or f"Training {model_record.name} with {trainer_record.name}",
            tags=tags,
        )

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


def _slugify_name(name: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "-", name.lower()).strip("-")
    return slug or "model"


__all__ = [
    "MLRuntime",
    "MLRuntimeError",
    "PredictionSummary",
]
