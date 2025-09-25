"""Reusable runtime helpers for ML operations."""

from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass
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
        self.job_service = services.jobs
        self.ml_data_service = services.ml_data

        self.artifacts_root = Path(artifacts_dir or "artifacts")
        self.artifacts_root.mkdir(parents=True, exist_ok=True)

        self.artifact_manager = ModelArtifactManager(self.artifacts_root)
        self.training_service = TrainingService(
            self.job_service, artifacts_dir=self.artifacts_root
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
        model_spec_payload = json.dumps(asdict(model_spec), default=str)

        model = Model(
            id=model_id,
            type=model_type or "ml.model_spec",
            name=name,
            version=version,
            description=description or "",
            base_model_id=None,
            spec=schema_text,
            arc_graph=model_spec_payload,  # Keep field name for backward compatibility
            created_at=now,
            updated_at=now,
        )

        try:
            self.model_service.create_model(model)
        except DatabaseError as exc:
            raise MLRuntimeError(f"Failed to register model: {exc}") from exc

        return model

    def train_model(
        self,
        *,
        model_name: str,
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
        """Submit a training job for the given model and dataset."""
        _validate_model_name(model_name)

        train_table = str(train_table)
        validation_table = str(validation_table) if validation_table else None

        model_record = self.model_service.get_latest_model_by_name(model_name)
        if model_record is None:
            raise MLRuntimeError(
                f"Model '{model_name}' not found. Use create-model first."
            )

        if not model_record.arc_graph:
            raise MLRuntimeError(
                "Stored model is missing arc graph specification; cannot train."
            )

        try:
            # Use the model spec stored in the model record
            model_spec = ModelSpec.from_yaml(model_record.spec)
        except Exception as exc:  # noqa: BLE001 - rewrap into runtime error
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

        resolved_target = target_column
        if not resolved_target:
            # In the new model spec structure, target column must be provided explicitly
            # since model specs focus on architecture rather than data preprocessing
            raise MLRuntimeError(
                "Target column must be provided explicitly when starting training."
            )

        columns_to_check = list(feature_columns)
        if resolved_target not in columns_to_check:
            columns_to_check.append(resolved_target)

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

        overrides: dict[str, float | int] = {}
        if epochs is not None:
            overrides["epochs"] = epochs
        if batch_size is not None:
            overrides["batch_size"] = batch_size
        if learning_rate is not None:
            overrides["learning_rate"] = learning_rate
        if validation_split is not None:
            overrides["validation_split"] = validation_split

        # Extract training parameters for job config
        training_epochs = overrides.get("epochs", 10)
        training_batch_size = overrides.get("batch_size", 32)
        training_learning_rate = overrides.get("learning_rate", 0.001)

        resolved_validation_split = overrides.get("validation_split", 0.2)

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

        job_config = TrainingJobConfig(
            model_id=model_key,
            model_version=record_version,
            model_name=model_record.name,
            train_table=train_table,
            target_column=resolved_target,
            arc_graph=model_spec,  # Pass model spec
            feature_columns=list(feature_columns),
            validation_table=validation_table,
            validation_split=resolved_validation_split,
            epochs=training_epochs,
            batch_size=training_batch_size,
            learning_rate=training_learning_rate,
            artifacts_dir=str(self.artifacts_root),
            checkpoint_dir=checkpoint_path,
            description=description or "Training job",
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
