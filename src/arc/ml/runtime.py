"""Reusable runtime helpers for ML operations."""

from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING, Optional

import torch

from ..database import DatabaseError
from ..database.models.model import Model
from ..graph.spec import ArcGraph
from .artifacts import ModelArtifactManager
from .predictor import ArcPredictor
from .training_service import TrainingJobConfig, TrainingService

if TYPE_CHECKING:  # pragma: no cover - import heavy modules only for typing
    from ..database.services.container import ServiceContainer


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

    def __init__(self, services: "ServiceContainer", artifacts_dir: Path | None = None):
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
        schema_file = schema_path.expanduser()
        if not schema_file.exists():
            raise MLRuntimeError(f"Schema file not found: {schema_file}")

        try:
            schema_text = schema_file.read_text(encoding="utf-8")
        except OSError as exc:
            raise MLRuntimeError(f"Failed to read schema file: {exc}") from exc

        try:
            arc_graph = ArcGraph.from_yaml(schema_text)
            arc_graph.to_training_config()
        except Exception as exc:  # noqa: BLE001 - propagate as runtime error
            raise MLRuntimeError(f"Invalid Arc-Graph schema: {exc}") from exc

        latest = self.model_service.get_latest_model_by_name(name)
        version = 1 if latest is None else latest.version + 1
        model_id = f"{_slugify_name(name)}-v{version}"

        now = datetime.now(UTC)
        arc_graph_payload = json.dumps(asdict(arc_graph), default=str)

        model = Model(
            id=model_id,
            type=model_type or "ml.arc_graph",
            name=name,
            version=version,
            description=description or arc_graph.description or "",
            base_model_id=None,
            spec=schema_text,
            arc_graph=arc_graph_payload,
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
            arc_graph_dict = json.loads(model_record.arc_graph)
            arc_graph = ArcGraph.from_dict(arc_graph_dict)
        except Exception as exc:  # noqa: BLE001 - rewrap into runtime error
            raise MLRuntimeError(
                f"Failed to load Arc-Graph from stored spec: {exc}"
            ) from exc

        feature_columns = arc_graph.features.feature_columns
        if not feature_columns:
            raise MLRuntimeError(
                "Arc-Graph must define features.feature_columns before training."
            )

        if not self.ml_data_service.dataset_exists(train_table):
            raise MLRuntimeError(
                f"Training table '{train_table}' does not exist in user DB"
            )

        resolved_target = target_column
        if not resolved_target:
            targets = arc_graph.features.target_columns or []
            if not targets:
                raise MLRuntimeError(
                    "Unable to determine target column. Provide target explicitly."
                )
            resolved_target = targets[0]

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

        training_config = arc_graph.to_training_config(overrides or None)

        resolved_validation_split = (
            overrides.get("validation_split")
            if overrides.get("validation_split") is not None
            else training_config.validation_split
        )

        if validation_table:
            if not self.ml_data_service.dataset_exists(validation_table):
                raise MLRuntimeError(
                    f"Validation table '{validation_table}' does not exist in user DB"
                )

        checkpoint_path: Optional[str] = None
        if checkpoint_dir:
            checkpoint_path = str(Path(checkpoint_dir).expanduser())

        job_config = TrainingJobConfig(
            model_id=model_record.id,
            model_name=model_record.name,
            arc_graph=arc_graph,
            train_table=train_table,
            target_column=resolved_target,
            feature_columns=list(feature_columns),
            validation_table=validation_table,
            validation_split=resolved_validation_split,
            training_config=training_config,
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
    ) -> PredictionSummary:
        """Run inference on a dataset and optionally persist predictions."""
        table_name = str(table_name)
        output_table = str(output_table) if output_table else None

        if not self.ml_data_service.dataset_exists(table_name):
            raise MLRuntimeError(f"Table '{table_name}' does not exist")

        predictor = self.load_predictor(model_name=model_name, device=device)

        try:
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
        model_record = self.model_service.get_latest_model_by_name(model_name)
        if model_record is None:
            raise MLRuntimeError(f"Model '{model_name}' not found")

        try:
            return ArcPredictor.load_from_artifact(
                artifact_manager=self.artifact_manager,
                model_id=model_record.id,
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
        feature_columns = predictor.arc_graph.features.feature_columns
        if not feature_columns:
            raise MLRuntimeError(
                "No feature columns found in Arc-Graph specification"
            )

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
            raise MLRuntimeError(
                f"Failed to save predictions to table: {exc}"
            ) from exc


def _slugify_name(name: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "-", name.lower()).strip("-")
    return slug or "model"


__all__ = [
    "MLRuntime",
    "MLRuntimeError",
    "PredictionSummary",
]
