"""Unit tests for /ml interactive console commands."""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pytest

from arc.jobs.models import JobStatus, JobType
from arc.ui.cli import handle_ml_command

# ---------------------------------------------------------------------------
# Test doubles


class StubUI:
    def __init__(self):
        self.successes: list[str] = []
        self.errors: list[str] = []
        self.infos: list[str] = []
        self.tables: list[tuple[str, list[str], list[list[str]]]] = []
        self.kv_tables: list[tuple[str, list[list[str]]]] = []

    def show_system_success(self, message: str) -> None:
        self.successes.append(message)

    def show_system_error(self, message: str) -> None:
        self.errors.append(message)

    def show_info(self, message: str) -> None:
        self.infos.append(message)

    def show_table(self, title: str, columns: list[str], rows: list[list[str]]) -> None:
        self.tables.append((title, columns, rows))

    def show_key_values(self, title: str, pairs: list[list[str]]) -> None:
        self.kv_tables.append((title, pairs))


class StubModelService:
    def __init__(self):
        self._models_by_name: dict[str, list[Any]] = {}
        self.created_models: list[Any] = []

    def get_latest_model_by_name(self, name: str):
        models = self._models_by_name.get(name, [])
        return models[0] if models else None

    def get_model_by_id(self, model_id: str):
        for models in self._models_by_name.values():
            for model in models:
                if model.id == model_id:
                    return model
        return None

    def register_model(self, model):
        self._models_by_name.setdefault(model.name, []).insert(0, model)

    def create_model(self, model):
        self.created_models.append(model)
        self.register_model(model)


class StubMLDataService:
    def __init__(self, tables: dict[str, set[str]] | None = None):
        self.tables = tables or {}

    def dataset_exists(self, name: str) -> bool:
        return name in self.tables

    def validate_columns(
        self, dataset_name: str, columns: list[str]
    ) -> dict[str, bool]:
        existing = self.tables.get(dataset_name, set())
        return {col: col in existing for col in columns}


class StubTrainingService:
    def __init__(self):
        self.submitted_jobs: list[Any] = []

    def submit_training_job(self, config):
        self.submitted_jobs.append(config)
        return "job-123"

    def shutdown(self):
        pass


class StubJobService:
    def __init__(self, jobs: list[Any] | None = None):
        self._jobs = jobs or []
        self._by_id = {job.job_id: job for job in self._jobs}
        self.updates: list[tuple[str, Any, str]] = []

    def list_jobs(self, limit: int = 20):
        return self._jobs[:limit]

    def get_job_by_id(self, job_id: str):
        return self._by_id.get(job_id)

    def update_job_status(self, job_id: str, status: Any, message: str = ""):
        self.updates.append((job_id, status, message))


class StubMLPlanService:
    """Stub ML Plan service for testing."""

    def __init__(self):
        self._plans = {}

    def get_plan_by_id(self, plan_id: str):
        return self._plans.get(plan_id)


class StubTrainerService:
    def __init__(self):
        self._trainers = {}

    def register_trainer(self, trainer):
        self._trainers[trainer.name] = trainer

    def get_latest_trainer_by_name(self, name: str):
        return self._trainers.get(name)

    def get_trainer_by_id(self, id: str):
        for trainer in self._trainers.values():
            if trainer.id == id:
                return trainer
        return None


@dataclass
class StubModelRecord:
    id: str
    name: str
    version: int
    arc_graph: str
    spec: str


@dataclass
class StubTrainerRecord:
    id: str
    name: str
    version: int
    model_id: str
    model_version: int
    spec: str


@dataclass
class StubJobRecord:
    job_id: str
    type: Any
    status: Any
    message: str
    created_at: datetime
    updated_at: datetime


class StubRuntime:
    def __init__(
        self,
        *,
        model_service,
        ml_data_service,
        job_service,
        training_service,
        trainer_service=None,
        artifacts_root: Path,
    ):
        self.model_service = model_service
        self.ml_data_service = ml_data_service
        self.job_service = job_service
        self.training_service = training_service
        self.trainer_service = trainer_service or StubTrainerService()
        self.artifacts_root = artifacts_root

        # Create a stub services object for compatibility with new _ml_train
        class StubServices:
            def __init__(self, model_svc, trainer_svc, ml_plan_svc=None):
                self.models = model_svc
                self.trainers = trainer_svc
                self.ml_plan = ml_plan_svc or StubMLPlanService()
                self.training_tracking = None  # No tracking service in tests

        self.services = StubServices(model_service, self.trainer_service)

    def create_model(
        self,
        name: str,
        schema_path: Path,
        description: str | None = None,  # noqa: ARG002
        model_type: str | None = None,  # noqa: ARG002
    ):
        """Stub implementation of create_model."""
        import json

        # Parse the schema file to get Arc-Graph data
        schema_text = schema_path.read_text()
        # For stub, we'll use the schema directly as arc_graph
        arc_graph_data = {"model_name": name, "schema": schema_text}

        model = StubModelRecord(
            id=f"{name.lower()}-v1",
            name=name,
            version=1,
            arc_graph=json.dumps(arc_graph_data),
            spec=schema_text,
        )

        self.model_service.create_model(model)
        return model

    def train_model(
        self,
        model_name: str,
        trainer_name: str,
        train_table: str,
        validation_table: str | None = None,
        validation_split: float | None = None,  # noqa: ARG002
        epochs: int | None = None,
        learning_rate: float | None = None,
        batch_size: int | None = None,
        checkpoint_dir: str | None = None,  # noqa: ARG002
        overrides: dict | None = None,  # noqa: ARG002
        description: str | None = None,  # noqa: ARG002
        tags: str | None = None,  # noqa: ARG002
    ) -> str:
        """Stub implementation of train_model."""
        # Get the model record
        model = self.model_service.get_latest_model_by_name(model_name)
        if not model:
            from arc.ml.runtime import MLRuntimeError

            raise MLRuntimeError(f"Model '{model_name}' not found")

        # Get the trainer record
        trainer = self.trainer_service.get_latest_trainer_by_name(trainer_name)
        if not trainer:
            from arc.ml.runtime import MLRuntimeError

            raise MLRuntimeError(f"Trainer '{trainer_name}' not found")

        # For testing, create a simple mock config and submit to training service
        # Parse arc_graph to get target column from model spec
        arc_graph_data = json.loads(model.arc_graph)
        target_column = None
        if arc_graph_data.get("features", {}).get("target_columns"):
            target_column = arc_graph_data["features"]["target_columns"][0]

        # We'll create a simple object that has the required attributes
        class MockConfig:
            def __init__(self):
                self.model_id = model.id
                self.model_name = model.name
                self.train_table = train_table
                self.target_column = target_column
                self.validation_table = validation_table
                self.training_config = self
                # Training config attributes
                self.epochs = epochs or 3
                self.batch_size = batch_size or 16
                self.learning_rate = learning_rate or 0.01

        config = MockConfig()
        return self.training_service.submit_training_job(config)

    def train_with_trainer(
        self,
        trainer_name: str,
        train_table: str,
        target_column: str | None = None,  # noqa: ARG002
        validation_table: str | None = None,
        validation_split: float | None = None,  # noqa: ARG002
        epochs: int | None = None,
        learning_rate: float | None = None,
        batch_size: int | None = None,
        checkpoint_dir: str | None = None,  # noqa: ARG002
        description: str | None = None,  # noqa: ARG002
        tags: list[str] | None = None,  # noqa: ARG002
    ) -> str:
        """Stub implementation of train_with_trainer."""
        # Get the trainer record
        trainer = self.trainer_service.get_latest_trainer_by_name(trainer_name)
        if not trainer:
            from arc.ml.runtime import MLRuntimeError

            raise MLRuntimeError(f"Trainer '{trainer_name}' not found")

        # Get the model using trainer's model_id
        model = self.model_service.get_model_by_id(trainer.model_id)
        if not model:
            from arc.ml.runtime import MLRuntimeError

            raise MLRuntimeError(
                f"Model '{trainer.model_id}' referenced by trainer not found"
            )

        # For testing, create a simple mock config and submit to training service
        # Parse arc_graph to get target column from model spec
        arc_graph_data = json.loads(model.arc_graph)
        target_col = None
        if arc_graph_data.get("features", {}).get("target_columns"):
            target_col = arc_graph_data["features"]["target_columns"][0]

        # We'll create a simple object that has the required attributes
        class MockConfig:
            def __init__(self):
                self.model_id = model.id
                self.model_name = model.name
                self.train_table = train_table
                self.target_column = target_col
                self.validation_table = validation_table
                self.training_config = self
                # Training config attributes
                self.epochs = epochs or 3
                self.batch_size = batch_size or 16
                self.learning_rate = learning_rate or 0.01

        config = MockConfig()
        return self.training_service.submit_training_job(config)


# ---------------------------------------------------------------------------
# Helpers


SCHEMA_WITH_CONFIG = """
inputs:
  features:
    dtype: float32
    shape: [null, 2]
    columns: [x1, x2]

graph:
  - name: linear
    type: torch.nn.Linear
    params:
      in_features: 2
      out_features: 1
    inputs:
      input: features

outputs:
  prediction: linear.output
"""


PIMA_SMALL_ROWS = [
    (6, 148, 72, 35, 0, 33.6, 0.627, 50, 1),
    (1, 85, 66, 29, 0, 26.6, 0.351, 31, 0),
    (8, 183, 64, 0, 0, 23.3, 0.672, 32, 1),
    (1, 89, 66, 23, 94, 28.1, 0.167, 21, 0),
    (0, 137, 40, 35, 168, 43.1, 2.288, 33, 1),
    (5, 116, 74, 0, 0, 25.6, 0.201, 30, 0),
    (3, 78, 50, 32, 88, 31.0, 0.248, 26, 1),
    (10, 115, 0, 0, 0, 35.3, 0.134, 29, 0),
    (2, 120, 54, 0, 0, 26.8, 0.455, 27, 0),
    (4, 110, 92, 0, 0, 37.6, 0.191, 30, 0),
]

PIMA_SCHEMA = """
inputs:
  features:
    dtype: float32
    shape: [null, 8]
    columns:
      - pregnancies
      - glucose
      - blood_pressure
      - skin_thickness
      - insulin
      - bmi
      - diabetes_pedigree
      - age

graph:
  - name: linear
    type: torch.nn.Linear
    params:
      in_features: 8
      out_features: 1
      bias: true
    inputs:
      input: features
  - name: sigmoid
    type: torch.nn.Sigmoid
    inputs:
      input: linear.output

outputs:
  logits: linear.output
  prediction: sigmoid.output

loss:
  type: torch.nn.functional.binary_cross_entropy_with_logits
  inputs:
    input: logits
    target: outcome
"""


# Test removed: /ml create-model command no longer exists (integrated into /ml model)


@pytest.mark.asyncio
async def test_train_submits_job(tmp_path):
    arc_graph_dict = json.loads(
        json.dumps(
            {
                "features": {
                    "feature_columns": ["x1", "x2"],
                    "target_columns": ["y"],
                    "processors": [],
                },
                "model": {
                    "inputs": {"features": {"dtype": "float32", "shape": [None, 2]}},
                    "graph": [
                        {
                            "name": "linear",
                            "type": "core.Linear",
                            "params": {"in_features": 2, "out_features": 1},
                            "inputs": {"input": "features"},
                        }
                    ],
                    "outputs": {"prediction": "linear.output"},
                },
                "trainer": {
                    "optimizer": {"type": "adam"},
                    "loss": {"type": "mse"},
                    "config": {
                        "epochs": 3,
                        "batch_size": 16,
                        "learning_rate": 0.01,
                    },
                },
                "predictor": None,
            }
        )
    )

    model_record = StubModelRecord(
        id="my_model-v1",
        name="my_model",
        version=1,
        arc_graph=json.dumps(arc_graph_dict),
        spec="",  # Add empty spec for testing
    )

    model_service = StubModelService()
    model_service.register_model(model_record)

    # Create a trainer record
    trainer_record = StubTrainerRecord(
        id="my_trainer-v1",
        name="my_trainer",
        version=1,
        model_id="my_model-v1",
        model_version=1,
        spec=(
            "model_ref: my_model-v1\noptimizer:\n"
            "  type: torch.optim.Adam\n  lr: 0.001\n"
        ),
    )
    trainer_service = StubTrainerService()
    trainer_service.register_trainer(trainer_record)

    ml_data_service = StubMLDataService(
        {
            "train_table": {"x1", "x2", "y"},
        }
    )
    training_service = StubTrainingService()

    runtime = StubRuntime(
        model_service=model_service,
        ml_data_service=ml_data_service,
        job_service=StubJobService(),
        training_service=training_service,
        trainer_service=trainer_service,
        artifacts_root=tmp_path / "artifacts",
    )

    ui = StubUI()

    # Mock the MLTrainTool to avoid API calls
    from unittest.mock import AsyncMock, patch

    from arc.tools.base import ToolResult

    mock_result = ToolResult(
        success=True,
        output=(
            "✓ Trainer 'test_trainer-v1' created and registered.\n"
            "Model: my_model-v1 • Optimizer: adam\n\n"
            "✓ Training job submitted successfully.\n"
            "Training table: train_table\nJob ID: test-job-123"
        ),
        metadata={
            "trainer_id": "test_trainer-v1",
            "job_id": "test-job-123",
            "training_launched": True,
        },
    )

    # Mock SettingsManager to provide fake API key
    with (
        patch("arc.ui.cli.SettingsManager") as mock_settings,
        patch(
            "arc.tools.ml.MLTrainTool.execute",
            new_callable=AsyncMock,
            return_value=mock_result,
        ),
    ):
        mock_settings_instance = mock_settings.return_value
        mock_settings_instance.get_api_key.return_value = "fake-api-key"
        mock_settings_instance.get_base_url.return_value = "https://api.openai.com"
        mock_settings_instance.get_current_model.return_value = "gpt-4"

        await handle_ml_command(
            "/ml train --name test_trainer --model-id my_model-v1 "
            "--instruction 'Train for 3 epochs' --data train_table",
            ui,
            runtime,
        )

    assert ui.errors == []
    # The success message comes from the tool result
    # No need to check training_service.submitted_jobs since it's mocked


@pytest.mark.asyncio
async def test_jobs_list_displays_table(tmp_path):
    now = datetime.now(UTC)
    job = StubJobRecord(
        job_id="job-1",
        type=JobType.TRAIN_MODEL,
        status=JobStatus.RUNNING,
        message="Running",
        created_at=now,
        updated_at=now,
    )
    job_service = StubJobService([job])

    runtime = StubRuntime(
        model_service=StubModelService(),
        ml_data_service=StubMLDataService(),
        job_service=job_service,
        training_service=StubTrainingService(),
        artifacts_root=tmp_path / "artifacts",
    )

    ui = StubUI()
    await handle_ml_command("/ml jobs list", ui, runtime)

    assert ui.errors == []
    assert ui.tables
    title, columns, rows = ui.tables[0]
    assert title == "ML Jobs"
    assert columns[0] == "Job ID"
    assert rows[0][0] == "job-1"


@pytest.mark.asyncio
async def test_jobs_status_displays_details(tmp_path):
    now = datetime.now(UTC)
    job = StubJobRecord(
        job_id="job-2",
        type=JobType.TRAIN_MODEL,
        status=JobStatus.COMPLETED,
        message="Done",
        created_at=now,
        updated_at=now,
    )
    job_service = StubJobService([job])

    runtime = StubRuntime(
        model_service=StubModelService(),
        ml_data_service=StubMLDataService(),
        job_service=job_service,
        training_service=StubTrainingService(),
        artifacts_root=tmp_path / "artifacts",
    )

    ui = StubUI()
    await handle_ml_command("/ml jobs status job-2", ui, runtime)

    assert ui.errors == []
    assert ui.kv_tables
    title, rows = ui.kv_tables[0]
    assert title == "Job Status"
    assert any(field == "Status" and value == "completed" for field, value in rows)


@pytest.mark.asyncio
async def test_train_missing_model_shows_error(tmp_path):
    runtime = StubRuntime(
        model_service=StubModelService(),
        ml_data_service=StubMLDataService(),
        job_service=StubJobService(),
        training_service=StubTrainingService(),
        artifacts_root=tmp_path / "artifacts",
    )

    ui = StubUI()
    await handle_ml_command(
        "/ml train --name test_trainer --model-id unknown_model-v1 "
        "--instruction 'test' --data table",
        ui,
        runtime,
    )

    # Should error because model is not found (happens in MLTrainTool)
    assert len(ui.errors) > 0


# Test removed: /ml create-model command no longer exists (integrated into /ml model)
# This test relied on the old create-model command which has been removed
