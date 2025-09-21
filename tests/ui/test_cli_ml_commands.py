"""Unit tests for /ml interactive console commands."""

from __future__ import annotations

import asyncio
import json
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pytest

from arc.database.manager import DatabaseManager
from arc.database.services import ServiceContainer
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


@dataclass
class StubModelRecord:
    id: str
    name: str
    version: int
    arc_graph: str
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
        artifacts_root: Path,
    ):
        self.model_service = model_service
        self.ml_data_service = ml_data_service
        self.job_service = job_service
        self.training_service = training_service
        self.artifacts_root = artifacts_root

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
        train_table: str,
        target_column: str | None = None,
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

        # For testing, create a simple mock config and submit to training service
        # Parse arc_graph to get target column if not specified
        arc_graph_data = json.loads(model.arc_graph)
        if not target_column and arc_graph_data.get("features", {}).get(
            "target_columns"
        ):
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


# ---------------------------------------------------------------------------
# Helpers


SCHEMA_WITH_CONFIG = """
version: "0.1"
model_name: "test_model"

features:
  feature_columns: [x1, x2]
  target_columns: [y]
  processors: []

model:
  inputs:
    features: {dtype: float32, shape: [null, 2]}
  graph:
    - name: linear
      type: core.Linear
      params: {in_features: 2, out_features: 1}
      inputs: {input: features}
  outputs:
    prediction: linear.output

trainer:
  optimizer: {type: adam}
  loss: {type: mse}
  config:
    epochs: 3
    batch_size: 16
    learning_rate: 0.01
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
version: "0.1"
model_name: "pima_diabetes_classifier"

features:
  feature_columns:
    - pregnancies
    - glucose
    - blood_pressure
    - skin_thickness
    - insulin
    - bmi
    - diabetes_pedigree
    - age
  target_columns: [outcome]
  processors: []

model:
  inputs:
    features: {dtype: float32, shape: [null, 8]}
  graph:
    - name: linear
      type: core.Linear
      params: {in_features: 8, out_features: 1, bias: true}
      inputs: {input: features}
    - name: sigmoid
      type: core.Sigmoid
      inputs: {input: linear.output}
  outputs:
    logits: linear.output
    prediction: sigmoid.output

trainer:
  optimizer: {type: adam}
  loss: {type: binary_cross_entropy_with_logits}
  config:
    epochs: 2
    batch_size: 4
    learning_rate: 0.05
    target_output_key: logits
    reshape_targets: true
"""


@pytest.mark.asyncio
async def test_create_model_registers_model(tmp_path):
    schema_path = tmp_path / "schema.yaml"
    schema_path.write_text(SCHEMA_WITH_CONFIG)

    ui = StubUI()
    model_service = StubModelService()
    runtime = StubRuntime(
        model_service=model_service,
        ml_data_service=StubMLDataService(),
        job_service=StubJobService(),
        training_service=StubTrainingService(),
        artifacts_root=tmp_path / "artifacts",
    )

    await handle_ml_command(
        f"/ml create-model --name my_model --schema {schema_path}", ui, runtime
    )

    assert ui.errors == []
    assert any("registered" in msg for msg in ui.successes)
    assert len(model_service.created_models) == 1
    created = model_service.created_models[0]
    assert created.name == "my_model"
    assert created.version == 1
    assert created.spec == SCHEMA_WITH_CONFIG
    assert json.loads(created.arc_graph)["model_name"] == "my_model"


@pytest.mark.asyncio
async def test_train_submits_job(tmp_path):
    arc_graph_dict = json.loads(
        json.dumps(
            {
                "version": "0.1",
                "model_name": "test_model",
                "description": "",
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
        artifacts_root=tmp_path / "artifacts",
    )

    ui = StubUI()
    await handle_ml_command(
        "/ml train --model my_model --data train_table",
        ui,
        runtime,
    )

    assert ui.errors == []
    assert any("Training job submitted" in msg for msg in ui.successes)
    assert any("Job ID" in msg for msg in ui.infos)
    assert len(training_service.submitted_jobs) == 1

    job_config = training_service.submitted_jobs[0]
    assert job_config.train_table == "train_table"
    assert job_config.target_column == "y"
    assert job_config.training_config.epochs == 3  # Default from arc_graph config
    assert (
        job_config.training_config.learning_rate == 0.01
    )  # Default from arc_graph config


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
    await handle_ml_command("/ml train --model unknown --data table", ui, runtime)

    assert any("not found" in err for err in ui.errors)


@pytest.mark.asyncio
async def test_end_to_end_training_with_realistic_dataset(tmp_path):
    manager = DatabaseManager(":memory:", ":memory:")
    services = ServiceContainer(manager, artifacts_dir=str(tmp_path / "artifacts"))

    table_sql = """
    CREATE TABLE pima_small (
        pregnancies DOUBLE,
        glucose DOUBLE,
        blood_pressure DOUBLE,
        skin_thickness DOUBLE,
        insulin DOUBLE,
        bmi DOUBLE,
        diabetes_pedigree DOUBLE,
        age DOUBLE,
        outcome DOUBLE
    )
    """
    manager.user_execute(table_sql)

    for row in PIMA_SMALL_ROWS:
        values = ", ".join(str(val) for val in row)
        manager.user_execute(f"INSERT INTO pima_small VALUES ({values})")

    schema_path = tmp_path / "model.yaml"
    schema_path.write_text(PIMA_SCHEMA)

    ui = StubUI()

    await handle_ml_command(
        f"/ml create-model --name pima_cli --schema {schema_path}",
        ui,
        services.ml_runtime,
    )
    assert any("registered" in msg for msg in ui.successes)

    await handle_ml_command(
        "/ml train --model pima_cli --data pima_small",
        ui,
        services.ml_runtime,
    )

    assert ui.infos, "Expected job id information"
    job_info = ui.infos[-1]
    assert "Job ID:" in job_info
    job_id = job_info.split("Job ID:")[-1].strip()

    result = services.ml_runtime.training_service.wait_for_job(job_id, timeout=10)
    assert result is not None and result.success is True

    await asyncio.sleep(0)

    job_record = services.jobs.get_job_by_id(job_id)
    assert job_record.status == JobStatus.COMPLETED

    services.shutdown()
