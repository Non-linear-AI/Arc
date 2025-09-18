"""Tests for MLRuntime shared operations."""

from __future__ import annotations

import json
import textwrap
from pathlib import Path

import pytest

from arc.database.manager import DatabaseManager
from arc.database.services import ServiceContainer
from arc.ml.runtime import MLRuntime, MLRuntimeError

SIMPLE_GRAPH_YAML = textwrap.dedent(
    """
    version: "0.1"
    model_name: "runtime_test_model"
    description: "Runtime test model"

    features:
      feature_columns: [feature1, feature2]
      target_columns: [label]

    model:
      inputs:
        features:
          dtype: float32
          shape: [null, 2]

      graph:
        - name: linear
          type: core.Linear
          params: {in_features: 2, out_features: 1, bias: true}
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
        epochs: 1
        batch_size: 2
        learning_rate: 0.1
        target_output_key: logits
        reshape_targets: true

    predictor:
      returns: [prediction]
    """
)


@pytest.fixture()
def db_manager(tmp_path):
    system_db = tmp_path / "system.duckdb"
    user_db = tmp_path / "user.duckdb"
    manager = DatabaseManager(system_db, user_db)
    try:
        yield manager
    finally:
        manager.close()


@pytest.fixture()
def services(db_manager):
    return ServiceContainer(db_manager)


@pytest.fixture()
def runtime(tmp_path, services):
    runtime = MLRuntime(services, artifacts_dir=tmp_path / "artifacts")
    try:
        yield runtime
    finally:
        runtime.shutdown()


def _prepare_training_table(manager: DatabaseManager) -> None:
    manager.user_execute(
        """
        CREATE TABLE train_data (
            feature1 DOUBLE,
            feature2 DOUBLE,
            label DOUBLE
        )
        """
    )

    rows = [
        (0.0, 0.1, 0.0),
        (0.2, 0.3, 0.0),
        (0.8, 0.7, 1.0),
        (0.9, 0.6, 1.0),
    ]
    for feature1, feature2, label in rows:
        manager.user_execute(
            f"INSERT INTO train_data VALUES ({feature1}, {feature2}, {label})"
        )


def test_create_model_registers_versions(tmp_path, runtime, services):
    schema_path = tmp_path / "graph.yaml"
    schema_path.write_text(SIMPLE_GRAPH_YAML)

    model = runtime.create_model(name="TestModel", schema_path=schema_path)
    assert model.version == 1

    model_v2 = runtime.create_model(name="TestModel", schema_path=schema_path)
    assert model_v2.version == 2
    assert model_v2.id.endswith("-v2")

    stored_models = services.models.get_models_by_name("TestModel")
    assert len(stored_models) == 2


def test_create_model_missing_file_raises(runtime):
    missing_path = Path("/nonexistent/schema.yaml")
    with pytest.raises(MLRuntimeError, match="Schema file not found"):
        runtime.create_model(name="MissingModel", schema_path=missing_path)


@pytest.mark.parametrize("output_table", ["predictions", "pred_table"])
def test_train_and_predict_flow(tmp_path, runtime, db_manager, output_table):
    schema_path = tmp_path / "schema.yaml"
    schema_path.write_text(SIMPLE_GRAPH_YAML)

    runtime.create_model(name="RuntimeFlow", schema_path=schema_path)

    _prepare_training_table(db_manager)

    job_id = runtime.train_model(
        model_name="RuntimeFlow",
        train_table="train_data",
        target_column="label",
        description="runtime test job",
    )

    result = runtime.training_service.wait_for_job(job_id, timeout=15)
    assert result is not None and result.success is True

    runtime.training_service.cleanup_completed_jobs()

    # Verify trained_models table updated
    trained_rows = db_manager.system_query("SELECT * FROM trained_models").rows
    assert len(trained_rows) == 1
    record = trained_rows[0]
    assert record["job_id"] == job_id
    assert record["model_id"] == "runtimeflow-v1"
    assert Path(record["artifact_path"]).exists()

    metrics = record["metrics"]
    if isinstance(metrics, str):
        metrics = json.loads(metrics)
    assert "final_metrics" in metrics

    summary = runtime.predict(
        model_name="RuntimeFlow",
        table_name="train_data",
        output_table=output_table,
    )

    assert summary.saved_table == output_table
    assert summary.total_predictions == 4
    assert summary.outputs == ["prediction"]

    prediction_count = db_manager.user_query(
        f"SELECT COUNT(*) as cnt FROM {output_table}"
    ).first()["cnt"]
    assert prediction_count == 4
