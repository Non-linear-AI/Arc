"""Tests for MLRuntime shared operations."""

from __future__ import annotations

import textwrap
from pathlib import Path

import pytest

from arc.database.manager import DatabaseManager
from arc.database.services import ServiceContainer
from arc.ml.runtime import MLRuntimeError

SIMPLE_GRAPH_YAML = textwrap.dedent(
    """
    inputs:
      features:
        dtype: float32
        shape: [null, 2]
        columns: [feature1, feature2]

    graph:
      - name: linear
        type: torch.nn.Linear
        params:
          in_features: 2
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
def services(tmp_path, db_manager):
    return ServiceContainer(db_manager, artifacts_dir=str(tmp_path / "artifacts"))


@pytest.fixture()
def runtime(services):
    runtime = services.ml_runtime
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


def test_register_data_processor(runtime, services):
    """Test registering a data processor."""
    from arc.graph.features.data_source import DataSourceSpec

    # Create a simple spec
    spec_dict = {
        "name": "test_processor",
        "description": "Test data processor",
        "steps": [
            {
                "name": "step1",
                "depends_on": [],
                "sql": "SELECT * FROM test_table",
            }
        ],
        "outputs": ["step1"],
    }
    spec = DataSourceSpec.from_dict(spec_dict)

    # Register the processor
    processor = runtime.register_data_processor(
        name="test_processor", spec=spec, description="Test processor"
    )

    assert processor.version == 1
    assert processor.name == "test_processor"
    assert "test_processor" in processor.spec

    # Verify it was stored
    stored = services.data_processors.get_latest_data_processor_by_name(
        "test_processor"
    )
    assert stored is not None
    assert stored.id == processor.id


def test_load_data_processor(runtime):
    """Test loading a data processor from database."""
    from arc.graph.features.data_source import DataSourceSpec

    # Create and register a processor using valid YAML directly
    yaml_str = """name: load_test
description: Load test processor
steps:
  - name: step1
    depends_on: []
    sql: SELECT * FROM test
outputs:
  - step1
"""
    spec = DataSourceSpec.from_yaml(yaml_str)
    runtime.register_data_processor(name="load_test", spec=spec)

    # Load it back
    processor, loaded_spec = runtime.load_data_processor("load_test")

    assert processor.name == "load_test"
    assert processor.version == 1
    assert loaded_spec.name == "load_test"
    assert len(loaded_spec.steps) == 1


def test_load_nonexistent_data_processor_raises(runtime):
    """Test that loading a nonexistent processor raises an error."""
    with pytest.raises(MLRuntimeError, match="not found"):
        runtime.load_data_processor("nonexistent_processor")
