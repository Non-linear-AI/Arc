import os

import pytest

from src.arc.graph import ArcGraph, validate_graph_dict

yaml = pytest.importorskip("yaml")


def fixture_path(name: str) -> str:
    return os.path.join(os.path.dirname(__file__), "..", "fixtures", name)


def test_parse_valid_yaml():
    graph = ArcGraph.from_yaml_file(os.path.normpath(fixture_path("simple_model.yaml")))
    assert isinstance(graph, ArcGraph)
    assert graph.version == "0.1"
    assert graph.model_name == "simple_classifier"
    # Model spec
    assert "features" in graph.model.inputs
    assert graph.model.graph[0].type.startswith("core.")
    assert "probability" in graph.model.outputs
    # Trainer
    assert graph.trainer.optimizer.type.lower().startswith("adam")


def test_parse_invalid_yaml_raises_error():
    from src.arc.graph.validator import GraphValidationError

    with pytest.raises((ValueError, GraphValidationError)):
        ArcGraph.from_yaml_file(os.path.normpath(fixture_path("invalid_model.yaml")))


def test_parse_yaml_string():
    # Test parsing from YAML string
    yaml_content = """
version: "0.1"
model_name: "test_model"

features:
  feature_columns: [x, y]
  processors: []

model:
  inputs:
    features: {dtype: float32, shape: [null, 2]}
  graph:
    - name: linear
      type: core.Linear
  outputs:
    pred: linear.output

trainer:
  optimizer: {type: AdamW}
  loss: {type: core.MSELoss}
"""
    graph = ArcGraph.from_yaml(yaml_content)
    assert graph.model_name == "test_model"
    assert len(graph.features.feature_columns) == 2


def test_validate_schema_roundtrip():
    # Load, validate again, and reconstruct to ensure symmetry
    with open(os.path.normpath(fixture_path("simple_model.yaml"))) as f:
        data = yaml.safe_load(f)
    validate_graph_dict(data)
    g = ArcGraph.from_dict(data)
    assert g.model.outputs.get("probability")
