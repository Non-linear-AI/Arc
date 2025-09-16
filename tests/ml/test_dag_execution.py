"""Tests for complex DAG execution in Arc models."""

import pytest
import torch

from src.arc.graph import ArcGraph
from src.arc.ml.builder import ModelBuilder


class TestDAGExecution:
    """Test complex graph topologies and DAG execution."""

    def test_simple_dag_execution(self):
        """Test basic DAG with explicit input mappings."""
        yaml_content = """
version: "0.1"
model_name: "simple_dag"

features:
  feature_columns: [x1, x2]
  processors: []

model:
  inputs:
    input_a: {dtype: float32, shape: [null, 2]}
    input_b: {dtype: float32, shape: [null, 2]}
  graph:
    - name: linear1
      type: core.Linear
      params: {in_features: 2, out_features: 3}
      inputs: {input: input_a}
    - name: linear2
      type: core.Linear
      params: {in_features: 2, out_features: 3}
      inputs: {input: input_b}
    - name: relu1
      type: core.ReLU
      inputs: {input: linear1.output}
    - name: relu2
      type: core.ReLU
      inputs: {input: linear2.output}
  outputs:
    output1: relu1.output
    output2: relu2.output

trainer:
  optimizer: {type: AdamW}
  loss: {type: core.MSELoss}
"""

        graph = ArcGraph.from_yaml(yaml_content)
        builder = ModelBuilder()
        model = builder.build_model(graph)

        # Test forward pass with two inputs
        batch_size = 5
        input_a = torch.randn(batch_size, 2)
        input_b = torch.randn(batch_size, 2)
        inputs = {"input_a": input_a, "input_b": input_b}

        outputs = model(inputs)

        assert "output1" in outputs
        assert "output2" in outputs
        assert outputs["output1"].shape == (batch_size, 3)
        assert outputs["output2"].shape == (batch_size, 3)

        # Verify outputs are different (parallel processing)
        assert not torch.equal(outputs["output1"], outputs["output2"])

    def test_convergent_dag_execution(self):
        """Test DAG where multiple paths converge (needs concatenation)."""
        yaml_content = """
version: "0.1"
model_name: "convergent_dag"

features:
  feature_columns: [x]
  processors: []

model:
  inputs:
    features: {dtype: float32, shape: [null, 4]}
  graph:
    - name: linear1
      type: core.Linear
      params: {in_features: 4, out_features: 2}
      inputs: {input: features}
    - name: linear2
      type: core.Linear
      params: {in_features: 4, out_features: 2}
      inputs: {input: features}
    - name: relu1
      type: core.ReLU
      inputs: {input: linear1.output}
    - name: relu2
      type: core.ReLU
      inputs: {input: linear2.output}
  outputs:
    path1: relu1.output
    path2: relu2.output

trainer:
  optimizer: {type: AdamW}
  loss: {type: core.MSELoss}
"""

        graph = ArcGraph.from_yaml(yaml_content)
        builder = ModelBuilder()
        model = builder.build_model(graph)

        # Test forward pass
        batch_size = 3
        features = torch.randn(batch_size, 4)
        inputs = {"features": features}

        outputs = model(inputs)

        assert "path1" in outputs
        assert "path2" in outputs
        assert outputs["path1"].shape == (batch_size, 2)
        assert outputs["path2"].shape == (batch_size, 2)

    def test_sequential_fallback(self):
        """Test that sequential execution still works without explicit inputs."""
        yaml_content = """
version: "0.1"
model_name: "sequential_fallback"

features:
  feature_columns: [x]
  processors: []

model:
  inputs:
    features: {dtype: float32, shape: [null, 3]}
  graph:
    - name: linear
      type: core.Linear
      params: {in_features: 3, out_features: 2}
    - name: relu
      type: core.ReLU
    - name: output_layer
      type: core.Linear
      params: {in_features: 2, out_features: 1}
  outputs:
    prediction: output_layer.output

trainer:
  optimizer: {type: AdamW}
  loss: {type: core.MSELoss}
"""

        graph = ArcGraph.from_yaml(yaml_content)
        builder = ModelBuilder()
        model = builder.build_model(graph)

        # Test forward pass
        batch_size = 4
        features = torch.randn(batch_size, 3)
        inputs = {"features": features}

        outputs = model(inputs)

        assert "prediction" in outputs
        assert outputs["prediction"].shape == (batch_size, 1)

    def test_execution_order_correctness(self):
        """Test that layers execute in correct topological order."""
        yaml_content = """
version: "0.1"
model_name: "order_test"

features:
  feature_columns: [x]
  processors: []

model:
  inputs:
    features: {dtype: float32, shape: [null, 2]}
  graph:
    # Intentionally define nodes out of execution order
    - name: final_layer
      type: core.Linear
      params: {in_features: 2, out_features: 1}
      inputs: {input: relu.output}
    - name: relu
      type: core.ReLU
      inputs: {input: first_layer.output}
    - name: first_layer
      type: core.Linear
      params: {in_features: 2, out_features: 2}
      inputs: {input: features}
  outputs:
    result: final_layer.output

trainer:
  optimizer: {type: AdamW}
  loss: {type: core.MSELoss}
"""

        graph = ArcGraph.from_yaml(yaml_content)
        builder = ModelBuilder()
        model = builder.build_model(graph)

        # Verify execution order is correct
        expected_order = ["first_layer", "relu", "final_layer"]
        assert model.execution_order == expected_order

        # Test forward pass works
        batch_size = 2
        features = torch.randn(batch_size, 2)
        inputs = {"features": features}

        outputs = model(inputs)
        assert "result" in outputs
        assert outputs["result"].shape == (batch_size, 1)

    def test_circular_dependency_detection(self):
        """Test that circular dependencies are detected and raise error."""
        yaml_content = """
version: "0.1"
model_name: "circular_test"

features:
  feature_columns: [x]
  processors: []

model:
  inputs:
    features: {dtype: float32, shape: [null, 2]}
  graph:
    - name: layer1
      type: core.Linear
      params: {in_features: 2, out_features: 2}
      inputs: {input: layer2.output}  # Depends on layer2
    - name: layer2
      type: core.Linear
      params: {in_features: 2, out_features: 2}
      inputs: {input: layer1.output}  # Depends on layer1 - circular!
  outputs:
    result: layer1.output

trainer:
  optimizer: {type: AdamW}
  loss: {type: core.MSELoss}
"""

        graph = ArcGraph.from_yaml(yaml_content)
        builder = ModelBuilder()

        with pytest.raises(ValueError, match="Circular dependency detected"):
            builder.build_model(graph)

    def test_missing_dependency_error(self):
        """Test error handling for missing dependencies."""
        yaml_content = """
version: "0.1"
model_name: "missing_dep_test"

features:
  feature_columns: [x]
  processors: []

model:
  inputs:
    features: {dtype: float32, shape: [null, 2]}
  graph:
    - name: layer1
      type: core.Linear
      params: {in_features: 2, out_features: 2}
      inputs: {input: nonexistent_layer.output}  # References missing layer
  outputs:
    result: layer1.output

trainer:
  optimizer: {type: AdamW}
  loss: {type: core.MSELoss}
"""

        graph = ArcGraph.from_yaml(yaml_content)
        builder = ModelBuilder()

        # Model building should fail with missing reference during shape validation
        with pytest.raises(ValueError, match="Shape validation failed"):
            builder.build_model(graph)
