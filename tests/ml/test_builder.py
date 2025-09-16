import pytest
import torch

from src.arc.graph import ArcGraph
from src.arc.ml.builder import ArcModel, ModelBuilder


class TestModelBuilder:
    """Test PyTorch model building from Arc-Graph."""

    def test_build_simple_linear_model(self):
        """Test building a simple linear model."""
        yaml_content = """
version: "0.1"
model_name: "simple_linear"

features:
  feature_columns: [x1, x2, x3]
  processors: []

model:
  inputs:
    features: {dtype: float32, shape: [null, 3]}
  graph:
    - name: linear
      type: core.Linear
      params: {in_features: 3, out_features: 1}
  outputs:
    prediction: linear.output

trainer:
  optimizer: {type: AdamW}
  loss: {type: core.MSELoss}
"""

        graph = ArcGraph.from_yaml(yaml_content)
        builder = ModelBuilder()

        # Build model
        model = builder.build_model(graph)

        assert isinstance(model, ArcModel)
        assert "linear" in model.layers
        assert model.input_names == ["features"]
        assert model.output_mapping == {"prediction": "linear.output"}

    def test_build_model_with_variables(self):
        """Test building model with variable references."""
        yaml_content = """
version: "0.1"
model_name: "variable_model"

features:
  feature_columns: [x1, x2]
  processors: []

model:
  inputs:
    features: {dtype: float32, shape: [null, vars.n_features]}
  graph:
    - name: linear
      type: core.Linear
      params: {in_features: vars.n_features, out_features: 1}
  outputs:
    prediction: linear.output

trainer:
  optimizer: {type: AdamW}
  loss: {type: core.MSELoss}
"""

        graph = ArcGraph.from_yaml(yaml_content)
        builder = ModelBuilder()

        # Set variable manually
        builder.set_variable("vars.n_features", 2)

        # Build model
        model = builder.build_model(graph)

        assert isinstance(model, ArcModel)
        assert builder.get_variable("vars.n_features") == 2

    def test_build_model_with_auto_detection(self):
        """Test building model with automatic size detection."""
        yaml_content = """
version: "0.1"
model_name: "auto_detect_model"

features:
  feature_columns: [x1, x2, x3, x4]
  processors: []

model:
  inputs:
    features: {dtype: float32, shape: [null, vars.n_features]}
  graph:
    - name: linear
      type: core.Linear
      params: {in_features: vars.n_features, out_features: 1}
  outputs:
    prediction: linear.output

trainer:
  optimizer: {type: AdamW}
  loss: {type: core.MSELoss}
"""

        graph = ArcGraph.from_yaml(yaml_content)
        builder = ModelBuilder()

        # Create sample data for auto-detection
        sample_data = torch.randn(100, 4)

        # Build model with auto-detection
        model = builder.build_model(graph, sample_data)

        assert isinstance(model, ArcModel)
        assert builder.get_variable("vars.n_features") == 4

    def test_build_logistic_regression_model(self):
        """Test building a logistic regression model."""
        yaml_content = """
version: "0.1"
model_name: "logistic_regression"

features:
  feature_columns: [x1, x2, x3]
  processors: []

model:
  inputs:
    features: {dtype: float32, shape: [null, 3]}
  graph:
    - name: linear
      type: core.Linear
      params: {in_features: 3, out_features: 1}
    - name: sigmoid
      type: core.Sigmoid
  outputs:
    probability: sigmoid.output

trainer:
  optimizer: {type: AdamW}
  loss: {type: core.BCELoss}
"""

        graph = ArcGraph.from_yaml(yaml_content)
        builder = ModelBuilder()

        # Build model
        model = builder.build_model(graph)

        assert isinstance(model, ArcModel)
        assert len(model.layers) == 2
        assert "linear" in model.layers
        assert "sigmoid" in model.layers

    def test_model_forward_pass(self):
        """Test forward pass through built model."""
        yaml_content = """
version: "0.1"
model_name: "test_forward"

features:
  feature_columns: [x1, x2]
  processors: []

model:
  inputs:
    features: {dtype: float32, shape: [null, 2]}
  graph:
    - name: linear
      type: core.Linear
      params: {in_features: 2, out_features: 1}
    - name: sigmoid
      type: core.Sigmoid
  outputs:
    probability: sigmoid.output

trainer:
  optimizer: {type: AdamW}
  loss: {type: core.BCELoss}
"""

        graph = ArcGraph.from_yaml(yaml_content)
        builder = ModelBuilder()
        model = builder.build_model(graph)

        # Test forward pass
        batch_size = 10
        features = torch.randn(batch_size, 2)
        inputs = {"features": features}

        outputs = model(inputs)

        assert "probability" in outputs
        assert outputs["probability"].shape == (batch_size, 1)
        assert torch.all(outputs["probability"] >= 0)
        assert torch.all(outputs["probability"] <= 1)

    def test_build_model_invalid_layer_type(self):
        """Test building model with invalid layer type."""
        yaml_content = """
version: "0.1"
model_name: "invalid_model"

features:
  feature_columns: [x1]
  processors: []

model:
  inputs:
    features: {dtype: float32, shape: [null, 1]}
  graph:
    - name: invalid
      type: invalid.Layer
      params: {}
  outputs:
    output: invalid.output

trainer:
  optimizer: {type: AdamW}
  loss: {type: core.MSELoss}
"""

        graph = ArcGraph.from_yaml(yaml_content)
        builder = ModelBuilder()

        with pytest.raises(ValueError, match="Unknown layer type"):
            builder.build_model(graph)

    def test_build_model_invalid_parameters(self):
        """Test building model with invalid layer parameters."""
        yaml_content = """
version: "0.1"
model_name: "invalid_params_model"

features:
  feature_columns: [x1]
  processors: []

model:
  inputs:
    features: {dtype: float32, shape: [null, 1]}
  graph:
    - name: linear
      type: core.Linear
      params: {in_features: -1, out_features: 1}  # Invalid negative size
  outputs:
    output: linear.output

trainer:
  optimizer: {type: AdamW}
  loss: {type: core.MSELoss}
"""

        graph = ArcGraph.from_yaml(yaml_content)
        builder = ModelBuilder()

        with pytest.raises(
            ValueError, match="Shape validation failed|Failed to create layer"
        ):
            builder.build_model(graph)
