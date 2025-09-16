"""Tests for shape inference and validation system."""

import pytest
import torch

from src.arc.graph import ArcGraph
from src.arc.ml.builder import ModelBuilder
from src.arc.ml.utils import ShapeInferenceError, ShapeValidator


class TestShapeValidator:
    """Test ShapeValidator functionality."""

    def test_parse_shape_spec(self):
        """Test parsing shape specifications."""
        validator = ShapeValidator({"vars.seq_len": 32, "vars.d_model": 128})

        # Test basic shapes
        assert validator.parse_shape_spec([None, 10]) == [None, 10]
        assert validator.parse_shape_spec(["null", 10]) == [None, 10]
        assert validator.parse_shape_spec([64, 32]) == [64, 32]

        # Test variable resolution
        assert validator.parse_shape_spec([None, "vars.seq_len"]) == [None, 32]
        assert validator.parse_shape_spec(["vars.d_model", "vars.seq_len"]) == [128, 32]

    def test_parse_shape_spec_invalid_variable(self):
        """Test error on undefined variable."""
        validator = ShapeValidator()

        with pytest.raises(ShapeInferenceError, match="Cannot resolve shape variable"):
            validator.parse_shape_spec([None, "vars.unknown"])

    def test_infer_linear_shape(self):
        """Test Linear layer shape inference."""
        validator = ShapeValidator()

        # Test 2D input
        input_shapes = {"input": [32, 10]}
        params = {"in_features": 10, "out_features": 5}
        output_shape = validator._infer_linear_shape(params, input_shapes)
        assert output_shape == [32, 5]

        # Test 3D input (batch, seq, features)
        input_shapes = {"input": [16, 20, 128]}
        params = {"in_features": 128, "out_features": 64}
        output_shape = validator._infer_linear_shape(params, input_shapes)
        assert output_shape == [16, 20, 64]

    def test_infer_linear_shape_mismatch(self):
        """Test Linear layer with mismatched input size."""
        validator = ShapeValidator()

        input_shapes = {"input": [32, 10]}
        params = {"in_features": 20, "out_features": 5}  # Wrong input size

        with pytest.raises(ShapeInferenceError, match="input size mismatch"):
            validator._infer_linear_shape(params, input_shapes)

    def test_infer_embedding_shape(self):
        """Test Embedding layer shape inference."""
        validator = ShapeValidator()

        # Test 2D input (batch, seq_len)
        input_shapes = {"input": [8, 50]}
        params = {"embedding_dim": 128}
        output_shape = validator._infer_embedding_shape(params, input_shapes)
        assert output_shape == [8, 50, 128]

        # Test 1D input (seq_len,)
        input_shapes = {"input": [50]}
        params = {"embedding_dim": 64}
        output_shape = validator._infer_embedding_shape(params, input_shapes)
        assert output_shape == [50, 64]

    def test_infer_concatenate_shape(self):
        """Test Concatenate layer shape inference."""
        validator = ShapeValidator()

        # Test concatenation along last dimension
        input_shapes = {"input1": [4, 10], "input2": [4, 15], "input3": [4, 5]}
        params = {"dim": -1}
        output_shape = validator._infer_concatenate_shape(params, input_shapes)
        assert output_shape == [4, 30]  # 10 + 15 + 5 = 30

        # Test concatenation along first dimension
        input_shapes = {"input1": [8, 32], "input2": [12, 32]}
        params = {"dim": 0}
        output_shape = validator._infer_concatenate_shape(params, input_shapes)
        assert output_shape == [20, 32]  # 8 + 12 = 20

    def test_infer_concatenate_shape_mismatch(self):
        """Test Concatenate layer with incompatible shapes."""
        validator = ShapeValidator()

        # Different number of dimensions
        input_shapes = {"input1": [4, 10], "input2": [4, 10, 5]}
        params = {"dim": -1}

        with pytest.raises(ShapeInferenceError, match="different number of dimensions"):
            validator._infer_concatenate_shape(params, input_shapes)

        # Mismatched non-concat dimensions
        input_shapes = {
            "input1": [4, 10],
            "input2": [8, 15],  # Different batch size
        }
        params = {"dim": -1}

        with pytest.raises(ShapeInferenceError, match="dimension .* mismatch"):
            validator._infer_concatenate_shape(params, input_shapes)

    def test_infer_add_shape(self):
        """Test Add layer shape inference."""
        validator = ShapeValidator()

        # Compatible shapes
        input_shapes = {"input1": [4, 32], "input2": [4, 32], "input3": [4, 32]}
        output_shape = validator._infer_add_shape(input_shapes)
        assert output_shape == [4, 32]

    def test_infer_add_shape_mismatch(self):
        """Test Add layer with incompatible shapes."""
        validator = ShapeValidator()

        # Different shapes
        input_shapes = {
            "input1": [4, 32],
            "input2": [4, 16],  # Different last dimension
        }

        with pytest.raises(ShapeInferenceError, match="dimension .* mismatch"):
            validator._infer_add_shape(input_shapes)

    def test_infer_rnn_shape(self):
        """Test LSTM/GRU shape inference."""
        validator = ShapeValidator()

        # Standard LSTM
        input_shapes = {"input": [8, 20, 128]}  # [batch, seq, input_size]
        params = {"hidden_size": 256, "bidirectional": False}
        output_shape = validator._infer_rnn_shape(params, input_shapes)
        assert output_shape == [8, 20, 256]

        # Bidirectional LSTM
        params = {"hidden_size": 256, "bidirectional": True}
        output_shape = validator._infer_rnn_shape(params, input_shapes)
        assert output_shape == [8, 20, 512]  # 256 * 2 = 512

    def test_infer_attention_shape(self):
        """Test MultiHeadAttention shape inference."""
        validator = ShapeValidator()

        # Self-attention
        input_shapes = {"input": [4, 20, 128]}
        params = {"embed_dim": 128, "num_heads": 8}
        output_shape = validator._infer_attention_shape(params, input_shapes)
        assert output_shape == [4, 20, 128]

        # Cross-attention with explicit query, key, value
        input_shapes = {
            "query": [4, 15, 128],
            "key": [4, 20, 128],
            "value": [4, 20, 128],
        }
        output_shape = validator._infer_attention_shape(params, input_shapes)
        assert output_shape == [4, 15, 128]  # Matches query shape


class TestModelBuilderShapeValidation:
    """Test shape validation integration in ModelBuilder."""

    def test_build_model_with_shape_validation(self):
        """Test building model with shape validation enabled."""
        yaml_content = """
version: "0.1"
model_name: "shape_test"

features:
  feature_columns: [features]
  processors: []

model:
  inputs:
    features: {dtype: float32, shape: [null, vars.n_features]}
  graph:
    - name: linear1
      type: core.Linear
      params: {in_features: vars.n_features, out_features: 64}
    - name: relu
      type: core.ReLU
    - name: linear2
      type: core.Linear
      params: {in_features: 64, out_features: 1}
  outputs:
    prediction: linear2.output

trainer:
  optimizer: {type: AdamW}
  loss: {type: core.MSELoss}
"""

        graph = ArcGraph.from_yaml(yaml_content)
        builder = ModelBuilder(enable_shape_validation=True)
        builder.set_variable("vars.n_features", 10)

        # Build model with shape validation
        _ = builder.build_model(graph)

        # Check shape info
        shape_info = builder.get_shape_info()
        assert "linear1" in shape_info
        assert "relu" in shape_info
        assert "linear2" in shape_info

        assert shape_info["linear1"] == [None, 64]
        assert shape_info["relu"] == [None, 64]
        assert shape_info["linear2"] == [None, 1]

    def test_build_model_with_sample_data_validation(self):
        """Test building model with sample data for shape validation."""
        yaml_content = """
version: "0.1"
model_name: "sample_shape_test"

features:
  feature_columns: [features]
  processors: []

model:
  inputs:
    features: {dtype: float32, shape: [null, vars.n_features]}
  graph:
    - name: linear1
      type: core.Linear
      params: {in_features: vars.n_features, out_features: 32}
    - name: relu
      type: core.ReLU
    - name: linear2
      type: core.Linear
      params: {in_features: 32, out_features: 10}
  outputs:
    logits: linear2.output

trainer:
  optimizer: {type: AdamW}
  loss: {type: core.CrossEntropyLoss}
"""

        graph = ArcGraph.from_yaml(yaml_content)
        builder = ModelBuilder(enable_shape_validation=True)

        # Create sample data
        sample_data = torch.randn(8, 16)  # [batch=8, features=16]

        # Build model with auto-detection
        _ = builder.build_model(graph, sample_data)

        # Check inferred shapes
        shape_info = builder.get_shape_info()
        assert shape_info["linear1"] == [8, 32]
        assert shape_info["relu"] == [8, 32]
        assert shape_info["linear2"] == [8, 10]

    def test_shape_validation_error(self):
        """Test shape validation error detection."""
        yaml_content = """
version: "0.1"
model_name: "invalid_shape_test"

features:
  feature_columns: [features]
  processors: []

model:
  inputs:
    features: {dtype: float32, shape: [null, 10]}
  graph:
    - name: linear
      type: core.Linear
      params: {in_features: 20, out_features: 5}  # Wrong input size!
  outputs:
    output: linear.output

trainer:
  optimizer: {type: AdamW}
  loss: {type: core.MSELoss}
"""

        graph = ArcGraph.from_yaml(yaml_content)
        builder = ModelBuilder(enable_shape_validation=True)

        # This should fail during shape validation due to parameter mismatch.
        # Input features (10) != linear in_features (20).
        with pytest.raises(ValueError, match="Shape validation failed"):
            builder.build_model(graph)

    def test_complex_model_shape_validation(self):
        """Test shape validation for complex model with multiple paths."""
        yaml_content = """
version: "0.1"
model_name: "complex_shape_test"

features:
  feature_columns: [input1, input2]
  processors: []

model:
  inputs:
    input1: {dtype: float32, shape: [null, 10]}
    input2: {dtype: float32, shape: [null, 15]}
  graph:
    - name: linear1
      type: core.Linear
      params: {in_features: 10, out_features: 32}
      inputs: {input: input1}
    - name: linear2
      type: core.Linear
      params: {in_features: 15, out_features: 32}
      inputs: {input: input2}
    - name: concat
      type: core.Concatenate
      params: {dim: -1}
      inputs: {path1: linear1.output, path2: linear2.output}
    - name: final_linear
      type: core.Linear
      params: {in_features: 64, out_features: 1}  # 32 + 32 = 64
      inputs: {input: concat.output}
  outputs:
    prediction: final_linear.output

trainer:
  optimizer: {type: AdamW}
  loss: {type: core.MSELoss}
"""

        graph = ArcGraph.from_yaml(yaml_content)
        builder = ModelBuilder(enable_shape_validation=True)

        # Build model
        model = builder.build_model(graph)

        # Check shape inference
        shape_info = builder.get_shape_info()
        assert shape_info["linear1"] == [None, 32]
        assert shape_info["linear2"] == [None, 32]
        assert shape_info["concat"] == [None, 64]  # 32 + 32
        assert shape_info["final_linear"] == [None, 1]

        # Test actual forward pass to verify shapes
        batch_size = 4
        input1 = torch.randn(batch_size, 10)
        input2 = torch.randn(batch_size, 15)
        inputs = {"input1": input1, "input2": input2}

        outputs = model(inputs)
        assert outputs["prediction"].shape == (batch_size, 1)

    def test_disable_shape_validation(self):
        """Test building model with shape validation disabled."""
        yaml_content = """
version: "0.1"
model_name: "no_validation_test"

features:
  feature_columns: [features]
  processors: []

model:
  inputs:
    features: {dtype: float32, shape: [null, 10]}
  graph:
    - name: linear
      type: core.Linear
      params: {in_features: 10, out_features: 5}
  outputs:
    output: linear.output

trainer:
  optimizer: {type: AdamW}
  loss: {type: core.MSELoss}
"""

        graph = ArcGraph.from_yaml(yaml_content)
        builder = ModelBuilder(enable_shape_validation=False)

        # Should build without validation
        _ = builder.build_model(graph)

        # No shape info should be available
        shape_info = builder.get_shape_info()
        assert len(shape_info) == 0

    def test_print_shape_summary(self, capsys):
        """Test shape summary printing."""
        yaml_content = """
version: "0.1"
model_name: "summary_test"

features:
  feature_columns: [features]
  processors: []

model:
  inputs:
    features: {dtype: float32, shape: [null, 5]}
  graph:
    - name: linear
      type: core.Linear
      params: {in_features: 5, out_features: 3}
  outputs:
    output: linear.output

trainer:
  optimizer: {type: AdamW}
  loss: {type: core.MSELoss}
"""

        graph = ArcGraph.from_yaml(yaml_content)
        builder = ModelBuilder(enable_shape_validation=True)
        _ = builder.build_model(graph)

        # Print shape summary
        builder.print_shape_summary()

        # Check output
        captured = capsys.readouterr()
        assert "Model Shape Summary" in captured.out
        assert "linear: [?, 3]" in captured.out

    def test_shape_validation_with_dynamic_dimensions(self):
        """Test shape validation with dynamic (None) dimensions."""
        yaml_content = """
version: "0.1"
model_name: "dynamic_test"

features:
  feature_columns: [features]
  processors: []

model:
  inputs:
    sequences: {dtype: int64, shape: [null, null]}  # Dynamic batch and sequence length
  graph:
    - name: embedding
      type: core.Embedding
      params: {num_embeddings: 1000, embedding_dim: 128}
    - name: lstm
      type: core.LSTM
      params: {input_size: 128, hidden_size: 256}
  outputs:
    features: lstm.output

trainer:
  optimizer: {type: AdamW}
  loss: {type: core.MSELoss}
"""

        graph = ArcGraph.from_yaml(yaml_content)
        builder = ModelBuilder(enable_shape_validation=True)

        # Build model
        model = builder.build_model(graph)

        # Check shape inference with dynamic dimensions
        shape_info = builder.get_shape_info()
        assert shape_info["embedding"] == [None, None, 128]  # [batch, seq, embed]
        assert shape_info["lstm"] == [None, None, 256]  # [batch, seq, hidden]

        # Test with different input sizes
        for batch_size, seq_len in [(4, 10), (8, 20), (2, 50)]:
            sequences = torch.randint(0, 1000, (batch_size, seq_len))
            inputs = {"sequences": sequences}
            outputs = model(inputs)
            assert outputs["features"].shape == (batch_size, seq_len, 256)
