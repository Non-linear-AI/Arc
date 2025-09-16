"""Tests for transformer/attention model examples."""

import torch

from src.arc.graph import ArcGraph
from src.arc.ml.builder import ModelBuilder


class TestTransformerExample:
    """Test complex transformer model examples."""

    def test_simple_transformer_encoder(self):
        """Test a simple transformer encoder model."""
        yaml_content = """
version: "0.1"
model_name: "simple_transformer"

features:
  feature_columns: [token_ids]
  processors: []

model:
  inputs:
    token_ids: {dtype: int64, shape: [null, vars.seq_len]}
  graph:
    - name: embedding
      type: core.Embedding
      params: {num_embeddings: vars.vocab_size, embedding_dim: vars.d_model}
      inputs: {input: token_ids}
    - name: pos_encoding
      type: core.PositionalEncoding
      params: {d_model: vars.d_model, max_len: vars.seq_len, dropout: 0.1}
      inputs: {input: embedding.output}
    - name: transformer_layer
      type: core.TransformerEncoderLayer
      params:
        d_model: vars.d_model
        nhead: vars.num_heads
        dim_feedforward: vars.ff_dim
      inputs: {input: pos_encoding.output}
    - name: layer_norm
      type: core.LayerNorm
      params: {normalized_shape: vars.d_model}
      inputs: {input: transformer_layer.output}
    - name: output_projection
      type: core.Linear
      params: {in_features: vars.d_model, out_features: vars.num_classes}
      inputs: {input: layer_norm.output}
  outputs:
    logits: output_projection.output

trainer:
  optimizer: {type: AdamW}
  loss: {type: core.CrossEntropyLoss}
"""

        graph = ArcGraph.from_yaml(yaml_content)
        builder = ModelBuilder()

        # Set transformer parameters
        builder.set_variable("vars.vocab_size", 1000)
        builder.set_variable("vars.d_model", 128)
        builder.set_variable("vars.seq_len", 32)
        builder.set_variable("vars.num_heads", 8)
        builder.set_variable("vars.ff_dim", 512)
        builder.set_variable("vars.num_classes", 10)

        model = builder.build_model(graph)

        # Test forward pass
        batch_size = 4
        seq_len = 32
        token_ids = torch.randint(0, 1000, (batch_size, seq_len))
        inputs = {"token_ids": token_ids}

        outputs = model(inputs)

        assert "logits" in outputs
        assert outputs["logits"].shape == (batch_size, seq_len, 10)

    def test_multi_input_attention_model(self):
        """Test transformer with explicit query, key, value inputs."""
        yaml_content = """
version: "0.1"
model_name: "multi_input_attention"

features:
  feature_columns: [source_tokens, target_tokens]
  processors: []

model:
  inputs:
    source_tokens: {dtype: int64, shape: [null, vars.src_len]}
    target_tokens: {dtype: int64, shape: [null, vars.tgt_len]}
  graph:
    # Source embeddings (for keys and values)
    - name: src_embedding
      type: core.Embedding
      params: {num_embeddings: vars.vocab_size, embedding_dim: vars.d_model}
      inputs: {input: source_tokens}

    # Target embeddings (for queries)
    - name: tgt_embedding
      type: core.Embedding
      params: {num_embeddings: vars.vocab_size, embedding_dim: vars.d_model}
      inputs: {input: target_tokens}

    # Cross attention: target queries attend to source keys/values
    - name: cross_attention
      type: core.MultiHeadAttention
      params: {embed_dim: vars.d_model, num_heads: vars.num_heads}
      inputs:
        query: tgt_embedding.output
        key: src_embedding.output
        value: src_embedding.output

    - name: output_layer
      type: core.Linear
      params: {in_features: vars.d_model, out_features: vars.vocab_size}
      inputs: {input: cross_attention.output}
  outputs:
    predictions: output_layer.output

trainer:
  optimizer: {type: AdamW}
  loss: {type: core.CrossEntropyLoss}
"""

        graph = ArcGraph.from_yaml(yaml_content)
        builder = ModelBuilder()

        # Set parameters
        builder.set_variable("vars.vocab_size", 500)
        builder.set_variable("vars.d_model", 64)
        builder.set_variable("vars.src_len", 20)
        builder.set_variable("vars.tgt_len", 15)
        builder.set_variable("vars.num_heads", 4)

        model = builder.build_model(graph)

        # Test forward pass
        batch_size = 3
        source_tokens = torch.randint(0, 500, (batch_size, 20))
        target_tokens = torch.randint(0, 500, (batch_size, 15))
        inputs = {"source_tokens": source_tokens, "target_tokens": target_tokens}

        outputs = model(inputs)

        assert "predictions" in outputs
        assert outputs["predictions"].shape == (batch_size, 15, 500)

    def test_complex_routing_model(self):
        """Test complex model with tensor routing and concatenation."""
        yaml_content = """
version: "0.1"
model_name: "complex_routing"

features:
  feature_columns: [numerical_features, categorical_features]
  processors: []

model:
  inputs:
    numerical_features: {dtype: float32, shape: [null, vars.num_features]}
    # Simplified to float
    categorical_features: {dtype: float32, shape: [null, vars.cat_features]}
  graph:
    # Process numerical features
    - name: num_linear1
      type: core.Linear
      params: {in_features: vars.num_features, out_features: 64}
      inputs: {input: numerical_features}
    - name: num_relu1
      type: core.ReLU
      inputs: {input: num_linear1.output}
    - name: num_linear2
      type: core.Linear
      params: {in_features: 64, out_features: 32}
      inputs: {input: num_relu1.output}

    # Process categorical features (simplified)
    - name: cat_linear1
      type: core.Linear
      params: {in_features: vars.cat_features, out_features: 16}
      inputs: {input: categorical_features}
    - name: cat_relu
      type: core.ReLU
      inputs: {input: cat_linear1.output}
    - name: cat_linear2
      type: core.Linear
      params: {in_features: 16, out_features: 32}
      inputs: {input: cat_relu.output}

    # Combine features
    - name: feature_concat
      type: core.Concatenate
      params: {dim: -1}
      inputs: {num_path: num_linear2.output, cat_path: cat_linear2.output}

    # Final processing
    - name: combined_linear
      type: core.Linear
      params: {in_features: 64, out_features: 16}  # 32 + 32 = 64
      inputs: {input: feature_concat.output}
    - name: final_relu
      type: core.ReLU
      inputs: {input: combined_linear.output}
    - name: output_layer
      type: core.Linear
      params: {in_features: 16, out_features: 1}
      inputs: {input: final_relu.output}
  outputs:
    prediction: output_layer.output

trainer:
  optimizer: {type: AdamW}
  loss: {type: core.MSELoss}
"""

        graph = ArcGraph.from_yaml(yaml_content)
        builder = ModelBuilder()

        # Set parameters
        builder.set_variable("vars.num_features", 10)
        builder.set_variable("vars.cat_features", 5)

        model = builder.build_model(graph)

        # Test forward pass
        batch_size = 6
        numerical_features = torch.randn(batch_size, 10)
        categorical_features = torch.randn(batch_size, 5)  # Simplified to float
        inputs = {
            "numerical_features": numerical_features,
            "categorical_features": categorical_features,
        }

        outputs = model(inputs)

        assert "prediction" in outputs
        assert outputs["prediction"].shape == (batch_size, 1)

    def test_residual_connections_with_add_layer(self):
        """Test residual connections using Add layer."""
        yaml_content = """
version: "0.1"
model_name: "residual_model"

features:
  feature_columns: [features]
  processors: []

model:
  inputs:
    features: {dtype: float32, shape: [null, vars.feature_dim]}
  graph:
    # First residual block
    - name: linear1
      type: core.Linear
      params: {in_features: vars.feature_dim, out_features: vars.feature_dim}
      inputs: {input: features}
    - name: relu1
      type: core.ReLU
      inputs: {input: linear1.output}
    - name: linear2
      type: core.Linear
      params: {in_features: vars.feature_dim, out_features: vars.feature_dim}
      inputs: {input: relu1.output}

    # Residual connection
    - name: residual1
      type: core.Add
      inputs: {original: features, transformed: linear2.output}

    # Second residual block
    - name: linear3
      type: core.Linear
      params: {in_features: vars.feature_dim, out_features: vars.feature_dim}
      inputs: {input: residual1.output}
    - name: relu2
      type: core.ReLU
      inputs: {input: linear3.output}
    - name: linear4
      type: core.Linear
      params: {in_features: vars.feature_dim, out_features: vars.feature_dim}
      inputs: {input: relu2.output}

    # Second residual connection
    - name: residual2
      type: core.Add
      inputs: {previous: residual1.output, current: linear4.output}

    # Output layer
    - name: output_layer
      type: core.Linear
      params: {in_features: vars.feature_dim, out_features: vars.num_classes}
      inputs: {input: residual2.output}
  outputs:
    logits: output_layer.output

trainer:
  optimizer: {type: AdamW}
  loss: {type: core.CrossEntropyLoss}
"""

        graph = ArcGraph.from_yaml(yaml_content)
        builder = ModelBuilder()

        # Set parameters
        builder.set_variable("vars.feature_dim", 128)
        builder.set_variable("vars.num_classes", 5)

        model = builder.build_model(graph)

        # Test forward pass
        batch_size = 4
        features = torch.randn(batch_size, 128)
        inputs = {"features": features}

        outputs = model(inputs)

        assert "logits" in outputs
        assert outputs["logits"].shape == (batch_size, 5)

    def test_execution_order_complex_dependencies(self):
        """Test that complex dependency graphs execute in correct order."""
        yaml_content = """
version: "0.1"
model_name: "complex_dependencies"

features:
  feature_columns: [input]
  processors: []

model:
  inputs:
    input: {dtype: float32, shape: [null, 4]}
  graph:
    # Define in intentionally confusing order
    - name: final_output
      type: core.Linear
      params: {in_features: 8, out_features: 1}
      inputs: {input: concat_layer.output}

    - name: branch_b
      type: core.Linear
      params: {in_features: 4, out_features: 4}
      inputs: {input: input}

    - name: concat_layer
      type: core.Concatenate
      params: {dim: -1}
      inputs: {path_a: branch_a_relu.output, path_b: branch_b_relu.output}

    - name: branch_a_relu
      type: core.ReLU
      inputs: {input: branch_a.output}

    - name: branch_b_relu
      type: core.ReLU
      inputs: {input: branch_b.output}

    - name: branch_a
      type: core.Linear
      params: {in_features: 4, out_features: 4}
      inputs: {input: input}
  outputs:
    result: final_output.output

trainer:
  optimizer: {type: AdamW}
  loss: {type: core.MSELoss}
"""

        graph = ArcGraph.from_yaml(yaml_content)
        builder = ModelBuilder()
        model = builder.build_model(graph)

        # Check execution order is correct
        expected_dependencies = {
            "branch_a": set(),
            "branch_b": set(),
            "branch_a_relu": {"branch_a"},
            "branch_b_relu": {"branch_b"},
            "concat_layer": {"branch_a_relu", "branch_b_relu"},
            "final_output": {"concat_layer"},
        }

        # Verify topological ordering
        executed = set()
        for layer_name in model.execution_order:
            # All dependencies should be executed before this layer
            deps = expected_dependencies[layer_name]
            assert deps.issubset(executed), (
                f"Dependencies {deps - executed} not executed before {layer_name}"
            )
            executed.add(layer_name)

        # Test forward pass works
        batch_size = 3
        input_tensor = torch.randn(batch_size, 4)
        inputs = {"input": input_tensor}

        outputs = model(inputs)
        assert "result" in outputs
        assert outputs["result"].shape == (batch_size, 1)
