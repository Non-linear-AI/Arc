"""Tests for Arc-Graph model builder functionality."""

import pytest
import torch

from arc.graph.model import ArcGraphModel, build_model_from_yaml
from arc.graph.model.validator import ModelValidationError


class TestBasicModelBuilding:
    """Test basic model building functionality."""

    def test_simple_linear_model(self):
        """Test building a simple linear regression model."""
        yaml_content = """
        inputs:
          features:
            dtype: float32
            shape: [null, 4]
            columns: [x1, x2, x3, x4]

        graph:
          - name: linear
            type: torch.nn.Linear
            params:
              in_features: 4
              out_features: 1
            inputs:
              input: features

        outputs:
          prediction: linear.output
        """

        model = build_model_from_yaml(yaml_content)
        assert isinstance(model, ArcGraphModel)
        assert model.input_names == ["features"]
        assert model.output_names == ["prediction"]

        # Test forward pass
        test_input = torch.randn(2, 4)
        output = model(features=test_input)
        assert output.shape == (2, 1)

    def test_multi_layer_classification(self):
        """Test building a multi-layer classification model."""
        yaml_content = """
        inputs:
          features:
            dtype: float32
            shape: [null, 8]
            columns: [f1, f2, f3, f4, f5, f6, f7, f8]

        graph:
          - name: hidden1
            type: torch.nn.Linear
            params:
              in_features: 8
              out_features: 16
            inputs:
              input: features

          - name: activation1
            type: torch.nn.functional.relu
            inputs: [hidden1.output]

          - name: hidden2
            type: torch.nn.Linear
            params:
              in_features: 16
              out_features: 8
            inputs:
              input: activation1

          - name: activation2
            type: torch.nn.functional.relu
            inputs: [hidden2.output]

          - name: output
            type: torch.nn.Linear
            params:
              in_features: 8
              out_features: 2
            inputs:
              input: activation2

        outputs:
          logits: output.output
        """

        model = build_model_from_yaml(yaml_content)

        # Test forward pass
        test_input = torch.randn(3, 8)
        output = model(features=test_input)
        assert output.shape == (3, 2)

    def test_multiple_inputs_outputs(self):
        """Test model with multiple inputs and outputs."""
        yaml_content = """
        inputs:
          input1:
            dtype: float32
            shape: [null, 5]
            columns: [a1, a2, a3, a4, a5]
          input2:
            dtype: float32
            shape: [null, 3]
            columns: [b1, b2, b3]

        graph:
          - name: linear1
            type: torch.nn.Linear
            params:
              in_features: 5
              out_features: 4
            inputs:
              input: input1

          - name: linear2
            type: torch.nn.Linear
            params:
              in_features: 3
              out_features: 4
            inputs:
              input: input2

          - name: concat
            type: torch.cat
            params:
              dim: 1
            inputs: [linear1.output, linear2.output]

          - name: final
            type: torch.nn.Linear
            params:
              in_features: 8
              out_features: 2
            inputs:
              input: concat

        outputs:
          combined: final.output
          separate1: linear1.output
          separate2: linear2.output
        """

        model = build_model_from_yaml(yaml_content)
        assert set(model.input_names) == {"input1", "input2"}
        assert set(model.output_names) == {"combined", "separate1", "separate2"}

        # Test forward pass
        test_input1 = torch.randn(2, 5)
        test_input2 = torch.randn(2, 3)
        outputs = model(input1=test_input1, input2=test_input2)

        assert isinstance(outputs, dict)
        assert outputs["combined"].shape == (2, 2)
        assert outputs["separate1"].shape == (2, 4)
        assert outputs["separate2"].shape == (2, 4)


class TestModulesAndStacking:
    """Test modules and arc.stack functionality."""

    def test_simple_module_definition(self):
        """Test basic module definition and usage."""
        yaml_content = """
        modules:
          MLP:
            inputs: [x]
            graph:
              - name: linear1
                type: torch.nn.Linear
                params:
                  in_features: 4
                  out_features: 8
                inputs:
                  input: x
              - name: activation
                type: torch.nn.functional.relu
                inputs: [linear1.output]
              - name: linear2
                type: torch.nn.Linear
                params:
                  in_features: 8
                  out_features: 4
                inputs:
                  input: activation
            outputs:
              output: linear2.output

        inputs:
          data:
            dtype: float32
            shape: [null, 4]
            columns: [x1, x2, x3, x4]

        graph:
          - name: mlp_layer
            type: module.MLP
            inputs:
              x: data

        outputs:
          result: mlp_layer.output
        """

        model = build_model_from_yaml(yaml_content)

        # Test forward pass
        test_input = torch.randn(2, 4)
        output = model(data=test_input)
        assert output.shape == (2, 4)

    def test_arc_stack_functionality(self):
        """Test arc.stack for deep networks."""
        yaml_content = """
        modules:
          ResidualBlock:
            inputs: [x]
            graph:
              - name: linear1
                type: torch.nn.Linear
                params:
                  in_features: 64
                  out_features: 64
                inputs:
                  input: x
              - name: activation1
                type: torch.nn.functional.relu
                inputs: [linear1.output]
              - name: linear2
                type: torch.nn.Linear
                params:
                  in_features: 64
                  out_features: 64
                inputs:
                  input: activation1
              - name: residual
                type: torch.add
                inputs: [linear2.output, x]
            outputs:
              output: residual

        inputs:
          features:
            dtype: float32
            shape: [null, 64]
            columns: []

        graph:
          - name: deep_stack
            type: arc.stack
            params:
              module: ResidualBlock
              count: 5
            inputs:
              input: features

          - name: classifier
            type: torch.nn.Linear
            params:
              in_features: 64
              out_features: 10
            inputs:
              input: deep_stack.output

        outputs:
          logits: classifier.output
        """

        model = build_model_from_yaml(yaml_content)

        # Check that the stack has 5 layers
        stack_module = model.graph_modules["deep_stack"]
        assert len(stack_module) == 5

        # Test forward pass
        test_input = torch.randn(2, 64)
        output = model(features=test_input)
        assert output.shape == (2, 10)

    def test_nested_modules(self):
        """Test modules that use other modules."""
        yaml_content = """
        modules:
          LinearBlock:
            inputs: [x]
            graph:
              - name: linear
                type: torch.nn.Linear
                params:
                  in_features: 32
                  out_features: 32
                inputs:
                  input: x
              - name: activation
                type: torch.nn.functional.gelu
                inputs: [linear.output]
            outputs:
              output: activation

          AttentionBlock:
            inputs: [x]
            graph:
              - name: attention
                type: torch.nn.MultiheadAttention
                params:
                  embed_dim: 32
                  num_heads: 4
                inputs:
                  query: x
                  key: x
                  value: x
              - name: ffn
                type: module.LinearBlock
                inputs:
                  x: attention.output.0
            outputs:
              output: ffn.output

        inputs:
          sequence:
            dtype: float32
            shape: [null, 10, 32]
            columns: []

        graph:
          - name: transformer_layers
            type: arc.stack
            params:
              module: AttentionBlock
              count: 3
            inputs:
              input: sequence

          - name: pooled
            type: torch.mean
            params:
              dim: 1
            inputs: [transformer_layers.output]

        outputs:
          representation: pooled
        """

        model = build_model_from_yaml(yaml_content)

        # Test forward pass
        test_input = torch.randn(2, 10, 32)
        output = model(sequence=test_input)
        assert output.shape == (2, 32)


class TestAdvancedFeatures:
    """Test advanced Arc-Graph features."""

    def test_list_input_format(self):
        """Test list input format for functions."""
        yaml_content = """
        inputs:
          a:
            dtype: float32
            shape: [null, 3]
            columns: [a1, a2, a3]
          b:
            dtype: float32
            shape: [null, 3]
            columns: [b1, b2, b3]
          c:
            dtype: float32
            shape: [null, 3]
            columns: [c1, c2, c3]

        graph:
          - name: stack_tensors
            type: torch.stack
            params:
              dim: 1
            inputs: [a, b, c]

          - name: flatten
            type: torch.nn.Flatten
            params:
              start_dim: 1
            inputs:
              input: stack_tensors

        outputs:
          stacked: stack_tensors
          flattened: flatten.output
        """

        model = build_model_from_yaml(yaml_content)

        # Test forward pass
        test_a = torch.randn(2, 3)
        test_b = torch.randn(2, 3)
        test_c = torch.randn(2, 3)
        outputs = model(a=test_a, b=test_b, c=test_c)

        assert outputs["stacked"].shape == (2, 3, 3)
        assert outputs["flattened"].shape == (2, 9)

    def test_tuple_output_indexing(self):
        """Test tuple output indexing (.output.0, .output.1)."""
        yaml_content = """
        inputs:
          sequence:
            dtype: float32
            shape: [null, 10, 64]
            columns: []

        graph:
          - name: lstm
            type: torch.nn.LSTM
            params:
              input_size: 64
              hidden_size: 32
              batch_first: true
            inputs:
              input: sequence

          - name: squeeze_hidden
            type: torch.squeeze
            params:
              dim: 0
            inputs: [lstm.output.1.0]

          - name: last_hidden
            type: torch.nn.Linear
            params:
              in_features: 32
              out_features: 10
            inputs:
              input: squeeze_hidden

        outputs:
          sequence_output: lstm.output.0
          final_hidden: lstm.output.1.0
          final_cell: lstm.output.1.1
          classification: last_hidden.output
        """

        model = build_model_from_yaml(yaml_content)

        # Test forward pass
        test_input = torch.randn(2, 10, 64)
        outputs = model(sequence=test_input)

        assert outputs["sequence_output"].shape == (2, 10, 32)
        assert outputs["final_hidden"].shape == (
            1,
            2,
            32,
        )  # LSTM hidden has shape (num_layers, batch, hidden_size)
        assert outputs["final_cell"].shape == (1, 2, 32)
        assert outputs["classification"].shape == (2, 10)

    def test_tensor_operations(self):
        """Test various tensor operations."""
        yaml_content = """
        inputs:
          matrix_a:
            dtype: float32
            shape: [null, 10, 5]
            columns: []
          matrix_b:
            dtype: float32
            shape: [null, 5, 8]
            columns: []

        graph:
          - name: matmul
            type: torch.matmul
            inputs: [matrix_a, matrix_b]

          - name: reshaped
            type: torch.reshape
            params:
              shape: [-1, 80]
            inputs: [matmul]

          - name: summed
            type: torch.sum
            params:
              dim: 1
              keepdim: true
            inputs: [reshaped]

        outputs:
          matrix_product: matmul
          reshaped_result: reshaped
          sum_result: summed
        """

        model = build_model_from_yaml(yaml_content)

        # Test forward pass
        test_a = torch.randn(2, 10, 5)
        test_b = torch.randn(2, 5, 8)
        outputs = model(matrix_a=test_a, matrix_b=test_b)

        assert outputs["matrix_product"].shape == (2, 10, 8)
        assert outputs["reshaped_result"].shape == (2, 80)
        assert outputs["sum_result"].shape == (2, 1)


class TestErrorHandling:
    """Test error handling and validation."""

    def test_invalid_module_reference(self):
        """Test error when referencing non-existent module."""
        yaml_content = """
        inputs:
          data:
            dtype: float32
            shape: [null, 4]
            columns: [x1, x2, x3, x4]

        graph:
          - name: invalid_module
            type: module.NonExistentModule
            inputs:
              x: data

        outputs:
          result: invalid_module.output
        """

        with pytest.raises(ValueError, match="Referenced module.*not found"):
            build_model_from_yaml(yaml_content)

    def test_invalid_arc_stack_params(self):
        """Test error when arc.stack has invalid parameters."""
        yaml_content = """
        modules:
          TestModule:
            inputs: [x]
            graph:
              - name: linear
                type: torch.nn.Linear
                params:
                  in_features: 4
                  out_features: 4
                inputs:
                  input: x
            outputs:
              output: linear.output

        inputs:
          data:
            dtype: float32
            shape: [null, 4]
            columns: [x1, x2, x3, x4]

        graph:
          - name: invalid_stack
            type: arc.stack
            params:
              module: TestModule
              count: 0  # Invalid count
            inputs:
              input: data

        outputs:
          result: invalid_stack.output
        """

        with pytest.raises(
            ModelValidationError, match="count must be a positive integer"
        ):
            build_model_from_yaml(yaml_content)

    def test_missing_required_inputs(self):
        """Test error when model inputs are missing during forward pass."""
        yaml_content = """
        inputs:
          required_input:
            dtype: float32
            shape: [null, 4]
            columns: [x1, x2, x3, x4]

        graph:
          - name: linear
            type: torch.nn.Linear
            params:
              in_features: 4
              out_features: 1
            inputs:
              input: required_input

        outputs:
          result: linear.output
        """

        model = build_model_from_yaml(yaml_content)

        with pytest.raises(ValueError, match="Missing required input"):
            model()  # Missing required_input


class TestRealWorldExamples:
    """Test real-world model architectures."""

    def test_convolutional_neural_network(self):
        """Test CNN for image classification."""
        yaml_content = """
        inputs:
          image:
            dtype: float32
            shape: [null, 3, 32, 32]
            columns: []

        graph:
          - name: conv1
            type: torch.nn.Conv2d
            params:
              in_channels: 3
              out_channels: 32
              kernel_size: 3
              padding: 1
            inputs:
              input: image

          - name: relu1
            type: torch.nn.functional.relu
            inputs: [conv1.output]

          - name: pool1
            type: torch.nn.functional.max_pool2d
            params:
              kernel_size: 2
            inputs: [relu1]

          - name: conv2
            type: torch.nn.Conv2d
            params:
              in_channels: 32
              out_channels: 64
              kernel_size: 3
              padding: 1
            inputs:
              input: pool1

          - name: relu2
            type: torch.nn.functional.relu
            inputs: [conv2.output]

          - name: pool2
            type: torch.nn.functional.max_pool2d
            params:
              kernel_size: 2
            inputs: [relu2]

          - name: flatten
            type: torch.nn.Flatten
            inputs:
              input: pool2

          - name: classifier
            type: torch.nn.Linear
            params:
              in_features: 4096  # 64 * 8 * 8
              out_features: 10
            inputs:
              input: flatten.output

        outputs:
          logits: classifier.output
        """

        model = build_model_from_yaml(yaml_content)

        # Test forward pass
        test_input = torch.randn(2, 3, 32, 32)
        output = model(image=test_input)
        assert output.shape == (2, 10)

    def test_transformer_encoder(self):
        """Test transformer encoder architecture."""
        yaml_content = """
        inputs:
          tokens:
            dtype: float32
            shape: [null, 100, 512]  # sequence_length=100, d_model=512
            columns: []

        graph:
          - name: encoder_layer1
            type: torch.nn.TransformerEncoderLayer
            params:
              d_model: 512
              nhead: 8
              dim_feedforward: 2048
              batch_first: true
            inputs:
              src: tokens

          - name: encoder_layer2
            type: torch.nn.TransformerEncoderLayer
            params:
              d_model: 512
              nhead: 8
              dim_feedforward: 2048
              batch_first: true
            inputs:
              src: encoder_layer1.output

          - name: pooled
            type: torch.mean
            params:
              dim: 1
            inputs: [encoder_layer2.output]

          - name: classifier
            type: torch.nn.Linear
            params:
              in_features: 512
              out_features: 2
            inputs:
              input: pooled

        outputs:
          sequence_representation: encoder_layer2.output
          classification_logits: classifier.output
        """

        model = build_model_from_yaml(yaml_content)

        # Test forward pass
        test_input = torch.randn(2, 100, 512)
        outputs = model(tokens=test_input)

        assert outputs["sequence_representation"].shape == (2, 100, 512)
        assert outputs["classification_logits"].shape == (2, 2)
