"""Tests for ModelSpec in the separated architecture."""

import pytest

from arc.graph.model import GraphNode, ModelInput, ModelSpec, validate_model_dict
from arc.graph.model.validator import ModelValidationError


class TestModelInput:
    """Test ModelInput dataclass."""

    def test_model_input_creation(self):
        """Test basic ModelInput creation."""
        input_spec = ModelInput(
            dtype="float32",
            shape=[None, 10],
            columns=["feature1", "feature2", "feature3"],
        )

        assert input_spec.dtype == "float32"
        assert input_spec.shape == [None, 10]
        assert input_spec.columns == ["feature1", "feature2", "feature3"]

    def test_model_input_optional_columns(self):
        """Test ModelInput with optional columns."""
        input_spec = ModelInput(dtype="int32", shape=[None])

        assert input_spec.dtype == "int32"
        assert input_spec.shape == [None]
        assert input_spec.columns is None

    def test_model_input_with_embedding(self):
        """Test ModelInput with embedding parameters for categorical features."""
        input_spec = ModelInput(
            dtype="long",
            shape=[None],
            embedding_dim=32,
            vocab_size=100,
            categorical=True,
        )

        assert input_spec.dtype == "long"
        assert input_spec.shape == [None]
        assert input_spec.embedding_dim == 32
        assert input_spec.vocab_size == 100
        assert input_spec.categorical is True

    def test_model_input_embedding_defaults(self):
        """Test ModelInput with embedding fields defaulting to None."""
        input_spec = ModelInput(dtype="float32", shape=[None, 10])

        assert input_spec.embedding_dim is None
        assert input_spec.vocab_size is None
        assert input_spec.categorical is False  # Default to False

    def test_model_input_partial_embedding_spec(self):
        """Test ModelInput with only some embedding parameters."""
        # Categorical flag alone
        input_spec1 = ModelInput(
            dtype="long",
            shape=[None],
            categorical=True,
        )
        assert input_spec1.categorical is True
        assert input_spec1.embedding_dim is None

        # With embedding dim but no vocab size
        input_spec2 = ModelInput(
            dtype="long",
            shape=[None],
            embedding_dim=64,
            categorical=True,
        )
        assert input_spec2.embedding_dim == 64
        assert input_spec2.vocab_size is None


class TestGraphNode:
    """Test GraphNode dataclass."""

    def test_graph_node_creation(self):
        """Test basic GraphNode creation."""
        node = GraphNode(
            name="linear1",
            type="torch.nn.Linear",
            params={"in_features": 10, "out_features": 5},
            inputs={"input": "data"},
        )

        assert node.name == "linear1"
        assert node.type == "torch.nn.Linear"
        assert node.params == {"in_features": 10, "out_features": 5}
        assert node.inputs == {"input": "data"}

    def test_graph_node_minimal(self):
        """Test GraphNode with minimal parameters."""
        node = GraphNode(name="relu1", type="torch.nn.ReLU")

        assert node.name == "relu1"
        assert node.type == "torch.nn.ReLU"
        assert node.params is None
        assert node.inputs is None


class TestModelSpec:
    """Test ModelSpec functionality."""

    @pytest.fixture
    def sample_model_dict(self):
        """Sample model dictionary for testing."""
        return {
            "inputs": {
                "patient_data": {
                    "dtype": "float32",
                    "shape": [None, 8],
                    "columns": [
                        "pregnancies",
                        "glucose",
                        "blood_pressure",
                        "skin_thickness",
                        "insulin",
                        "bmi",
                        "diabetes_pedigree",
                        "age",
                    ],
                }
            },
            "graph": [
                {
                    "name": "classifier",
                    "type": "torch.nn.Linear",
                    "params": {"in_features": 8, "out_features": 1, "bias": True},
                    "inputs": {"input": "patient_data"},
                },
                {
                    "name": "sigmoid",
                    "type": "torch.nn.Sigmoid",
                    "inputs": {"input": "classifier.output"},
                },
            ],
            "outputs": {"logits": "classifier.output", "prediction": "sigmoid.output"},
        }

    @pytest.fixture
    def sample_model_yaml(self, sample_model_dict):
        """Sample model YAML string."""
        import yaml

        return yaml.dump(sample_model_dict)

    def test_model_spec_from_yaml(self, sample_model_yaml):
        """Test ModelSpec creation from YAML."""
        model_spec = ModelSpec.from_yaml(sample_model_yaml)

        assert len(model_spec.inputs) == 1
        assert "patient_data" in model_spec.inputs
        assert model_spec.inputs["patient_data"].dtype == "float32"
        assert model_spec.inputs["patient_data"].shape == [None, 8]
        assert len(model_spec.inputs["patient_data"].columns) == 8

        assert len(model_spec.graph) == 2
        assert model_spec.graph[0].name == "classifier"
        assert model_spec.graph[0].type == "torch.nn.Linear"
        assert model_spec.graph[1].name == "sigmoid"
        assert model_spec.graph[1].type == "torch.nn.Sigmoid"

        assert len(model_spec.outputs) == 2
        assert model_spec.outputs["logits"] == "classifier.output"
        assert model_spec.outputs["prediction"] == "sigmoid.output"

    def test_model_spec_to_yaml(self, sample_model_yaml):
        """Test ModelSpec conversion to YAML."""
        model_spec = ModelSpec.from_yaml(sample_model_yaml)
        regenerated_yaml = model_spec.to_yaml()

        import yaml

        regenerated_dict = yaml.safe_load(regenerated_yaml)

        # Check structure is preserved
        assert regenerated_dict["inputs"]["patient_data"]["dtype"] == "float32"
        assert len(regenerated_dict["graph"]) == 2
        assert len(regenerated_dict["outputs"]) == 2

    def test_model_spec_helper_methods(self, sample_model_yaml):
        """Test ModelSpec helper methods."""
        model_spec = ModelSpec.from_yaml(sample_model_yaml)

        input_names = model_spec.get_input_names()
        assert input_names == ["patient_data"]

        output_names = model_spec.get_output_names()
        assert set(output_names) == {"logits", "prediction"}

        layer_names = model_spec.get_layer_names()
        assert layer_names == ["classifier", "sigmoid"]

        layer_types = model_spec.get_layer_types()
        assert layer_types == {
            "classifier": "torch.nn.Linear",
            "sigmoid": "torch.nn.Sigmoid",
        }

    def test_invalid_yaml_structure(self):
        """Test ModelSpec with invalid YAML structure."""
        invalid_yaml = "- not_a_dict\n- but_a_list"

        with pytest.raises(ValueError, match="Top-level YAML must be a mapping"):
            ModelSpec.from_yaml(invalid_yaml)

    def test_missing_required_fields(self):
        """Test ModelSpec with missing required fields."""
        incomplete_yaml = """
        inputs:
          data:
            dtype: float32
            shape: [null, 10]
        # Missing graph and outputs
        """

        with pytest.raises(ModelValidationError):
            ModelSpec.from_yaml(incomplete_yaml)


class TestModelValidation:
    """Test model validation functionality."""

    def test_validate_model_dict_valid(self):
        """Test validation of valid model dictionary."""
        valid_dict = {
            "inputs": {"data": {"dtype": "float32", "shape": [None, 10]}},
            "graph": [
                {
                    "name": "linear",
                    "type": "torch.nn.Linear",
                    "params": {"in_features": 10, "out_features": 1},
                    "inputs": {"input": "data"},
                }
            ],
            "outputs": {"result": "linear.output"},
        }

        # Should not raise any exceptions
        validate_model_dict(valid_dict)

    def test_validate_model_dict_missing_inputs(self):
        """Test validation with missing inputs section."""
        invalid_dict = {"graph": [], "outputs": {}}

        with pytest.raises(ModelValidationError, match="model.inputs section required"):
            validate_model_dict(invalid_dict)

    def test_validate_model_dict_invalid_node_reference(self):
        """Test validation with invalid node reference."""
        invalid_dict = {
            "inputs": {"data": {"dtype": "float32", "shape": [None, 10]}},
            "graph": [
                {
                    "name": "linear",
                    "type": "torch.nn.Linear",
                    "inputs": {"input": "nonexistent_node"},
                }
            ],
            "outputs": {"result": "linear.output"},
        }

        with pytest.raises(ModelValidationError, match="references undefined node"):
            validate_model_dict(invalid_dict)

    def test_validate_model_dict_invalid_output_reference(self):
        """Test validation with invalid output reference."""
        invalid_dict = {
            "inputs": {"data": {"dtype": "float32", "shape": [None, 10]}},
            "graph": [
                {
                    "name": "linear",
                    "type": "torch.nn.Linear",
                    "inputs": {"input": "data"},
                }
            ],
            "outputs": {"result": "nonexistent_node.output"},
        }

        with pytest.raises(ModelValidationError, match="references undefined node"):
            validate_model_dict(invalid_dict)


class TestArcGraphV1Features:
    """Test Arc-Graph v1.0 specific features."""

    def test_modules_section_parsing(self):
        """Test parsing of modules section."""
        yaml_content = """
        modules:
          TestModule:
            inputs: [x, y]
            graph:
              - name: add
                type: torch.add
                inputs: [x, y]
            outputs:
              result: add

        inputs:
          a:
            dtype: float32
            shape: [null, 5]
          b:
            dtype: float32
            shape: [null, 5]

        graph:
          - name: custom_op
            type: module.TestModule
            inputs:
              x: a
              y: b

        outputs:
          sum: custom_op.result
        """

        model_spec = ModelSpec.from_yaml(yaml_content)

        # Check modules were parsed correctly
        assert model_spec.modules is not None
        assert "TestModule" in model_spec.modules

        test_module = model_spec.modules["TestModule"]
        assert test_module.inputs == ["x", "y"]
        assert len(test_module.graph) == 1
        assert test_module.graph[0].name == "add"
        assert test_module.outputs == {"result": "add"}

        # Check main graph references the module
        assert model_spec.graph[0].type == "module.TestModule"

    def test_list_inputs_in_graph_nodes(self):
        """Test list input format in graph nodes."""
        yaml_content = """
        inputs:
          tensor1:
            dtype: float32
            shape: [null, 3]
          tensor2:
            dtype: float32
            shape: [null, 3]

        graph:
          - name: concatenate
            type: torch.cat
            params:
              dim: 1
            inputs: [tensor1, tensor2]

        outputs:
          combined: concatenate
        """

        model_spec = ModelSpec.from_yaml(yaml_content)

        # Check that inputs were parsed as list
        concat_node = model_spec.graph[0]
        assert isinstance(concat_node.inputs, list)
        assert concat_node.inputs == ["tensor1", "tensor2"]

    def test_arc_stack_node_type(self):
        """Test arc.stack node type parsing."""
        yaml_content = """
        modules:
          Block:
            inputs: [x]
            graph:
              - name: linear
                type: torch.nn.Linear
                params:
                  in_features: 10
                  out_features: 10
                inputs:
                  input: x
            outputs:
              output: linear.output

        inputs:
          data:
            dtype: float32
            shape: [null, 10]

        graph:
          - name: deep_stack
            type: arc.stack
            params:
              module: Block
              count: 5
            inputs:
              input: data

        outputs:
          result: deep_stack.output
        """

        model_spec = ModelSpec.from_yaml(yaml_content)

        # Check arc.stack node was parsed correctly
        stack_node = model_spec.graph[0]
        assert stack_node.type == "arc.stack"
        assert stack_node.params["module"] == "Block"
        assert stack_node.params["count"] == 5

    def test_pytorch_function_nodes(self):
        """Test PyTorch function nodes."""
        yaml_content = """
        inputs:
          data:
            dtype: float32
            shape: [null, 10, 20]

        graph:
          - name: mean_pooling
            type: torch.mean
            params:
              dim: 1
              keepdim: true
            inputs: [data]

          - name: squeeze_result
            type: torch.squeeze
            params:
              dim: 1
            inputs: [mean_pooling]

          - name: relu_activation
            type: torch.nn.functional.relu
            inputs: [squeeze_result]

        outputs:
          pooled: mean_pooling
          squeezed: squeeze_result
          activated: relu_activation
        """

        model_spec = ModelSpec.from_yaml(yaml_content)

        # Check that function nodes were parsed correctly
        assert model_spec.graph[0].type == "torch.mean"
        assert model_spec.graph[1].type == "torch.squeeze"
        assert model_spec.graph[2].type == "torch.nn.functional.relu"

    def test_tuple_output_references(self):
        """Test tuple output reference parsing (.output.0, .output.1)."""
        yaml_content = """
        inputs:
          sequence:
            dtype: float32
            shape: [null, 10, 64]

        graph:
          - name: lstm
            type: torch.nn.LSTM
            params:
              input_size: 64
              hidden_size: 32
              batch_first: true
            inputs:
              input: sequence

          - name: process_hidden
            type: torch.nn.Linear
            params:
              in_features: 32
              out_features: 10
            inputs:
              input: lstm.output.1.0

        outputs:
          sequence_out: lstm.output.0
          hidden_state: lstm.output.1.0
          cell_state: lstm.output.1.1
          processed: process_hidden.output
        """

        model_spec = ModelSpec.from_yaml(yaml_content)

        # Check that tuple references were parsed correctly
        assert model_spec.outputs["sequence_out"] == "lstm.output.0"
        assert model_spec.outputs["hidden_state"] == "lstm.output.1.0"
        assert model_spec.outputs["cell_state"] == "lstm.output.1.1"

        # Check that graph node also uses tuple reference
        assert model_spec.graph[1].inputs["input"] == "lstm.output.1.0"

    def test_backward_compatibility_modules_none(self):
        """Test backward compatibility with modules: null."""
        yaml_content = """
        inputs:
          data:
            dtype: float32
            shape: [null, 10]

        modules: null

        graph:
          - name: linear
            type: torch.nn.Linear
            params:
              in_features: 10
              out_features: 5
            inputs:
              input: data

        outputs:
          result: linear.output
        """

        # Should not raise validation error
        model_spec = ModelSpec.from_yaml(yaml_content)
        assert model_spec.modules is None

    def test_complex_nested_modules(self):
        """Test complex nested module definitions."""
        yaml_content = """
        modules:
          Attention:
            inputs: [query, key, value]
            graph:
              - name: attention
                type: torch.nn.MultiheadAttention
                params:
                  embed_dim: 64
                  num_heads: 8
                inputs:
                  query: query
                  key: key
                  value: value
            outputs:
              attended: attention.output.0
              weights: attention.output.1

          FeedForward:
            inputs: [x]
            graph:
              - name: linear1
                type: torch.nn.Linear
                params:
                  in_features: 64
                  out_features: 256
                inputs:
                  input: x
              - name: activation
                type: torch.nn.functional.gelu
                inputs: [linear1.output]
              - name: linear2
                type: torch.nn.Linear
                params:
                  in_features: 256
                  out_features: 64
                inputs:
                  input: activation
            outputs:
              output: linear2.output

          TransformerBlock:
            inputs: [x]
            graph:
              - name: self_attention
                type: module.Attention
                inputs:
                  query: x
                  key: x
                  value: x
              - name: add1
                type: torch.add
                inputs: [self_attention.attended, x]
              - name: feedforward
                type: module.FeedForward
                inputs:
                  x: add1
              - name: add2
                type: torch.add
                inputs: [feedforward.output, add1]
            outputs:
              output: add2

        inputs:
          embeddings:
            dtype: float32
            shape: [null, 100, 64]

        graph:
          - name: transformer_layers
            type: arc.stack
            params:
              module: TransformerBlock
              count: 6
            inputs:
              input: embeddings

        outputs:
          encoded: transformer_layers.output
        """

        model_spec = ModelSpec.from_yaml(yaml_content)

        # Verify complex module structure
        assert len(model_spec.modules) == 3
        assert "Attention" in model_spec.modules
        assert "FeedForward" in model_spec.modules
        assert "TransformerBlock" in model_spec.modules

        # Check TransformerBlock uses other modules
        transformer_block = model_spec.modules["TransformerBlock"]
        attention_node = next(
            n for n in transformer_block.graph if n.name == "self_attention"
        )
        assert attention_node.type == "module.Attention"

        ff_node = next(n for n in transformer_block.graph if n.name == "feedforward")
        assert ff_node.type == "module.FeedForward"

        # Check main graph uses arc.stack with the complex module
        stack_node = model_spec.graph[0]
        assert stack_node.type == "arc.stack"
        assert stack_node.params["module"] == "TransformerBlock"
        assert stack_node.params["count"] == 6


class TestModelSpecValidation:
    """Test enhanced model validation for v1.0 features."""

    def test_validate_invalid_module_reference_in_arc_stack(self):
        """Test validation error for invalid module reference in arc.stack."""
        invalid_dict = {
            "inputs": {"data": {"dtype": "float32", "shape": [None, 10]}},
            "modules": {},
            "graph": [
                {
                    "name": "stack",
                    "type": "arc.stack",
                    "params": {"module": "NonExistentModule", "count": 3},
                    "inputs": {"input": "data"},
                }
            ],
            "outputs": {"result": "stack.output"},
        }

        with pytest.raises(ModelValidationError, match="references undefined module"):
            validate_model_dict(invalid_dict)

    def test_validate_arc_stack_invalid_count(self):
        """Test validation error for invalid count in arc.stack."""
        invalid_dict = {
            "inputs": {"data": {"dtype": "float32", "shape": [None, 10]}},
            "modules": {
                "TestModule": {
                    "inputs": ["x"],
                    "graph": [
                        {
                            "name": "linear",
                            "type": "torch.nn.Linear",
                            "params": {"in_features": 10, "out_features": 10},
                            "inputs": {"input": "x"},
                        }
                    ],
                    "outputs": {"output": "linear.output"},
                }
            },
            "graph": [
                {
                    "name": "stack",
                    "type": "arc.stack",
                    "params": {
                        "module": "TestModule",
                        "count": -1,  # Invalid negative count
                    },
                    "inputs": {"input": "data"},
                }
            ],
            "outputs": {"result": "stack.output"},
        }

        with pytest.raises(
            ModelValidationError, match="count must be a positive integer"
        ):
            validate_model_dict(invalid_dict)

    def test_validate_module_internal_references(self):
        """Test validation of internal module references."""
        invalid_dict = {
            "inputs": {"data": {"dtype": "float32", "shape": [None, 10]}},
            "modules": {
                "BadModule": {
                    "inputs": ["x"],
                    "graph": [
                        {
                            "name": "linear",
                            "type": "torch.nn.Linear",
                            "params": {"in_features": 10, "out_features": 5},
                            "inputs": {"input": "undefined_input"},  # Invalid reference
                        }
                    ],
                    "outputs": {"output": "linear.output"},
                }
            },
            "graph": [
                {
                    "name": "module_instance",
                    "type": "module.BadModule",
                    "inputs": {"x": "data"},
                }
            ],
            "outputs": {"result": "module_instance.output"},
        }

        with pytest.raises(ModelValidationError, match="references undefined node"):
            validate_model_dict(invalid_dict)


class TestModelInputEmbedding:
    """Test ModelInput embedding support for categorical features."""

    def test_parse_yaml_with_categorical_input(self):
        """Test parsing YAML with categorical input and embedding spec."""
        yaml_content = """
        inputs:
          user_id:
            dtype: long
            shape: [null]
            categorical: true
            embedding_dim: 32
            vocab_size: 1000
          features:
            dtype: float32
            shape: [null, 10]

        graph:
          - name: classifier
            type: torch.nn.Linear
            params:
              in_features: 10
              out_features: 1
            inputs:
              input: features

        outputs:
          prediction: classifier.output
        """

        model_spec = ModelSpec.from_yaml(yaml_content)

        # Check categorical input parsed correctly
        assert "user_id" in model_spec.inputs
        user_input = model_spec.inputs["user_id"]
        assert user_input.dtype == "long"
        assert user_input.shape == [None]
        assert user_input.categorical is True
        assert user_input.embedding_dim == 32
        assert user_input.vocab_size == 1000

        # Check regular input still works
        assert "features" in model_spec.inputs
        features_input = model_spec.inputs["features"]
        assert features_input.dtype == "float32"
        assert features_input.categorical is False
        assert features_input.embedding_dim is None

    def test_parse_yaml_categorical_minimal(self):
        """Test parsing YAML with minimal categorical specification."""
        yaml_content = """
        inputs:
          category:
            dtype: long
            shape: [null]
            categorical: true

        graph:
          - name: dense
            type: torch.nn.Linear
            params:
              in_features: 1
              out_features: 1
            inputs:
              input: category

        outputs:
          result: dense.output
        """

        model_spec = ModelSpec.from_yaml(yaml_content)

        category_input = model_spec.inputs["category"]
        assert category_input.categorical is True
        assert category_input.embedding_dim is None
        assert category_input.vocab_size is None

    def test_yaml_round_trip_with_embedding(self):
        """Test YAML round-trip preserves embedding parameters."""
        original_yaml = """
        inputs:
          genre_id:
            dtype: long
            shape: [null]
            categorical: true
            embedding_dim: 16
            vocab_size: 50
            columns: [genre_encoded]

        graph:
          - name: output
            type: torch.nn.Linear
            params:
              in_features: 1
              out_features: 1
            inputs:
              input: genre_id

        outputs:
          prediction: output.output
        """

        # Parse
        model_spec = ModelSpec.from_yaml(original_yaml)

        # Convert back to YAML
        regenerated_yaml = model_spec.to_yaml()

        # Parse again
        model_spec2 = ModelSpec.from_yaml(regenerated_yaml)

        # Check embedding parameters preserved
        genre_input = model_spec2.inputs["genre_id"]
        assert genre_input.categorical is True
        assert genre_input.embedding_dim == 16
        assert genre_input.vocab_size == 50
        assert genre_input.columns == ["genre_encoded"]

    def test_multiple_categorical_inputs(self):
        """Test model with multiple categorical inputs."""
        yaml_content = """
        inputs:
          user_id:
            dtype: long
            shape: [null]
            categorical: true
            embedding_dim: 64
            vocab_size: 10000
          movie_id:
            dtype: long
            shape: [null]
            categorical: true
            embedding_dim: 128
            vocab_size: 5000
          rating:
            dtype: float32
            shape: [null]

        graph:
          - name: classifier
            type: torch.nn.Linear
            params:
              in_features: 1
              out_features: 1
            inputs:
              input: rating

        outputs:
          prediction: classifier.output
        """

        model_spec = ModelSpec.from_yaml(yaml_content)

        # Check first categorical input
        user_input = model_spec.inputs["user_id"]
        assert user_input.categorical is True
        assert user_input.embedding_dim == 64
        assert user_input.vocab_size == 10000

        # Check second categorical input
        movie_input = model_spec.inputs["movie_id"]
        assert movie_input.categorical is True
        assert movie_input.embedding_dim == 128
        assert movie_input.vocab_size == 5000

        # Check regular input
        rating_input = model_spec.inputs["rating"]
        assert rating_input.categorical is False
        assert rating_input.embedding_dim is None
