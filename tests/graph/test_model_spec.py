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
