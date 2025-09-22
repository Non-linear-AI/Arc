"""Tests for PredictorSpec in the separated architecture."""

import pytest

from arc.graph.predictor import (
    PredictorSpec,
    PredictorValidationError,
    validate_predictor_dict,
)


class TestPredictorSpec:
    """Test PredictorSpec functionality."""

    @pytest.fixture
    def sample_predictor_dict(self):
        """Sample predictor dictionary for testing."""
        return {
            "name": "diabetes_risk_predictor",
            "model_id": "diabetes_classifier",
            "model_version": 1,
            "outputs": {
                "prediction": "sigmoid.output",
                "confidence": "sigmoid.output",
                "raw_logits": "classifier.output",
            },
        }

    @pytest.fixture
    def sample_predictor_yaml(self, sample_predictor_dict):
        """Sample predictor YAML string."""
        import yaml

        return yaml.dump(sample_predictor_dict)

    def test_predictor_spec_from_yaml(self, sample_predictor_yaml):
        """Test PredictorSpec creation from YAML."""
        predictor_spec = PredictorSpec.from_yaml(sample_predictor_yaml)

        assert predictor_spec.name == "diabetes_risk_predictor"
        assert predictor_spec.model_id == "diabetes_classifier"
        assert predictor_spec.model_version == 1

        assert len(predictor_spec.outputs) == 3
        assert predictor_spec.outputs["prediction"] == "sigmoid.output"
        assert predictor_spec.outputs["confidence"] == "sigmoid.output"
        assert predictor_spec.outputs["raw_logits"] == "classifier.output"

    def test_predictor_spec_to_yaml(self, sample_predictor_yaml):
        """Test PredictorSpec conversion to YAML."""
        predictor_spec = PredictorSpec.from_yaml(sample_predictor_yaml)
        regenerated_yaml = predictor_spec.to_yaml()

        import yaml

        regenerated_dict = yaml.safe_load(regenerated_yaml)

        # Check structure is preserved
        assert regenerated_dict["name"] == "diabetes_risk_predictor"
        assert regenerated_dict["model_id"] == "diabetes_classifier"
        assert regenerated_dict["model_version"] == 1
        assert len(regenerated_dict["outputs"]) == 3

    def test_predictor_spec_minimal(self):
        """Test PredictorSpec with minimal configuration."""
        minimal_yaml = """
        name: simple_predictor
        model_id: test_model
        """

        predictor_spec = PredictorSpec.from_yaml(minimal_yaml)

        assert predictor_spec.name == "simple_predictor"
        assert predictor_spec.model_id == "test_model"
        assert predictor_spec.model_version is None  # Optional
        assert predictor_spec.outputs is None  # Optional

    def test_predictor_spec_without_version(self):
        """Test PredictorSpec without model version."""
        yaml_content = """
        name: version_free_predictor
        model_id: some_model
        outputs:
          result: "model.output"
        """

        predictor_spec = PredictorSpec.from_yaml(yaml_content)

        assert predictor_spec.name == "version_free_predictor"
        assert predictor_spec.model_id == "some_model"
        assert predictor_spec.model_version is None
        assert predictor_spec.outputs["result"] == "model.output"

    def test_predictor_spec_with_complex_outputs(self):
        """Test PredictorSpec with complex output mappings."""
        complex_yaml = """
        name: complex_predictor
        model_id: multi_output_model
        model_version: 2
        outputs:
          classification: "classifier.output"
          regression: "regressor.output"
          attention_weights: "attention.weights"
          embeddings: "encoder.hidden_state"
        """

        predictor_spec = PredictorSpec.from_yaml(complex_yaml)

        assert len(predictor_spec.outputs) == 4
        assert predictor_spec.outputs["classification"] == "classifier.output"
        assert predictor_spec.outputs["regression"] == "regressor.output"
        assert predictor_spec.outputs["attention_weights"] == "attention.weights"
        assert predictor_spec.outputs["embeddings"] == "encoder.hidden_state"

    def test_invalid_yaml_structure(self):
        """Test PredictorSpec with invalid YAML structure."""
        invalid_yaml = "- not_a_dict\n- but_a_list"

        with pytest.raises(ValueError, match="Top-level YAML must be a mapping"):
            PredictorSpec.from_yaml(invalid_yaml)

    def test_missing_required_fields(self):
        """Test PredictorSpec with missing required fields."""
        incomplete_yaml = """
        name: incomplete_predictor
        # Missing model_id
        """

        with pytest.raises(PredictorValidationError):
            PredictorSpec.from_yaml(incomplete_yaml)


class TestPredictorValidation:
    """Test predictor validation functionality."""

    def test_validate_predictor_dict_valid(self):
        """Test validation of valid predictor dictionary."""
        valid_dict = {
            "name": "test_predictor",
            "model_id": "test_model",
            "outputs": {"prediction": "model.output"},
        }

        # Should not raise any exceptions
        validate_predictor_dict(valid_dict)

    def test_validate_predictor_dict_minimal(self):
        """Test validation of minimal predictor dictionary."""
        minimal_dict = {"name": "minimal_predictor", "model_id": "minimal_model"}

        # Should not raise any exceptions
        validate_predictor_dict(minimal_dict)

    def test_validate_predictor_dict_missing_name(self):
        """Test validation with missing name field."""
        invalid_dict = {"model_id": "test_model"}

        with pytest.raises(
            PredictorValidationError, match="Missing required field: name"
        ):
            validate_predictor_dict(invalid_dict)

    def test_validate_predictor_dict_missing_model_id(self):
        """Test validation with missing model_id field."""
        invalid_dict = {"name": "test_predictor"}

        with pytest.raises(
            PredictorValidationError, match="Missing required field: model_id"
        ):
            validate_predictor_dict(invalid_dict)

    def test_validate_predictor_dict_empty_name(self):
        """Test validation with empty name."""
        invalid_dict = {"name": "", "model_id": "test_model"}

        with pytest.raises(PredictorValidationError, match="name.*non-empty"):
            validate_predictor_dict(invalid_dict)

    def test_validate_predictor_dict_empty_model_id(self):
        """Test validation with empty model_id."""
        invalid_dict = {"name": "test_predictor", "model_id": ""}

        with pytest.raises(PredictorValidationError, match="model_id.*non-empty"):
            validate_predictor_dict(invalid_dict)

    def test_validate_predictor_dict_invalid_outputs_type(self):
        """Test validation with invalid outputs type."""
        invalid_dict = {
            "name": "test_predictor",
            "model_id": "test_model",
            "outputs": "not_a_dict",  # Should be dict
        }

        with pytest.raises(
            PredictorValidationError, match="outputs must be a dictionary"
        ):
            validate_predictor_dict(invalid_dict)

    def test_validate_predictor_dict_empty_output_name(self):
        """Test validation with empty output name."""
        invalid_dict = {
            "name": "test_predictor",
            "model_id": "test_model",
            "outputs": {
                "": "model.output"  # Empty key
            },
        }

        with pytest.raises(PredictorValidationError, match="output name.*non-empty"):
            validate_predictor_dict(invalid_dict)

    def test_validate_predictor_dict_empty_model_output_reference(self):
        """Test validation with empty model output reference."""
        invalid_dict = {
            "name": "test_predictor",
            "model_id": "test_model",
            "outputs": {
                "prediction": ""  # Empty model output reference
            },
        }

        with pytest.raises(
            PredictorValidationError, match="Model output reference.*non-empty"
        ):
            validate_predictor_dict(invalid_dict)

    def test_validate_predictor_dict_invalid_model_version_type(self):
        """Test validation with invalid model version type."""
        invalid_dict = {
            "name": "test_predictor",
            "model_id": "test_model",
            "model_version": "not_an_int",  # Should be int
        }

        with pytest.raises(PredictorValidationError, match="model_version.*integer"):
            validate_predictor_dict(invalid_dict)
