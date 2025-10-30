"""Tests for evaluator specification."""

import pytest
import yaml

from arc.graph.evaluator import (
    EvaluatorSpec,
    EvaluatorValidationError,
    load_evaluator_from_yaml,
    save_evaluator_to_yaml,
    validate_evaluator_dict,
)


class TestEvaluatorSpec:
    """Tests for EvaluatorSpec class."""

    def test_evaluator_spec_creation(self):
        """Test creating an evaluator spec with all fields."""
        spec = EvaluatorSpec(
            name="diabetes_eval",
            model_id="diabetes_trainer",
            dataset="test_diabetes_data",
            target_column="outcome",
            metrics=["accuracy", "precision", "recall"],
            version=1,
        )

        assert spec.name == "diabetes_eval"
        assert spec.model_id == "diabetes_trainer"
        assert spec.dataset == "test_diabetes_data"
        assert spec.target_column == "outcome"
        assert spec.metrics == ["accuracy", "precision", "recall"]
        assert spec.version == 1

    def test_evaluator_spec_minimal(self):
        """Test creating an evaluator spec with only required fields."""
        spec = EvaluatorSpec(
            name="minimal_eval",
            model_id="trainer_v1",
            dataset="test_data",
            target_column="target",
        )

        assert spec.name == "minimal_eval"
        assert spec.model_id == "trainer_v1"
        assert spec.dataset == "test_data"
        assert spec.target_column == "target"
        assert spec.metrics is None
        assert spec.version is None

    def test_evaluator_spec_from_yaml(self):
        """Test parsing evaluator spec from YAML."""
        yaml_str = """
name: diabetes_eval
model_id: diabetes_trainer
dataset: test_diabetes_data
target_column: outcome
metrics:
  - accuracy
  - precision
  - recall
  - f1_score
version: 2
"""
        spec = EvaluatorSpec.from_yaml(yaml_str)

        assert spec.name == "diabetes_eval"
        assert spec.model_id == "diabetes_trainer"
        assert spec.dataset == "test_diabetes_data"
        assert spec.target_column == "outcome"
        assert spec.metrics == ["accuracy", "precision", "recall", "f1_score"]
        assert spec.version == 2

    def test_evaluator_spec_from_yaml_minimal(self):
        """Test parsing minimal evaluator spec from YAML."""
        yaml_str = """
name: minimal_eval
model_id: trainer_v1
dataset: test_data
target_column: target
"""
        spec = EvaluatorSpec.from_yaml(yaml_str)

        assert spec.name == "minimal_eval"
        assert spec.model_id == "trainer_v1"
        assert spec.dataset == "test_data"
        assert spec.target_column == "target"
        assert spec.metrics is None
        assert spec.version is None

    def test_evaluator_spec_to_yaml(self):
        """Test converting evaluator spec to YAML."""
        spec = EvaluatorSpec(
            name="diabetes_eval",
            model_id="diabetes_trainer",
            dataset="test_diabetes_data",
            target_column="outcome",
            metrics=["accuracy", "precision"],
            version=1,
        )

        yaml_str = spec.to_yaml()
        data = yaml.safe_load(yaml_str)

        assert data["name"] == "diabetes_eval"
        assert data["model_id"] == "diabetes_trainer"
        assert data["dataset"] == "test_diabetes_data"
        assert data["target_column"] == "outcome"
        assert data["metrics"] == ["accuracy", "precision"]
        assert data["version"] == 1

    def test_evaluator_spec_to_yaml_minimal(self):
        """Test converting minimal evaluator spec to YAML."""
        spec = EvaluatorSpec(
            name="minimal_eval",
            model_id="trainer_v1",
            dataset="test_data",
            target_column="target",
        )

        yaml_str = spec.to_yaml()
        data = yaml.safe_load(yaml_str)

        assert data["name"] == "minimal_eval"
        assert data["model_id"] == "trainer_v1"
        assert data["dataset"] == "test_data"
        assert data["target_column"] == "target"
        assert "metrics" not in data
        assert "version" not in data

    def test_evaluator_spec_roundtrip(self):
        """Test that spec can be converted to YAML and back."""
        original = EvaluatorSpec(
            name="roundtrip_eval",
            model_id="roundtrip_trainer",
            dataset="roundtrip_data",
            target_column="target",
            metrics=["accuracy", "f1_score"],
            version=3,
        )

        yaml_str = original.to_yaml()
        restored = EvaluatorSpec.from_yaml(yaml_str)

        assert restored.name == original.name
        assert restored.model_id == original.model_id
        assert restored.dataset == original.dataset
        assert restored.target_column == original.target_column
        assert restored.metrics == original.metrics
        assert restored.version == original.version

    def test_invalid_yaml_structure(self):
        """Test that non-dict YAML is rejected."""
        yaml_str = "- just a list"

        with pytest.raises(ValueError, match="Top-level YAML must be a mapping"):
            EvaluatorSpec.from_yaml(yaml_str)


class TestEvaluatorValidation:
    """Tests for evaluator validation."""

    def test_validate_evaluator_dict_valid(self):
        """Test validation with valid evaluator dict."""
        data = {
            "name": "test_eval",
            "model_id": "test_trainer",
            "dataset": "test_data",
            "target_column": "target",
            "metrics": ["accuracy"],
            "version": 1,
        }

        spec = validate_evaluator_dict(data)

        assert spec.name == "test_eval"
        assert spec.model_id == "test_trainer"
        assert spec.dataset == "test_data"
        assert spec.target_column == "target"
        assert spec.metrics == ["accuracy"]
        assert spec.version == 1

    def test_validate_evaluator_dict_missing_name(self):
        """Test validation fails when name is missing."""
        data = {
            "model_id": "test_trainer",
            "dataset": "test_data",
            "target_column": "target",
        }

        with pytest.raises(
            EvaluatorValidationError, match="Missing required field: name"
        ):
            validate_evaluator_dict(data)

    def test_validate_evaluator_dict_missing_model_id(self):
        """Test validation fails when model_id is missing."""
        data = {
            "name": "test_eval",
            "dataset": "test_data",
            "target_column": "target",
        }

        with pytest.raises(
            EvaluatorValidationError, match="Missing required field: model_id"
        ):
            validate_evaluator_dict(data)

    def test_validate_evaluator_dict_missing_dataset(self):
        """Test validation fails when dataset is missing."""
        data = {
            "name": "test_eval",
            "model_id": "test_trainer",
            "target_column": "target",
        }

        with pytest.raises(
            EvaluatorValidationError, match="Missing required field: dataset"
        ):
            validate_evaluator_dict(data)

    def test_validate_evaluator_dict_missing_target_column(self):
        """Test validation fails when target_column is missing."""
        data = {
            "name": "test_eval",
            "model_id": "test_trainer",
            "dataset": "test_data",
        }

        with pytest.raises(
            EvaluatorValidationError, match="Missing required field: target_column"
        ):
            validate_evaluator_dict(data)

    def test_validate_evaluator_dict_empty_name(self):
        """Test validation fails when name is empty string."""
        data = {
            "name": "   ",
            "model_id": "test_trainer",
            "dataset": "test_data",
            "target_column": "target",
        }

        with pytest.raises(
            EvaluatorValidationError, match="name must be a non-empty string"
        ):
            validate_evaluator_dict(data)

    def test_validate_evaluator_dict_empty_model_id(self):
        """Test validation fails when model_id is empty string."""
        data = {
            "name": "test_eval",
            "model_id": "  ",
            "dataset": "test_data",
            "target_column": "target",
        }

        with pytest.raises(
            EvaluatorValidationError, match="model_id must be a non-empty string"
        ):
            validate_evaluator_dict(data)

    def test_validate_evaluator_dict_empty_dataset(self):
        """Test validation fails when dataset is empty string."""
        data = {
            "name": "test_eval",
            "model_id": "test_trainer",
            "dataset": "",
            "target_column": "target",
        }

        with pytest.raises(
            EvaluatorValidationError, match="dataset must be a non-empty string"
        ):
            validate_evaluator_dict(data)

    def test_validate_evaluator_dict_invalid_metrics_type(self):
        """Test validation fails when metrics is not a list."""
        data = {
            "name": "test_eval",
            "model_id": "test_trainer",
            "dataset": "test_data",
            "target_column": "target",
            "metrics": "accuracy",  # Should be a list
        }

        with pytest.raises(EvaluatorValidationError, match="metrics must be a list"):
            validate_evaluator_dict(data)

    def test_validate_evaluator_dict_invalid_metric_element(self):
        """Test validation fails when metrics contains non-strings."""
        data = {
            "name": "test_eval",
            "model_id": "test_trainer",
            "dataset": "test_data",
            "target_column": "target",
            "metrics": ["accuracy", 123],  # 123 is not a string
        }

        with pytest.raises(
            EvaluatorValidationError, match="All metrics must be strings"
        ):
            validate_evaluator_dict(data)

    def test_validate_evaluator_dict_empty_metric_string(self):
        """Test validation fails when a metric name is empty."""
        data = {
            "name": "test_eval",
            "model_id": "test_trainer",
            "dataset": "test_data",
            "target_column": "target",
            "metrics": ["accuracy", "  "],  # Empty metric name
        }

        with pytest.raises(
            EvaluatorValidationError, match="Metric names cannot be empty strings"
        ):
            validate_evaluator_dict(data)

    def test_validate_evaluator_dict_invalid_version_type(self):
        """Test validation fails when version is not an integer."""
        data = {
            "name": "test_eval",
            "model_id": "test_trainer",
            "dataset": "test_data",
            "target_column": "target",
            "version": "1",  # Should be an int
        }

        with pytest.raises(
            EvaluatorValidationError, match="version must be an integer"
        ):
            validate_evaluator_dict(data)

    def test_validate_evaluator_dict_strips_whitespace(self):
        """Test that validation strips whitespace from string fields."""
        data = {
            "name": "  test_eval  ",
            "model_id": "  test_trainer  ",
            "dataset": "  test_data  ",
            "target_column": "  target  ",
            "metrics": ["  accuracy  ", "  precision  "],
        }

        spec = validate_evaluator_dict(data)

        assert spec.name == "test_eval"
        assert spec.model_id == "test_trainer"
        assert spec.dataset == "test_data"
        assert spec.target_column == "target"
        assert spec.metrics == ["accuracy", "precision"]


class TestEvaluatorFileOperations:
    """Tests for loading and saving evaluator specs to files."""

    def test_load_evaluator_from_yaml_file(self, tmp_path):
        """Test loading evaluator spec from file."""
        yaml_content = """
name: file_eval
model_id: file_trainer
dataset: file_data
target_column: target
metrics:
  - accuracy
  - precision
version: 1
"""
        yaml_file = tmp_path / "evaluator.yaml"
        yaml_file.write_text(yaml_content)

        spec = load_evaluator_from_yaml(yaml_file)

        assert spec.name == "file_eval"
        assert spec.model_id == "file_trainer"
        assert spec.dataset == "file_data"
        assert spec.target_column == "target"
        assert spec.metrics == ["accuracy", "precision"]
        assert spec.version == 1

    def test_load_evaluator_from_nonexistent_file(self, tmp_path):
        """Test loading evaluator from non-existent file fails."""
        nonexistent = tmp_path / "nonexistent.yaml"

        with pytest.raises(EvaluatorValidationError, match="Evaluator file not found"):
            load_evaluator_from_yaml(nonexistent)

    def test_save_evaluator_to_yaml_file(self, tmp_path):
        """Test saving evaluator spec to file."""
        spec = EvaluatorSpec(
            name="save_eval",
            model_id="save_trainer",
            dataset="save_data",
            target_column="target",
            metrics=["accuracy", "recall"],
            version=2,
        )

        yaml_file = tmp_path / "evaluator_output.yaml"
        save_evaluator_to_yaml(spec, yaml_file)

        # Verify file was created and contains correct content
        assert yaml_file.exists()
        data = yaml.safe_load(yaml_file.read_text())

        assert data["name"] == "save_eval"
        assert data["model_id"] == "save_trainer"
        assert data["dataset"] == "save_data"
        assert data["target_column"] == "target"
        assert data["metrics"] == ["accuracy", "recall"]
        assert data["version"] == 2

    def test_save_and_load_roundtrip(self, tmp_path):
        """Test that saving and loading produces identical spec."""
        original = EvaluatorSpec(
            name="roundtrip_eval",
            model_id="roundtrip_trainer",
            dataset="roundtrip_data",
            target_column="target",
            metrics=["accuracy", "f1_score"],
            version=3,
        )

        yaml_file = tmp_path / "roundtrip.yaml"
        save_evaluator_to_yaml(original, yaml_file)
        restored = load_evaluator_from_yaml(yaml_file)

        assert restored.name == original.name
        assert restored.model_id == original.model_id
        assert restored.dataset == original.dataset
        assert restored.metrics == original.metrics
        assert restored.version == original.version
