"""Tests for TrainerSpec in the separated architecture."""

import pytest

from arc.graph.trainer import (
    LossConfig,
    OptimizerConfig,
    TrainerSpec,
    validate_trainer_dict,
)
from arc.graph.trainer.validator import TrainerValidationError


class TestOptimizerConfig:
    """Test OptimizerConfig dataclass."""

    def test_optimizer_config_creation(self):
        """Test basic OptimizerConfig creation."""
        optimizer = OptimizerConfig(
            type="torch.optim.Adam", params={"lr": 0.001, "betas": [0.9, 0.999]}
        )

        assert optimizer.type == "torch.optim.Adam"
        assert optimizer.params == {"lr": 0.001, "betas": [0.9, 0.999]}

    def test_optimizer_config_no_params(self):
        """Test OptimizerConfig without parameters."""
        optimizer = OptimizerConfig(type="torch.optim.SGD")

        assert optimizer.type == "torch.optim.SGD"
        assert optimizer.params is None


class TestLossConfig:
    """Test LossConfig dataclass."""

    def test_loss_config_creation(self):
        """Test basic LossConfig creation."""
        loss = LossConfig(
            type="torch.nn.CrossEntropyLoss",
            params={"ignore_index": -100},
            inputs={"input": "model.logits", "target": "target"},
        )

        assert loss.type == "torch.nn.CrossEntropyLoss"
        assert loss.params == {"ignore_index": -100}
        assert loss.inputs == {"input": "model.logits", "target": "target"}

    def test_loss_config_minimal(self):
        """Test LossConfig with minimal parameters."""
        loss = LossConfig(type="torch.nn.MSELoss")

        assert loss.type == "torch.nn.MSELoss"
        assert loss.params is None
        assert loss.inputs is None


class TestTrainerSpec:
    """Test TrainerSpec functionality."""

    @pytest.fixture
    def sample_trainer_dict(self):
        """Sample trainer dictionary for testing."""
        return {
            "model_ref": "test-model-v1",
            "loss": {
                "type": "torch.nn.CrossEntropyLoss",
                "inputs": {"input": "model.logits", "target": "target"},
            },
            "optimizer": {
                "type": "torch.optim.Adam",
                "params": {"lr": 0.001, "betas": [0.9, 0.999], "weight_decay": 0.0001},
            },
            "config": {
                "epochs": 100,
                "batch_size": 32,
                "validation_split": 0.2,
                "early_stopping_patience": 10,
                "device": "auto",
            },
        }

    @pytest.fixture
    def sample_trainer_yaml(self, sample_trainer_dict):
        """Sample trainer YAML string."""
        import yaml

        return yaml.dump(sample_trainer_dict)

    def test_trainer_spec_from_yaml(self, sample_trainer_yaml):
        """Test TrainerSpec creation from YAML."""
        trainer_spec = TrainerSpec.from_yaml(sample_trainer_yaml)

        # Check model_ref
        assert trainer_spec.model_ref == "test-model-v1"

        # Check optimizer
        assert trainer_spec.optimizer.type == "torch.optim.Adam"
        assert trainer_spec.optimizer.params["lr"] == 0.001
        assert trainer_spec.optimizer.params["betas"] == [0.9, 0.999]

        # Check config
        assert trainer_spec.epochs == 100
        assert trainer_spec.batch_size == 32
        assert trainer_spec.validation_split == 0.2
        assert trainer_spec.early_stopping_patience == 10
        assert trainer_spec.device == "auto"

    def test_trainer_spec_to_yaml(self, sample_trainer_yaml):
        """Test TrainerSpec conversion to YAML."""
        trainer_spec = TrainerSpec.from_yaml(sample_trainer_yaml)
        regenerated_yaml = trainer_spec.to_yaml()

        import yaml

        regenerated_dict = yaml.safe_load(regenerated_yaml)

        # Check structure is preserved
        assert regenerated_dict["model_ref"] == "test-model-v1"
        assert regenerated_dict["optimizer"]["type"] == "torch.optim.Adam"
        assert regenerated_dict["config"]["epochs"] == 100

    def test_trainer_spec_minimal(self):
        """Test TrainerSpec with minimal configuration."""
        minimal_yaml = """
        model_ref: test-model-v1
        loss:
          type: torch.nn.MSELoss
        optimizer:
          type: torch.optim.SGD
          lr: 0.01
        """

        trainer_spec = TrainerSpec.from_yaml(minimal_yaml)

        assert trainer_spec.model_ref == "test-model-v1"
        assert trainer_spec.optimizer.type == "torch.optim.SGD"
        assert trainer_spec.optimizer.params is None

        # Check defaults
        assert trainer_spec.epochs == 10  # Default value
        assert trainer_spec.batch_size == 32  # Default value
        assert trainer_spec.learning_rate == 0.001  # Default value

    def test_trainer_spec_with_custom_config(self):
        """Test TrainerSpec with custom configuration."""
        custom_yaml = """
        model_ref: test-model-v1
        loss:
          type: torch.nn.CrossEntropyLoss
        optimizer:
          type: torch.optim.AdamW
          lr: 0.0001
        config:
          epochs: 50
          batch_size: 64
          learning_rate: 0.0001
          validation_split: 0.15
          device: cuda
        """

        trainer_spec = TrainerSpec.from_yaml(custom_yaml)

        assert trainer_spec.model_ref == "test-model-v1"
        assert trainer_spec.epochs == 50
        assert trainer_spec.batch_size == 64
        assert trainer_spec.learning_rate == 0.0001
        assert trainer_spec.validation_split == 0.15
        assert trainer_spec.device == "cuda"

    def test_invalid_yaml_structure(self):
        """Test TrainerSpec with invalid YAML structure."""
        invalid_yaml = "- not_a_dict\n- but_a_list"

        with pytest.raises(ValueError, match="Top-level YAML must be a mapping"):
            TrainerSpec.from_yaml(invalid_yaml)

    def test_missing_required_fields(self):
        """Test TrainerSpec with missing required fields."""
        incomplete_yaml = """
        optimizer:
          type: torch.optim.Adam
          lr: 0.001
        # Missing model_ref
        """

        with pytest.raises(TrainerValidationError):
            TrainerSpec.from_yaml(incomplete_yaml)


class TestTrainerValidation:
    """Test trainer validation functionality."""

    def test_validate_trainer_dict_valid(self):
        """Test validation of valid trainer dictionary."""
        valid_dict = {
            "model_ref": "test-model-v1",
            "loss": {"type": "torch.nn.CrossEntropyLoss"},
            "optimizer": {"type": "torch.optim.Adam", "lr": 0.001},
        }

        # Should not raise any exceptions
        validate_trainer_dict(valid_dict)

    def test_validate_trainer_dict_missing_optimizer(self):
        """Test validation with missing optimizer section."""
        invalid_dict = {"model_ref": "test-model-v1"}

        with pytest.raises(TrainerValidationError, match="trainer.optimizer required"):
            validate_trainer_dict(invalid_dict)

    def test_validate_trainer_dict_missing_model_ref(self):
        """Test validation with missing model_ref section."""
        invalid_dict = {"optimizer": {"type": "torch.optim.SGD", "lr": 0.01}}

        with pytest.raises(TrainerValidationError, match="trainer.model_ref required"):
            validate_trainer_dict(invalid_dict)

    def test_validate_trainer_dict_invalid_batch_size(self):
        """Test validation with invalid batch size."""
        invalid_dict = {
            "model_ref": "test-model-v1",
            "loss": {"type": "torch.nn.MSELoss"},
            "optimizer": {"type": "torch.optim.Adam", "lr": 0.001},
            "config": {
                "batch_size": -1  # Invalid
            },
        }

        with pytest.raises(
            TrainerValidationError, match="batch_size must be a positive integer"
        ):
            validate_trainer_dict(invalid_dict)

    def test_validate_trainer_dict_invalid_validation_split(self):
        """Test validation with invalid validation split."""
        invalid_dict = {
            "model_ref": "test-model-v1",
            "loss": {"type": "torch.nn.MSELoss"},
            "optimizer": {"type": "torch.optim.Adam", "lr": 0.001},
            "config": {
                "validation_split": 1.5  # Invalid (> 1)
            },
        }

        with pytest.raises(
            TrainerValidationError, match="validation_split must be between 0 and 1"
        ):
            validate_trainer_dict(invalid_dict)

    def test_validate_trainer_dict_invalid_device(self):
        """Test validation with invalid device."""
        invalid_dict = {
            "model_ref": "test-model-v1",
            "loss": {"type": "torch.nn.MSELoss"},
            "optimizer": {"type": "torch.optim.Adam", "lr": 0.001},
            "config": {"device": "invalid_device"},
        }

        with pytest.raises(TrainerValidationError, match="device must be one of"):
            validate_trainer_dict(invalid_dict)
