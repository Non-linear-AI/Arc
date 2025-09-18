"""Tests for the Arc Graph model trainer."""

import tempfile
from pathlib import Path

import pytest
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from arc.ml.trainer import (
    ArcTrainer,
    SimpleProgressCallback,
    TrainingConfig,
    TrainingResult,
)


class SimpleTestModel(nn.Module):
    """Simple model for testing."""

    def __init__(self, input_size: int = 10, output_size: int = 2):
        super().__init__()
        self.linear1 = nn.Linear(input_size, 16)
        self.relu = nn.ReLU()
        self.linear2 = nn.Linear(16, output_size)

    def forward(self, x):
        x = self.linear1(x)
        x = self.relu(x)
        x = self.linear2(x)
        return x


class DictOutputModel(nn.Module):
    """Model that returns dictionary output for testing."""

    def __init__(self, input_size: int = 10, output_size: int = 2):
        super().__init__()
        self.linear1 = nn.Linear(input_size, 16)
        self.relu = nn.ReLU()
        self.prediction_head = nn.Linear(16, output_size)
        self.auxiliary_head = nn.Linear(16, 3)

    def forward(self, x):
        x = self.linear1(x)
        x = self.relu(x)
        return {
            "predictions": self.prediction_head(x),
            "auxiliary": self.auxiliary_head(x),
        }


@pytest.fixture
def sample_data():
    """Create sample training data."""
    # Create synthetic binary classification data
    torch.manual_seed(42)

    # Training data
    train_x = torch.randn(100, 10)
    train_y = torch.randint(0, 2, (100,))
    train_dataset = TensorDataset(train_x, train_y)
    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)

    # Validation data
    val_x = torch.randn(50, 10)
    val_y = torch.randint(0, 2, (50,))
    val_dataset = TensorDataset(val_x, val_y)
    val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False)

    return train_loader, val_loader


@pytest.fixture
def basic_config():
    """Create basic training configuration."""
    return TrainingConfig(
        epochs=3,
        batch_size=16,
        learning_rate=0.01,
        optimizer="adam",
        loss_function="cross_entropy",
        device="cpu",
        verbose=False,
    )


class TestTrainingConfig:
    """Test training configuration."""

    def test_default_config(self):
        """Test default configuration values."""
        config = TrainingConfig()

        assert config.epochs == 10
        assert config.batch_size == 32
        assert config.learning_rate == 0.001
        assert config.optimizer == "adam"
        assert config.loss_function == "cross_entropy"
        assert config.device == "auto"

    def test_custom_config(self):
        """Test custom configuration."""
        config = TrainingConfig(
            epochs=5,
            batch_size=64,
            learning_rate=0.01,
            optimizer="sgd",
            loss_function="mse",
        )

        assert config.epochs == 5
        assert config.batch_size == 64
        assert config.learning_rate == 0.01
        assert config.optimizer == "sgd"
        assert config.loss_function == "mse"


class TestArcTrainer:
    """Test Arc trainer functionality."""

    def test_trainer_initialization(self, basic_config):
        """Test trainer initialization."""
        trainer = ArcTrainer(basic_config)

        assert trainer.config == basic_config
        assert trainer.device == torch.device("cpu")
        assert trainer.model is None
        assert trainer.optimizer is None
        assert trainer.loss_fn is None

    def test_device_setup_cpu(self):
        """Test CPU device setup."""
        config = TrainingConfig(device="cpu")
        trainer = ArcTrainer(config)

        assert trainer.device == torch.device("cpu")

    def test_device_setup_auto(self):
        """Test auto device setup."""
        config = TrainingConfig(device="auto")
        trainer = ArcTrainer(config)

        # Should default to CPU in test environment
        assert trainer.device.type in ["cpu", "cuda", "mps"]

    def test_optimizer_setup_adam(self, basic_config):
        """Test Adam optimizer setup."""
        trainer = ArcTrainer(basic_config)
        model = SimpleTestModel()

        optimizer = trainer._setup_optimizer(model)

        assert isinstance(optimizer, torch.optim.Adam)
        assert optimizer.param_groups[0]["lr"] == basic_config.learning_rate

    def test_optimizer_setup_sgd(self):
        """Test SGD optimizer setup."""
        config = TrainingConfig(optimizer="sgd", learning_rate=0.01)
        trainer = ArcTrainer(config)
        model = SimpleTestModel()

        optimizer = trainer._setup_optimizer(model)

        assert isinstance(optimizer, torch.optim.SGD)
        assert optimizer.param_groups[0]["lr"] == 0.01

    def test_optimizer_setup_invalid(self):
        """Test invalid optimizer raises error."""
        config = TrainingConfig(optimizer="invalid_optimizer")
        trainer = ArcTrainer(config)
        model = SimpleTestModel()

        with pytest.raises(ValueError, match="Unsupported optimizer"):
            trainer._setup_optimizer(model)

    def test_loss_function_setup_cross_entropy(self, basic_config):
        """Test cross entropy loss setup."""
        trainer = ArcTrainer(basic_config)

        loss_fn = trainer._setup_loss_function()

        assert isinstance(loss_fn, nn.CrossEntropyLoss)

    def test_loss_function_setup_mse(self):
        """Test MSE loss setup."""
        config = TrainingConfig(loss_function="mse")
        trainer = ArcTrainer(config)

        loss_fn = trainer._setup_loss_function()

        assert isinstance(loss_fn, nn.MSELoss)

    def test_loss_function_setup_invalid(self):
        """Test invalid loss function raises error."""
        config = TrainingConfig(loss_function="invalid_loss")
        trainer = ArcTrainer(config)

        with pytest.raises(ValueError, match="Unsupported loss function"):
            trainer._setup_loss_function()

    def test_basic_training(self, basic_config, sample_data):
        """Test basic training functionality."""
        train_loader, val_loader = sample_data
        model = SimpleTestModel()
        trainer = ArcTrainer(basic_config)

        result = trainer.train(model, train_loader, val_loader)

        assert isinstance(result, TrainingResult)
        assert result.success is True
        assert result.total_epochs == basic_config.epochs
        assert len(result.train_losses) == basic_config.epochs
        assert len(result.val_losses) == basic_config.epochs
        assert result.training_time > 0
        assert result.final_train_loss > 0
        assert result.final_val_loss is not None

    def test_training_without_validation(self, basic_config, sample_data):
        """Test training without validation data."""
        train_loader, _ = sample_data
        model = SimpleTestModel()
        trainer = ArcTrainer(basic_config)

        result = trainer.train(model, train_loader, val_loader=None)

        assert result.success is True
        assert len(result.val_losses) == 0
        assert result.final_val_loss is None
        assert result.best_val_loss is None

    def test_training_with_checkpointing(self, basic_config, sample_data):
        """Test training with checkpoint saving."""
        train_loader, val_loader = sample_data
        model = SimpleTestModel()
        trainer = ArcTrainer(basic_config)

        with tempfile.TemporaryDirectory() as temp_dir:
            checkpoint_dir = Path(temp_dir)

            result = trainer.train(
                model, train_loader, val_loader, checkpoint_dir=checkpoint_dir
            )

            assert result.success is True
            assert result.best_model_path is not None
            assert Path(result.best_model_path).exists()

    def test_training_with_callback(self, basic_config, sample_data):
        """Test training with progress callback."""
        train_loader, val_loader = sample_data
        model = SimpleTestModel()
        trainer = ArcTrainer(basic_config)

        # Track callback calls
        callback_calls = {
            "training_start": 0,
            "epoch_start": 0,
            "epoch_end": 0,
            "training_end": 0,
        }

        class TestCallback:
            def on_training_start(self):
                callback_calls["training_start"] += 1

            def on_epoch_start(self, _epoch, _total_epochs):
                callback_calls["epoch_start"] += 1

            def on_epoch_end(self, _epoch, _metrics):
                callback_calls["epoch_end"] += 1

            def on_batch_end(self, batch, total_batches, loss):
                pass

            def on_training_end(self, _final_metrics):
                callback_calls["training_end"] += 1

        callback = TestCallback()
        result = trainer.train(model, train_loader, val_loader, callback=callback)

        assert result.success is True
        assert callback_calls["training_start"] == 1
        assert callback_calls["epoch_start"] == basic_config.epochs
        assert callback_calls["epoch_end"] == basic_config.epochs
        assert callback_calls["training_end"] == 1

    def test_early_stopping(self, sample_data):
        """Test early stopping functionality."""
        config = TrainingConfig(
            epochs=10,
            early_stopping_patience=2,
            early_stopping_min_delta=0.001,
            device="cpu",
            verbose=False,
        )

        train_loader, val_loader = sample_data
        model = SimpleTestModel()
        trainer = ArcTrainer(config)

        result = trainer.train(model, train_loader, val_loader)

        assert result.success is True
        # Should stop early if no improvement
        assert len(result.train_losses) <= config.epochs

    def test_checkpoint_loading(self, basic_config, sample_data):
        """Test checkpoint loading."""
        train_loader, val_loader = sample_data
        model = SimpleTestModel()
        trainer = ArcTrainer(basic_config)

        with tempfile.TemporaryDirectory() as temp_dir:
            checkpoint_dir = Path(temp_dir)

            # Train and save checkpoint
            result = trainer.train(
                model, train_loader, val_loader, checkpoint_dir=checkpoint_dir
            )

            assert result.success is True

            # Load checkpoint
            checkpoint_path = result.best_model_path
            if checkpoint_path:
                metadata = trainer.load_checkpoint(checkpoint_path)

                assert "epoch" in metadata
                assert "val_loss" in metadata
                assert metadata["epoch"] > 0

    def test_training_failure_handling(self, sample_data):
        """Test training failure handling."""
        # Create config that will cause failure
        config = TrainingConfig(device="invalid_device")
        train_loader, _ = sample_data
        model = SimpleTestModel()

        with pytest.raises(RuntimeError, match="Expected one of .* device type"):
            trainer = ArcTrainer(config)
            trainer.train(model, train_loader)


class TestSimpleProgressCallback:
    """Test simple progress callback."""

    def test_callback_methods(self):
        """Test callback method calls."""
        callback = SimpleProgressCallback(log_every=1)

        # These should not raise exceptions
        callback.on_training_start()
        callback.on_epoch_start(1, 3)
        callback.on_epoch_end(1, {"train_loss": 1.0, "val_loss": 0.8})
        callback.on_batch_end(1, 10, 0.5)
        callback.on_training_end({"final_loss": 0.3})


class TestTrainingResult:
    """Test training result data structure."""

    def test_training_result_creation(self):
        """Test training result creation."""
        result = TrainingResult(
            success=True,
            total_epochs=5,
            best_epoch=3,
            final_train_loss=0.5,
            final_val_loss=0.4,
            best_val_loss=0.3,
            training_time=120.5,
        )

        assert result.success is True
        assert result.total_epochs == 5
        assert result.best_epoch == 3
        assert result.final_train_loss == 0.5
        assert result.final_val_loss == 0.4
        assert result.best_val_loss == 0.3
        assert result.training_time == 120.5

    def test_training_result_defaults(self):
        """Test training result with defaults."""
        result = TrainingResult(
            success=False,
            total_epochs=0,
            best_epoch=0,
            final_train_loss=0.0,
        )

        assert result.success is False
        assert result.final_val_loss is None
        assert result.best_val_loss is None
        assert len(result.train_losses) == 0
        assert len(result.val_losses) == 0
        assert result.training_time == 0.0
        assert result.checkpoint_path is None
        assert result.error_message is None


class TestTrainerFlexibleConfiguration:
    """Test flexible configuration options for trainer."""

    def test_tensor_output_no_key(self, sample_data):
        """Test tensor output with no target_output_key."""
        train_loader, _ = sample_data

        config = TrainingConfig(
            epochs=1,
            loss_function="cross_entropy",
            target_output_key=None,  # No key specified
        )
        trainer = ArcTrainer(config)
        model = SimpleTestModel()

        # Should complete without error using whole tensor output
        result = trainer.train(model, train_loader)
        assert result.success

    def test_dict_output_with_key(self, sample_data):
        """Test dict output with explicit target_output_key."""
        train_loader, _ = sample_data

        config = TrainingConfig(
            epochs=1,
            loss_function="cross_entropy",
            target_output_key="predictions",  # Use specific key
        )
        trainer = ArcTrainer(config)
        model = DictOutputModel()

        # Should use "predictions" key from model output
        result = trainer.train(model, train_loader)
        assert result.success

    def test_dict_output_invalid_key(self, sample_data):
        """Test dict output with invalid target_output_key."""
        train_loader, _ = sample_data

        config = TrainingConfig(
            epochs=1,
            loss_function="cross_entropy",
            target_output_key="nonexistent",  # Invalid key
        )
        trainer = ArcTrainer(config)
        model = DictOutputModel()

        # Should raise RuntimeError for missing key
        with pytest.raises(RuntimeError, match="Target output key"):
            trainer.train(model, train_loader)

    def test_dict_output_no_key_error(self, sample_data):
        """Test dict output with no target_output_key specified (should error)."""
        train_loader, _ = sample_data

        config = TrainingConfig(
            epochs=1,
            loss_function="cross_entropy",
            target_output_key=None,  # No key specified for dict output
        )
        trainer = ArcTrainer(config)
        model = DictOutputModel()

        # Should raise an error since dict output requires a key
        with pytest.raises((RuntimeError, TypeError)):
            trainer.train(model, train_loader)

    def test_tensor_output_with_key_error(self, sample_data):
        """Test tensor output with target_output_key specified (should error)."""
        train_loader, _ = sample_data

        config = TrainingConfig(
            epochs=1,
            loss_function="cross_entropy",
            target_output_key="predictions",  # Key specified for tensor output
        )
        trainer = ArcTrainer(config)
        model = SimpleTestModel()

        # Should raise RuntimeError since output is not a dict
        with pytest.raises(RuntimeError, match="but model output is not a dictionary"):
            trainer.train(model, train_loader)
