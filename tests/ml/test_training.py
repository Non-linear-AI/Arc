"""Tests for the training module (train_model function and helpers)."""

import tempfile
from pathlib import Path
from threading import Event
from types import SimpleNamespace

import pytest
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from arc.ml.training import (
    TrainingResult,
    _get_loss_function,
    _get_optimizer,
    train_model,
)


class SimpleModel(nn.Module):
    """Simple model for testing."""

    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(10, 1)

    def forward(self, x):
        return self.fc(x)


class MockCallback:
    """Mock callback for testing."""

    def __init__(self):
        self.training_started = False
        self.training_ended = False
        self.epochs_started = []
        self.epochs_ended = []
        self.batches_ended = []

    def on_training_start(self):
        self.training_started = True

    def on_epoch_start(self, epoch: int, total_epochs: int):
        self.epochs_started.append((epoch, total_epochs))

    def on_batch_end(self, batch: int, total_batches: int, loss: float):
        self.batches_ended.append((batch, total_batches, loss))

    def on_epoch_end(self, epoch: int, metrics: dict[str, float]):
        self.epochs_ended.append((epoch, metrics))

    def on_training_end(self, final_metrics: dict[str, float]):
        self.training_ended = True


@pytest.fixture
def simple_data():
    """Create simple training data."""
    X = torch.randn(100, 10)
    y = torch.randn(100, 1)
    dataset = TensorDataset(X, y)
    train_loader = DataLoader(dataset, batch_size=10, shuffle=False)
    return train_loader


@pytest.fixture
def simple_config():
    """Create simple training configuration."""
    return SimpleNamespace(
        epochs=3,
        batch_size=10,
        learning_rate=0.001,
        device="cpu",
        optimizer="adam",
        optimizer_params={},
        loss_function="mse",
        loss_params={},
        reshape_targets=False,
        target_output_key=None,
        early_stopping_patience=None,
        gradient_clip_val=None,
    )


class TestGetOptimizer:
    """Tests for _get_optimizer helper function."""

    def test_adam_optimizer(self):
        """Test creating Adam optimizer."""
        model = SimpleModel()
        optimizer = _get_optimizer("adam", model.parameters(), learning_rate=0.001)
        assert isinstance(optimizer, torch.optim.Adam)
        assert optimizer.param_groups[0]["lr"] == 0.001

    def test_sgd_optimizer(self):
        """Test creating SGD optimizer."""
        model = SimpleModel()
        optimizer = _get_optimizer("sgd", model.parameters(), learning_rate=0.01)
        assert isinstance(optimizer, torch.optim.SGD)
        assert optimizer.param_groups[0]["lr"] == 0.01

    def test_adamw_optimizer(self):
        """Test creating AdamW optimizer."""
        model = SimpleModel()
        optimizer = _get_optimizer("adamw", model.parameters(), learning_rate=0.001)
        assert isinstance(optimizer, torch.optim.AdamW)

    def test_rmsprop_optimizer(self):
        """Test creating RMSprop optimizer."""
        model = SimpleModel()
        optimizer = _get_optimizer("rmsprop", model.parameters(), learning_rate=0.001)
        assert isinstance(optimizer, torch.optim.RMSprop)

    def test_optimizer_with_params(self):
        """Test creating optimizer with additional parameters."""
        model = SimpleModel()
        optimizer = _get_optimizer(
            "adam", model.parameters(), learning_rate=0.001, weight_decay=0.01
        )
        assert isinstance(optimizer, torch.optim.Adam)
        assert optimizer.param_groups[0]["weight_decay"] == 0.01

    def test_unsupported_optimizer(self):
        """Test that unsupported optimizer raises error."""
        model = SimpleModel()
        with pytest.raises(ValueError, match="Unsupported optimizer type"):
            _get_optimizer("invalid_optimizer", model.parameters(), learning_rate=0.001)

    def test_optimizer_case_insensitive(self):
        """Test that optimizer type is case insensitive."""
        model = SimpleModel()
        optimizer = _get_optimizer("ADAM", model.parameters(), learning_rate=0.001)
        assert isinstance(optimizer, torch.optim.Adam)


class TestGetLossFunction:
    """Tests for _get_loss_function helper function."""

    def test_mse_loss(self):
        """Test creating MSE loss."""
        loss_fn = _get_loss_function("mse")
        assert isinstance(loss_fn, nn.MSELoss)

    def test_mse_loss_alias(self):
        """Test creating MSE loss with alias."""
        loss_fn = _get_loss_function("mean_squared_error")
        assert isinstance(loss_fn, nn.MSELoss)

    def test_mae_loss(self):
        """Test creating MAE loss."""
        loss_fn = _get_loss_function("mae")
        assert isinstance(loss_fn, nn.L1Loss)

    def test_bce_loss(self):
        """Test creating BCE loss."""
        loss_fn = _get_loss_function("bce")
        assert isinstance(loss_fn, nn.BCELoss)

    def test_bce_with_logits_loss(self):
        """Test creating BCE with logits loss."""
        loss_fn = _get_loss_function("bce_with_logits")
        assert isinstance(loss_fn, nn.BCEWithLogitsLoss)

    def test_cross_entropy_loss(self):
        """Test creating cross entropy loss."""
        loss_fn = _get_loss_function("cross_entropy")
        assert isinstance(loss_fn, nn.CrossEntropyLoss)

    def test_nll_loss(self):
        """Test creating NLL loss."""
        loss_fn = _get_loss_function("nll")
        assert isinstance(loss_fn, nn.NLLLoss)

    def test_unsupported_loss(self):
        """Test that unsupported loss raises error."""
        with pytest.raises(ValueError, match="Unsupported loss function"):
            _get_loss_function("invalid_loss")

    def test_loss_case_insensitive(self):
        """Test that loss type is case insensitive."""
        loss_fn = _get_loss_function("MSE")
        assert isinstance(loss_fn, nn.MSELoss)


class TestTrainModel:
    """Tests for train_model function."""

    def test_basic_training(self, simple_data, simple_config):
        """Test basic training loop execution."""
        model = SimpleModel()
        callback = MockCallback()

        result, optimizer = train_model(
            model=model,
            train_loader=simple_data,
            training_config=simple_config,
            callback=callback,
        )

        # Check result
        assert result.success is True
        assert len(result.train_losses) == 3  # 3 epochs
        assert result.final_train_loss is not None
        assert result.training_time is not None
        assert result.total_epochs == 3

        # Check optimizer
        assert isinstance(optimizer, torch.optim.Adam)

        # Check callback invocations
        assert callback.training_started is True
        assert callback.training_ended is True
        assert len(callback.epochs_started) == 3
        assert len(callback.epochs_ended) == 3

    def test_training_with_validation(self, simple_data, simple_config):
        """Test training with validation data."""
        model = SimpleModel()

        result, optimizer = train_model(
            model=model,
            train_loader=simple_data,
            training_config=simple_config,
            val_loader=simple_data,  # Use same data for simplicity
        )

        assert result.success is True
        assert result.val_losses is not None
        assert len(result.val_losses) == 3
        assert result.final_val_loss is not None
        assert result.best_val_loss is not None
        assert result.best_epoch is not None

    def test_early_stopping(self, simple_data):
        """Test early stopping functionality."""
        model = SimpleModel()
        config = SimpleNamespace(
            epochs=100,  # Set high but expect early stop
            batch_size=10,
            learning_rate=0.001,
            device="cpu",
            optimizer="adam",
            optimizer_params={},
            loss_function="mse",
            loss_params={},
            reshape_targets=False,
            target_output_key=None,
            early_stopping_patience=2,  # Stop after 2 epochs without improvement
            gradient_clip_val=None,
        )

        result, optimizer = train_model(
            model=model,
            train_loader=simple_data,
            training_config=config,
            val_loader=simple_data,
        )

        assert result.success is True
        # With random data, the model might train all epochs or stop early
        # Just verify the mechanism works (validation losses are tracked)
        assert result.val_losses is not None
        assert result.best_val_loss is not None
        assert result.best_epoch is not None

    def test_gradient_clipping(self, simple_data):
        """Test gradient clipping functionality."""
        model = SimpleModel()
        config = SimpleNamespace(
            epochs=2,
            batch_size=10,
            learning_rate=0.001,
            device="cpu",
            optimizer="adam",
            optimizer_params={},
            loss_function="mse",
            loss_params={},
            reshape_targets=False,
            target_output_key=None,
            early_stopping_patience=None,
            gradient_clip_val=1.0,  # Clip gradients
        )

        result, optimizer = train_model(
            model=model,
            train_loader=simple_data,
            training_config=config,
        )

        assert result.success is True

    def test_cancellation(self, simple_data, simple_config):
        """Test training cancellation via stop_event."""
        model = SimpleModel()
        stop_event = Event()

        # Set the event before training starts (immediate cancellation)
        stop_event.set()

        result, optimizer = train_model(
            model=model,
            train_loader=simple_data,
            training_config=simple_config,
            stop_event=stop_event,
        )

        assert result.success is False
        assert "cancelled" in result.error_message.lower()

    def test_checkpoint_saving(self, simple_data, simple_config):
        """Test checkpoint saving to directory."""
        model = SimpleModel()

        with tempfile.TemporaryDirectory() as tmpdir:
            checkpoint_dir = Path(tmpdir) / "checkpoints"

            result, optimizer = train_model(
                model=model,
                train_loader=simple_data,
                training_config=simple_config,
                val_loader=simple_data,
                checkpoint_dir=checkpoint_dir,
            )

            assert result.success is True
            # Check that checkpoint was saved
            checkpoint_path = checkpoint_dir / "best_model.pt"
            assert checkpoint_path.exists()

            # Load checkpoint and verify structure
            checkpoint = torch.load(checkpoint_path, weights_only=False)
            assert "model_state_dict" in checkpoint
            assert "optimizer_state_dict" in checkpoint
            assert "epoch" in checkpoint
            assert "train_loss" in checkpoint
            assert "val_loss" in checkpoint

    def test_training_with_dict_outputs(self, simple_data):
        """Test training with model that outputs dictionary."""

        class DictOutputModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.fc = nn.Linear(10, 1)

            def forward(self, x):
                output = self.fc(x)
                return {"logits": output, "other": output * 2}

        model = DictOutputModel()
        config = SimpleNamespace(
            epochs=2,
            batch_size=10,
            learning_rate=0.001,
            device="cpu",
            optimizer="adam",
            optimizer_params={},
            loss_function="mse",
            loss_params={},
            reshape_targets=False,
            target_output_key="logits",  # Extract specific output
            early_stopping_patience=None,
            gradient_clip_val=None,
        )

        result, optimizer = train_model(
            model=model,
            train_loader=simple_data,
            training_config=config,
        )

        assert result.success is True

    def test_training_with_target_reshaping(self):
        """Test training with target reshaping."""
        # Create 1D targets that need reshaping
        X = torch.randn(100, 10)
        y = torch.randn(100)  # 1D targets
        dataset = TensorDataset(X, y)
        train_loader = DataLoader(dataset, batch_size=10, shuffle=False)

        model = SimpleModel()
        config = SimpleNamespace(
            epochs=2,
            batch_size=10,
            learning_rate=0.001,
            device="cpu",
            optimizer="adam",
            optimizer_params={},
            loss_function="mse",
            loss_params={},
            reshape_targets=True,  # Reshape 1D to 2D
            target_output_key=None,
            early_stopping_patience=None,
            gradient_clip_val=None,
        )

        result, optimizer = train_model(
            model=model,
            train_loader=train_loader,
            training_config=config,
        )

        assert result.success is True

    def test_training_error_handling(self, simple_data):
        """Test that training errors are properly caught and reported."""

        class BrokenModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.fc = nn.Linear(10, 1)

            def forward(self, x):
                raise RuntimeError("Intentional error for testing")

        model = BrokenModel()
        config = SimpleNamespace(
            epochs=2,
            batch_size=10,
            learning_rate=0.001,
            device="cpu",
            optimizer="adam",
            optimizer_params={},
            loss_function="mse",
            loss_params={},
            reshape_targets=False,
            target_output_key=None,
            early_stopping_patience=None,
            gradient_clip_val=None,
        )

        result, optimizer = train_model(
            model=model,
            train_loader=simple_data,
            training_config=config,
        )

        assert result.success is False
        assert result.error_message is not None
        assert "Intentional error" in result.error_message


class TestTrainingResult:
    """Tests for TrainingResult dataclass."""

    def test_successful_result(self):
        """Test creating a successful training result."""
        result = TrainingResult(
            success=True,
            train_losses=[0.5, 0.4, 0.3],
            val_losses=[0.6, 0.5, 0.4],
            final_train_loss=0.3,
            final_val_loss=0.4,
            best_val_loss=0.4,
            training_time=10.5,
            total_epochs=3,
            best_epoch=3,
        )

        assert result.success is True
        assert len(result.train_losses) == 3
        assert result.error_message is None

    def test_failed_result(self):
        """Test creating a failed training result."""
        result = TrainingResult(
            success=False,
            train_losses=[0.5],
            error_message="Training failed due to error",
        )

        assert result.success is False
        assert result.error_message is not None
        assert len(result.train_losses) == 1
