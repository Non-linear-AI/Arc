"""Unit tests for training.py helper functions and training execution."""

import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock, Mock, patch

import torch
import torch.nn as nn

from arc.ml.training import (
    TrainingResult,
    _get_loss_function,
    _get_optimizer,
    _normalize_device,
    _sanitize_optimizer_params,
    train_model,
)


class TestNormalizeDevice(unittest.TestCase):
    """Test device normalization function."""

    def test_normalize_device_cpu(self):
        """Test CPU device normalization."""
        self.assertEqual(_normalize_device("cpu"), "cpu")
        self.assertEqual(_normalize_device("CPU"), "cpu")
        self.assertEqual(_normalize_device("  cpu  "), "cpu")

    def test_normalize_device_cuda(self):
        """Test CUDA device normalization."""
        self.assertEqual(_normalize_device("cuda"), "cuda")
        self.assertEqual(_normalize_device("CUDA"), "cuda")

    def test_normalize_device_mps(self):
        """Test MPS device normalization."""
        self.assertEqual(_normalize_device("mps"), "mps")
        self.assertEqual(_normalize_device("MPS"), "mps")

    @patch("torch.cuda.is_available", return_value=True)
    def test_normalize_device_auto_cuda_available(self, mock_cuda):
        """Test auto device selection when CUDA is available."""
        result = _normalize_device("auto")
        self.assertEqual(result, "cuda")

    @patch("torch.cuda.is_available", return_value=False)
    @patch("torch.backends.mps.is_available", return_value=True)
    def test_normalize_device_auto_mps_available(self, mock_mps, mock_cuda):
        """Test auto device selection when MPS is available but CUDA isn't."""
        result = _normalize_device("auto")
        self.assertEqual(result, "mps")

    @patch("torch.cuda.is_available", return_value=False)
    @patch("torch.backends.mps.is_available", return_value=False)
    def test_normalize_device_auto_fallback_cpu(self, mock_mps, mock_cuda):
        """Test auto device selection falls back to CPU."""
        result = _normalize_device("auto")
        self.assertEqual(result, "cpu")

    def test_normalize_device_unknown_defaults_to_cpu(self):
        """Test unknown device defaults to CPU with warning."""
        with self.assertLogs("arc.ml.training", level="WARNING") as cm:
            result = _normalize_device("unknown_device")
            self.assertEqual(result, "cpu")
            self.assertIn("Unknown device 'unknown_device'", cm.output[0])


class TestSanitizeOptimizerParams(unittest.TestCase):
    """Test optimizer parameter sanitization."""

    def test_sanitize_float_strings(self):
        """Test conversion of string floats to floats."""
        params = {"lr": "0.001", "weight_decay": "1e-5"}
        result = _sanitize_optimizer_params(params)
        self.assertEqual(result["lr"], 0.001)
        self.assertEqual(result["weight_decay"], 1e-5)

    def test_sanitize_scientific_notation(self):
        """Test conversion of scientific notation strings."""
        params = {"weight_decay": "1e-5", "eps": "1e-8"}
        result = _sanitize_optimizer_params(params)
        self.assertAlmostEqual(result["weight_decay"], 0.00001)
        self.assertAlmostEqual(result["eps"], 0.00000001)

    def test_sanitize_keeps_non_numeric_strings(self):
        """Test non-numeric strings are kept as-is."""
        params = {"name": "adam", "mode": "min"}
        result = _sanitize_optimizer_params(params)
        self.assertEqual(result["name"], "adam")
        self.assertEqual(result["mode"], "min")

    def test_sanitize_keeps_numeric_values(self):
        """Test numeric values are unchanged."""
        params = {"lr": 0.001, "weight_decay": 1e-5, "betas": (0.9, 0.999)}
        result = _sanitize_optimizer_params(params)
        self.assertEqual(result["lr"], 0.001)
        self.assertEqual(result["weight_decay"], 1e-5)
        self.assertEqual(result["betas"], (0.9, 0.999))


class TestGetOptimizer(unittest.TestCase):
    """Test optimizer factory function."""

    def setUp(self):
        """Set up test model."""
        self.model = nn.Linear(10, 1)

    def test_get_optimizer_adam(self):
        """Test getting Adam optimizer."""
        optimizer = _get_optimizer("adam", self.model.parameters(), learning_rate=0.001)
        self.assertIsInstance(optimizer, torch.optim.Adam)

    def test_get_optimizer_sgd(self):
        """Test getting SGD optimizer."""
        optimizer = _get_optimizer("sgd", self.model.parameters(), learning_rate=0.01)
        self.assertIsInstance(optimizer, torch.optim.SGD)

    def test_get_optimizer_adamw(self):
        """Test getting AdamW optimizer."""
        optimizer = _get_optimizer(
            "adamw", self.model.parameters(), learning_rate=0.001
        )
        self.assertIsInstance(optimizer, torch.optim.AdamW)

    def test_get_optimizer_rmsprop(self):
        """Test getting RMSprop optimizer."""
        optimizer = _get_optimizer(
            "rmsprop", self.model.parameters(), learning_rate=0.01
        )
        self.assertIsInstance(optimizer, torch.optim.RMSprop)

    def test_get_optimizer_with_full_path(self):
        """Test getting optimizer with full PyTorch path."""
        optimizer = _get_optimizer(
            "torch.optim.Adam", self.model.parameters(), learning_rate=0.001
        )
        self.assertIsInstance(optimizer, torch.optim.Adam)

    def test_get_optimizer_case_insensitive(self):
        """Test optimizer type is case insensitive."""
        optimizer = _get_optimizer("ADAM", self.model.parameters(), learning_rate=0.001)
        self.assertIsInstance(optimizer, torch.optim.Adam)

    def test_get_optimizer_with_params(self):
        """Test optimizer with additional parameters."""
        optimizer = _get_optimizer(
            "adam",
            self.model.parameters(),
            learning_rate=0.001,
            weight_decay=1e-5,
            eps=1e-8,
        )
        self.assertIsInstance(optimizer, torch.optim.Adam)

    def test_get_optimizer_sanitizes_params(self):
        """Test optimizer parameter sanitization."""
        # Pass weight_decay as string (YAML scientific notation issue)
        optimizer = _get_optimizer(
            "adam",
            self.model.parameters(),
            learning_rate=0.001,
            weight_decay="1e-5",  # String instead of float
        )
        self.assertIsInstance(optimizer, torch.optim.Adam)

    def test_get_optimizer_unsupported_raises(self):
        """Test unsupported optimizer raises ValueError."""
        with self.assertRaises(ValueError) as cm:
            _get_optimizer("unsupported", self.model.parameters(), learning_rate=0.001)
        self.assertIn("Unsupported optimizer type", str(cm.exception))


class TestGetLossFunction(unittest.TestCase):
    """Test loss function factory."""

    def test_get_loss_mse(self):
        """Test getting MSE loss."""
        loss = _get_loss_function("mse")
        self.assertIsInstance(loss, nn.MSELoss)

    def test_get_loss_mae(self):
        """Test getting MAE loss."""
        loss = _get_loss_function("mae")
        self.assertIsInstance(loss, nn.L1Loss)

    def test_get_loss_bce(self):
        """Test getting BCE loss."""
        loss = _get_loss_function("bce")
        self.assertIsInstance(loss, nn.BCELoss)

    def test_get_loss_bce_with_logits(self):
        """Test getting BCE with logits loss."""
        loss = _get_loss_function("bce_with_logits")
        self.assertIsInstance(loss, nn.BCEWithLogitsLoss)

    def test_get_loss_cross_entropy(self):
        """Test getting cross entropy loss."""
        loss = _get_loss_function("cross_entropy")
        self.assertIsInstance(loss, nn.CrossEntropyLoss)

    def test_get_loss_nll(self):
        """Test getting NLL loss."""
        loss = _get_loss_function("nll")
        self.assertIsInstance(loss, nn.NLLLoss)

    def test_get_loss_with_full_path(self):
        """Test getting loss with full PyTorch path."""
        loss = _get_loss_function(
            "torch.nn.functional.binary_cross_entropy_with_logits"
        )
        self.assertIsInstance(loss, nn.BCEWithLogitsLoss)

    def test_get_loss_with_nn_path(self):
        """Test getting loss with torch.nn path."""
        loss = _get_loss_function("torch.nn.MSELoss")
        self.assertIsInstance(loss, nn.MSELoss)

    def test_get_loss_case_insensitive(self):
        """Test loss type is case insensitive."""
        loss = _get_loss_function("MSE")
        self.assertIsInstance(loss, nn.MSELoss)

    def test_get_loss_with_params(self):
        """Test loss function with parameters."""
        loss = _get_loss_function("mse", reduction="sum")
        self.assertIsInstance(loss, nn.MSELoss)
        self.assertEqual(loss.reduction, "sum")

    def test_get_loss_unsupported_raises(self):
        """Test unsupported loss raises ValueError."""
        with self.assertRaises(ValueError) as cm:
            _get_loss_function("unsupported_loss")
        self.assertIn("Unsupported loss function", str(cm.exception))


class TestTrainModel(unittest.TestCase):
    """Test train_model function with mocks."""

    def setUp(self):
        """Set up mock model and data loader."""
        # Create simple model
        self.model = nn.Linear(10, 1)

        # Create mock data loader
        self.train_loader = MagicMock()
        # Mock iteration to return batches
        batch = (torch.randn(32, 10), torch.randn(32, 1))
        self.train_loader.__iter__ = Mock(return_value=iter([batch, batch]))
        self.train_loader.__len__ = Mock(return_value=2)

        # Create training config
        self.training_config = SimpleNamespace(
            device="cpu",
            epochs=2,
            learning_rate=0.001,
            optimizer="adam",
            optimizer_params={},
            loss_function="mse",
            loss_params={},
            early_stopping_patience=None,
            gradient_clip_val=None,
            reshape_targets=False,
            target_output_key=None,
        )

    def test_train_model_basic_success(self):
        """Test basic successful training."""
        result, optimizer = train_model(
            model=self.model,
            train_loader=self.train_loader,
            training_config=self.training_config,
        )

        self.assertIsInstance(result, TrainingResult)
        self.assertTrue(result.success)
        self.assertEqual(len(result.train_losses), 2)  # 2 epochs
        self.assertIsNotNone(result.final_train_loss)
        self.assertIsNotNone(result.training_time)
        self.assertEqual(result.total_epochs, 2)
        self.assertIsInstance(optimizer, torch.optim.Adam)

    def test_train_model_with_validation(self):
        """Test training with validation data."""
        # Create validation loader
        val_batch = (torch.randn(16, 10), torch.randn(16, 1))
        val_loader = MagicMock()
        val_loader.__iter__ = Mock(return_value=iter([val_batch]))
        val_loader.__len__ = Mock(return_value=1)

        result, optimizer = train_model(
            model=self.model,
            train_loader=self.train_loader,
            training_config=self.training_config,
            val_loader=val_loader,
        )

        self.assertTrue(result.success)
        self.assertIsNotNone(result.val_losses)
        self.assertEqual(len(result.val_losses), 2)
        self.assertIsNotNone(result.final_val_loss)
        self.assertIsNotNone(result.best_val_loss)

    def test_train_model_with_callback(self):
        """Test training with progress callback."""
        callback = MagicMock()

        result, optimizer = train_model(
            model=self.model,
            train_loader=self.train_loader,
            training_config=self.training_config,
            callback=callback,
        )

        self.assertTrue(result.success)
        # Verify callback methods were called
        callback.on_training_start.assert_called_once()
        callback.on_training_end.assert_called_once()
        self.assertEqual(callback.on_epoch_start.call_count, 2)  # 2 epochs
        self.assertEqual(callback.on_epoch_end.call_count, 2)

    def test_train_model_with_auto_device(self):
        """Test training with auto device selection."""
        self.training_config.device = "auto"

        with (
            patch("torch.cuda.is_available", return_value=False),
            patch("torch.backends.mps.is_available", return_value=False),
        ):
            result, optimizer = train_model(
                model=self.model,
                train_loader=self.train_loader,
                training_config=self.training_config,
            )

        self.assertTrue(result.success)

    def test_train_model_with_full_optimizer_path(self):
        """Test training with full PyTorch optimizer path."""
        self.training_config.optimizer = "torch.optim.SGD"

        result, optimizer = train_model(
            model=self.model,
            train_loader=self.train_loader,
            training_config=self.training_config,
        )

        self.assertTrue(result.success)
        self.assertIsInstance(optimizer, torch.optim.SGD)

    def test_train_model_with_full_loss_path(self):
        """Test training with full PyTorch loss path."""
        self.training_config.loss_function = "torch.nn.L1Loss"

        result, optimizer = train_model(
            model=self.model,
            train_loader=self.train_loader,
            training_config=self.training_config,
        )

        self.assertTrue(result.success)

    def test_train_model_with_early_stopping(self):
        """Test training with early stopping."""
        # Create validation loader
        val_batch = (torch.randn(16, 10), torch.randn(16, 1))
        val_loader = MagicMock()
        val_loader.__iter__ = Mock(return_value=iter([val_batch]))
        val_loader.__len__ = Mock(return_value=1)

        self.training_config.early_stopping_patience = 1
        self.training_config.epochs = 10  # Would run 10 but should stop early

        result, optimizer = train_model(
            model=self.model,
            train_loader=self.train_loader,
            training_config=self.training_config,
            val_loader=val_loader,
        )

        self.assertTrue(result.success)
        # Should stop before 10 epochs
        self.assertLess(result.total_epochs, 10)

    def test_train_model_with_stop_event(self):
        """Test training can be cancelled with stop event."""
        from threading import Event

        stop_event = Event()

        # Create a callback that sets the stop event after first epoch
        def stop_after_first_epoch(epoch, total_epochs):
            if epoch == 1:
                stop_event.set()

        callback = MagicMock()
        callback.on_epoch_end = Mock(side_effect=stop_after_first_epoch)

        result, optimizer = train_model(
            model=self.model,
            train_loader=self.train_loader,
            training_config=self.training_config,
            callback=callback,
            stop_event=stop_event,
        )

        self.assertFalse(result.success)
        self.assertIn("cancelled", result.error_message.lower())

    def test_train_model_error_handling(self):
        """Test training error handling."""
        # Create a broken data loader that raises an error
        broken_loader = MagicMock()
        broken_loader.__iter__ = Mock(side_effect=RuntimeError("Data loading failed"))

        result, optimizer = train_model(
            model=self.model,
            train_loader=broken_loader,
            training_config=self.training_config,
        )

        self.assertFalse(result.success)
        self.assertIsNotNone(result.error_message)
        self.assertIn("Data loading failed", result.error_message)


if __name__ == "__main__":
    unittest.main()
