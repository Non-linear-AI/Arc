"""PyTorch model training implementation for Arc Graph."""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from threading import Event
from typing import Any, Protocol

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from arc.graph.trainer import TrainingConfig
from arc.ml.metrics import MetricsTracker
from arc.plugins import get_plugin_manager

logger = logging.getLogger(__name__)


class TrainingCancelledError(RuntimeError):
    """Raised when training is cancelled via stop event."""


class ProgressCallback(Protocol):
    """Protocol for training progress callbacks."""

    def on_epoch_start(self, epoch: int, total_epochs: int) -> None:
        """Called at the start of each epoch."""
        ...

    def on_epoch_end(self, epoch: int, metrics: dict[str, float]) -> None:
        """Called at the end of each epoch."""
        ...

    def on_batch_end(self, batch: int, total_batches: int, loss: float) -> None:
        """Called at the end of each batch."""
        ...

    def on_training_start(self) -> None:
        """Called when training starts."""
        ...

    def on_training_end(self, final_metrics: dict[str, float]) -> None:
        """Called when training ends."""
        ...


@dataclass
class TrainingResult:
    """Results from model training."""

    # Training completion
    success: bool
    total_epochs: int
    best_epoch: int

    # Final metrics
    final_train_loss: float
    final_val_loss: float | None = None
    best_val_loss: float | None = None

    # Training history
    train_losses: list[float] = field(default_factory=list)
    val_losses: list[float] = field(default_factory=list)
    metrics_history: dict[str, list[float]] = field(default_factory=dict)

    # Timing
    training_time: float = 0.0

    # Model checkpoints
    checkpoint_path: str | None = None
    best_model_path: str | None = None

    # Error information
    error_message: str | None = None


class ArcTrainer:
    """PyTorch model trainer for Arc Graph models."""

    def __init__(self, config: TrainingConfig):
        """Initialize trainer with configuration.

        Args:
            config: Training configuration
        """
        self.config = config
        self.device = self._setup_device()
        self.metrics_tracker = MetricsTracker()

        # Training state
        self.model: nn.Module | None = None
        self.optimizer: torch.optim.Optimizer | None = None
        self.loss_fn: nn.Module | None = None
        self.best_val_loss: float = float("inf")
        self.early_stopping_counter = 0

        logger.info(f"Trainer initialized with device: {self.device}")

    def _setup_device(self) -> torch.device:
        """Setup training device based on configuration."""
        if self.config.device == "auto":
            if torch.cuda.is_available():
                device = torch.device("cuda")
            elif hasattr(torch, "backends") and torch.backends.mps.is_available():
                device = torch.device("mps")
            else:
                device = torch.device("cpu")
        else:
            device = torch.device(self.config.device)

        logger.info(f"Using device: {device}")
        return device

    def _setup_optimizer(self, model: nn.Module) -> torch.optim.Optimizer:
        """Setup optimizer using exact PyTorch names from plugin system."""
        optimizer_name = self.config.optimizer

        # Start with optimizer params from config
        params = dict(self.config.optimizer_params)

        # Convert learning_rate to lr if present in optimizer_params
        if "learning_rate" in params:
            params["lr"] = params.pop("learning_rate")
        else:
            params["lr"] = self.config.learning_rate

        # Get optimizer class from plugin system
        pm = get_plugin_manager()
        optimizer_class = pm.get_optimizer(optimizer_name)

        if optimizer_class is None:
            optimizers = pm.get_optimizers()
            raise ValueError(
                f"Unsupported optimizer: {optimizer_name}. "
                f"Available: {list(optimizers.keys())}"
            )
        return optimizer_class(model.parameters(), **params)

    def _setup_loss_function(self) -> nn.Module:
        """Setup loss function using exact PyTorch names from plugin system."""
        loss_name = self.config.loss_function

        params = self.config.loss_params

        # Get loss class from plugin system
        pm = get_plugin_manager()
        loss_class = pm.get_loss(loss_name)

        if loss_class is None:
            losses = pm.get_losses()
            raise ValueError(
                f"Unsupported loss function: {loss_name}. "
                f"Available: {list(losses.keys())}"
            )

        return loss_class(**params)

    def train(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader | None = None,
        callback: ProgressCallback | None = None,
        checkpoint_dir: str | Path | None = None,
        stop_event: Event | None = None,
    ) -> TrainingResult:
        """Train a PyTorch model.

        Args:
            model: PyTorch model to train
            train_loader: Training data loader
            val_loader: Optional validation data loader
            callback: Optional progress callback
            checkpoint_dir: Directory to save checkpoints

        Returns:
            Training results
        """
        start_time = time.time()

        # Initialize result tracking first
        result = TrainingResult(
            success=False,
            total_epochs=self.config.epochs,
            best_epoch=0,
            final_train_loss=0.0,
        )

        try:
            # Setup training components
            self.model = model.to(self.device)
            self.optimizer = self._setup_optimizer(model)
            self.loss_fn = self._setup_loss_function()

            # Setup checkpoint directory
            if checkpoint_dir:
                checkpoint_dir = Path(checkpoint_dir)
                checkpoint_dir.mkdir(parents=True, exist_ok=True)

            # Start training
            if callback:
                callback.on_training_start()

            logger.info(f"Starting training for {self.config.epochs} epochs")

            for epoch in range(self.config.epochs):
                if stop_event and stop_event.is_set():
                    raise TrainingCancelledError("Training cancelled before epoch")

                if callback:
                    callback.on_epoch_start(epoch + 1, self.config.epochs)

                # Training phase
                train_loss = self._train_epoch(train_loader, callback, stop_event)
                result.train_losses.append(train_loss)

                # Validation phase
                val_loss = None
                if val_loader:
                    val_loss = self._validate_epoch(val_loader, stop_event)
                    result.val_losses.append(val_loss)

                # Compute additional metrics
                epoch_metrics = self._compute_epoch_metrics(
                    epoch + 1, train_loss, val_loss
                )

                # Update metrics history
                for metric_name, value in epoch_metrics.items():
                    if metric_name not in result.metrics_history:
                        result.metrics_history[metric_name] = []
                    result.metrics_history[metric_name].append(value)

                # Check for best model (only if we have validation data)
                if val_loss is not None:
                    if val_loss < self.best_val_loss:
                        self.best_val_loss = val_loss
                        result.best_epoch = epoch + 1
                        result.best_val_loss = val_loss
                    current_val_loss = val_loss
                else:
                    # No validation data, use train loss for checkpointing decisions
                    current_val_loss = train_loss

                # Save best model if this is the best epoch
                if (
                    (
                        (val_loss is not None and val_loss == self.best_val_loss)
                        or val_loss is None
                    )
                    and checkpoint_dir
                    and self.config.save_best_only
                ):
                    best_path = checkpoint_dir / "best_model.pt"
                    self._save_checkpoint(best_path, epoch + 1, current_val_loss)
                    result.best_model_path = str(best_path)

                # Regular checkpointing
                if checkpoint_dir and (epoch + 1) % self.config.checkpoint_every == 0:
                    checkpoint_path = (
                        checkpoint_dir / f"checkpoint_epoch_{epoch + 1}.pt"
                    )
                    self._save_checkpoint(checkpoint_path, epoch + 1, current_val_loss)
                    result.checkpoint_path = str(checkpoint_path)

                # Early stopping check
                if self._should_early_stop(current_val_loss):
                    logger.info(f"Early stopping at epoch {epoch + 1}")
                    break

                # Epoch end callback
                if callback:
                    callback.on_epoch_end(epoch + 1, epoch_metrics)

                if self.config.verbose:
                    self._log_epoch_results(epoch + 1, epoch_metrics)

            # Training completed successfully
            result.success = True
            result.final_train_loss = result.train_losses[-1]
            if result.val_losses:
                result.final_val_loss = result.val_losses[-1]

            result.training_time = time.time() - start_time

            if callback:
                callback.on_training_end(epoch_metrics)

            logger.info(f"Training completed in {result.training_time:.2f}s")
            return result

        except TrainingCancelledError as cancel_err:
            result.success = False
            result.error_message = str(cancel_err) or "Training cancelled"
            result.training_time = time.time() - start_time

            logger.info("Training cancelled after %.2fs", result.training_time)
            return result

        except Exception as e:
            # Training failed
            result.success = False
            result.error_message = str(e)
            result.training_time = time.time() - start_time

            logger.error(f"Training failed: {e}")
            raise

    def _train_epoch(
        self,
        train_loader: DataLoader,
        callback: ProgressCallback | None,
        stop_event: Event | None,
    ) -> float:
        """Train for one epoch."""
        self.model.train()
        total_loss = 0.0
        num_batches = len(train_loader)

        for batch_idx, (data, target) in enumerate(train_loader):
            if stop_event and stop_event.is_set():
                raise TrainingCancelledError("Training cancelled during batch")

            data, target = data.to(self.device), target.to(self.device)

            # Reshape targets if configured
            if self.config.reshape_targets and target.dim() == 1:
                target = target.unsqueeze(1)

            # Forward pass
            self.optimizer.zero_grad()
            output = self.model(data)

            # Extract target output if needed
            if self.config.target_output_key is not None:
                if isinstance(output, dict):
                    if self.config.target_output_key not in output:
                        available_keys = list(output.keys())
                        key_name = self.config.target_output_key
                        raise RuntimeError(
                            f"Target output key '{key_name}' not found. "
                            f"Available keys: {available_keys}"
                        )
                    output = output[self.config.target_output_key]
                else:
                    key_name = self.config.target_output_key
                    raise RuntimeError(
                        f"target_output_key '{key_name}' specified "
                        f"but model output is not a dictionary"
                    )
            loss = self.loss_fn(output, target)

            # Backward pass
            loss.backward()
            self.optimizer.step()

            total_loss += loss.item()

            # Batch callback
            if callback and (batch_idx + 1) % self.config.log_every == 0:
                callback.on_batch_end(batch_idx + 1, num_batches, loss.item())

        return total_loss / num_batches

    def _validate_epoch(
        self, val_loader: DataLoader, stop_event: Event | None
    ) -> float:
        """Validate for one epoch."""
        self.model.eval()
        total_loss = 0.0
        num_batches = len(val_loader)

        with torch.no_grad():
            for data, target in val_loader:
                if stop_event and stop_event.is_set():
                    raise TrainingCancelledError("Training cancelled during validation")

                data, target = data.to(self.device), target.to(self.device)

                # Reshape targets if configured
                if self.config.reshape_targets and target.dim() == 1:
                    target = target.unsqueeze(1)

                output = self.model(data)

                # Extract target output if needed
                if self.config.target_output_key is not None:
                    if isinstance(output, dict):
                        if self.config.target_output_key not in output:
                            available_keys = list(output.keys())
                            key_name = self.config.target_output_key
                            raise RuntimeError(
                                f"Target output key '{key_name}' not found. "
                                f"Available keys: {available_keys}"
                            )
                        output = output[self.config.target_output_key]
                    else:
                        key_name = self.config.target_output_key
                        raise RuntimeError(
                            f"target_output_key '{key_name}' specified "
                            f"but model output is not a dictionary"
                        )
                loss = self.loss_fn(output, target)
                total_loss += loss.item()

        return total_loss / num_batches

    def _compute_epoch_metrics(
        self, epoch: int, train_loss: float, val_loss: float | None
    ) -> dict[str, float]:
        """Compute metrics for the current epoch."""
        metrics = {
            "epoch": epoch,
            "train_loss": train_loss,
        }

        if val_loss is not None:
            metrics["val_loss"] = val_loss

        # Add any additional metrics from metrics tracker
        metrics.update(self.metrics_tracker.get_current_metrics())

        return metrics

    def _should_early_stop(self, current_val_loss: float) -> bool:
        """Check if training should stop early."""
        if self.config.early_stopping_patience is None:
            return False

        if current_val_loss < self.best_val_loss - self.config.early_stopping_min_delta:
            self.early_stopping_counter = 0
        else:
            self.early_stopping_counter += 1

        return self.early_stopping_counter >= self.config.early_stopping_patience

    def _save_checkpoint(self, path: Path, epoch: int, val_loss: float) -> None:
        """Save model checkpoint."""
        checkpoint = {
            "epoch": epoch,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "val_loss": val_loss,
            "config": self.config,
        }

        torch.save(checkpoint, path)
        logger.info(f"Checkpoint saved: {path}")

    def _log_epoch_results(self, epoch: int, metrics: dict[str, float]) -> None:
        """Log epoch results."""
        metrics_str = " | ".join(
            [
                f"{k}: {v:.4f}"
                for k, v in metrics.items()
                if isinstance(v, (int, float)) and k != "epoch"
            ]
        )
        logger.info(f"Epoch {epoch}: {metrics_str}")

    def load_checkpoint(self, checkpoint_path: str | Path) -> dict[str, Any]:
        """Load model checkpoint.

        Args:
            checkpoint_path: Path to checkpoint file

        Returns:
            Checkpoint metadata
        """
        checkpoint = torch.load(
            checkpoint_path, map_location=self.device, weights_only=False
        )

        if self.model is not None:
            self.model.load_state_dict(checkpoint["model_state_dict"])

        if self.optimizer is not None:
            self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

        logger.info(f"Checkpoint loaded: {checkpoint_path}")

        return {
            "epoch": checkpoint["epoch"],
            "val_loss": checkpoint["val_loss"],
            "config": checkpoint.get("config"),
        }


class SimpleProgressCallback:
    """Simple console progress callback."""

    def __init__(self, log_every: int = 1):
        self.log_every = log_every
        self.start_time: float = 0

    def on_training_start(self) -> None:
        self.start_time = time.time()
        print("Training started...")

    def on_epoch_start(self, epoch: int, total_epochs: int) -> None:
        if epoch % self.log_every == 0:
            print(f"Epoch {epoch}/{total_epochs}")

    def on_epoch_end(self, epoch: int, metrics: dict[str, float]) -> None:
        if epoch % self.log_every == 0:
            metrics_str = " | ".join(
                [
                    f"{k}: {v:.4f}"
                    for k, v in metrics.items()
                    if isinstance(v, (int, float)) and k != "epoch"
                ]
            )
            print(f"  {metrics_str}")

    def on_batch_end(self, batch: int, total_batches: int, loss: float) -> None:
        # Only log significant batch milestones
        if batch % max(1, total_batches // 4) == 0:
            progress = (batch / total_batches) * 100
            print(
                f"  Batch {batch}/{total_batches} ({progress:.1f}%) - Loss: {loss:.4f}"
            )

    def on_training_end(self, final_metrics: dict[str, float]) -> None:
        elapsed = time.time() - self.start_time
        print(f"Training completed in {elapsed:.2f}s")
        print("Final metrics:", final_metrics)
