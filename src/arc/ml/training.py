"""Simple training loop for Arc models without trainer abstraction.

This module provides a lightweight training function that executes the training loop
directly, following the architectural principle that training configuration lives in
the model YAML specification, not in a separate trainer concept.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from pathlib import Path
from threading import Event
from typing import Any, Protocol

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

logger = logging.getLogger(__name__)


def _sanitize_optimizer_params(params: dict) -> dict:
    """Convert optimizer parameters to proper numeric types.

    YAML 1.1 parses scientific notation like '1e-5' as strings.
    This function converts string parameters to floats where appropriate.

    Args:
        params: Dictionary of optimizer parameters (may contain strings)

    Returns:
        Dictionary with numeric values properly converted
    """
    sanitized = {}
    for key, value in params.items():
        if isinstance(value, str):
            try:
                # Try to convert string to float (handles scientific notation)
                sanitized[key] = float(value)
            except (ValueError, TypeError):
                # Not a numeric string, keep as is
                sanitized[key] = value
        else:
            sanitized[key] = value
    return sanitized


def _get_optimizer(
    optimizer_type: str,
    parameters,
    learning_rate: float,
    **kwargs
) -> torch.optim.Optimizer:
    """Get PyTorch optimizer by name.

    Args:
        optimizer_type: Type of optimizer (adam, sgd, adamw, etc.)
                       Can also be full path like "torch.optim.Adam"
        parameters: Model parameters
        learning_rate: Learning rate
        **kwargs: Additional optimizer parameters

    Returns:
        PyTorch optimizer instance
    """
    # Sanitize parameters to handle YAML string-to-float conversion issues
    kwargs = _sanitize_optimizer_params(kwargs)

    # Normalize optimizer type: strip "torch.optim." prefix if present and lowercase
    optimizer_type = optimizer_type.lower()
    if optimizer_type.startswith("torch.optim."):
        optimizer_type = optimizer_type.replace("torch.optim.", "")

    if optimizer_type == "adam":
        return torch.optim.Adam(parameters, lr=learning_rate, **kwargs)
    elif optimizer_type == "sgd":
        return torch.optim.SGD(parameters, lr=learning_rate, **kwargs)
    elif optimizer_type == "adamw":
        return torch.optim.AdamW(parameters, lr=learning_rate, **kwargs)
    elif optimizer_type == "rmsprop":
        return torch.optim.RMSprop(parameters, lr=learning_rate, **kwargs)
    else:
        raise ValueError(f"Unsupported optimizer type: {optimizer_type}")


def _get_loss_function(loss_type: str, **kwargs) -> nn.Module:
    """Get PyTorch loss function by name.

    Args:
        loss_type: Type of loss function
        **kwargs: Additional loss parameters

    Returns:
        PyTorch loss function
    """
    loss_type = loss_type.lower()

    if loss_type in ("mse", "mean_squared_error"):
        return nn.MSELoss(**kwargs)
    elif loss_type in ("mae", "l1", "mean_absolute_error"):
        return nn.L1Loss(**kwargs)
    elif loss_type in ("bce", "binary_cross_entropy"):
        return nn.BCELoss(**kwargs)
    elif loss_type in ("bce_with_logits", "binary_cross_entropy_with_logits"):
        return nn.BCEWithLogitsLoss(**kwargs)
    elif loss_type in ("ce", "cross_entropy", "categorical_cross_entropy"):
        return nn.CrossEntropyLoss(**kwargs)
    elif loss_type in ("nll", "negative_log_likelihood"):
        return nn.NLLLoss(**kwargs)
    else:
        raise ValueError(f"Unsupported loss function: {loss_type}")


class ProgressCallback(Protocol):
    """Protocol for training progress callbacks."""

    def on_training_start(self) -> None:
        """Called when training starts."""
        ...

    def on_epoch_start(self, epoch: int, total_epochs: int) -> None:
        """Called at the start of each epoch."""
        ...

    def on_batch_end(self, batch: int, total_batches: int, loss: float) -> None:
        """Called at the end of each batch."""
        ...

    def on_epoch_end(self, epoch: int, metrics: dict[str, float]) -> None:
        """Called at the end of each epoch."""
        ...

    def on_training_end(self, final_metrics: dict[str, float]) -> None:
        """Called when training ends."""
        ...


@dataclass
class TrainingResult:
    """Results from a training run."""

    success: bool
    train_losses: list[float]
    val_losses: list[float] | None = None
    final_train_loss: float | None = None
    final_val_loss: float | None = None
    best_val_loss: float | None = None
    metrics_history: dict[str, list[float]] | None = None
    training_time: float | None = None
    total_epochs: int | None = None
    best_epoch: int | None = None
    error_message: str | None = None


def train_model(
    model: nn.Module,
    train_loader: DataLoader,
    training_config: Any,
    val_loader: DataLoader | None = None,
    callback: ProgressCallback | None = None,
    checkpoint_dir: Path | None = None,
    stop_event: Event | None = None,
) -> tuple[TrainingResult, torch.optim.Optimizer]:
    """Execute training loop for a model.

    Args:
        model: PyTorch model to train
        train_loader: Training data loader
        training_config: Training configuration (SimpleNamespace with training params)
        val_loader: Optional validation data loader
        callback: Optional progress callback
        checkpoint_dir: Optional directory for saving checkpoints
        stop_event: Optional event to signal training cancellation

    Returns:
        Tuple of (training result, optimizer instance)
    """
    start_time = time.time()

    # Extract configuration
    device = getattr(training_config, "device", "cpu")
    epochs = getattr(training_config, "epochs", 10)
    learning_rate = getattr(training_config, "learning_rate", 0.001)

    # Move model to device
    model = model.to(device)

    # Create optimizer
    optimizer_type = getattr(training_config, "optimizer", "adam")
    optimizer_params = getattr(training_config, "optimizer_params", {})
    optimizer = _get_optimizer(
        optimizer_type,
        model.parameters(),
        learning_rate=learning_rate,
        **optimizer_params
    )

    # Create loss function
    loss_fn_name = getattr(training_config, "loss_function", "mse")
    loss_params = getattr(training_config, "loss_params", {})
    loss_fn = _get_loss_function(loss_fn_name, **loss_params)

    # Training state
    train_losses = []
    val_losses = []
    best_val_loss = float("inf")
    best_epoch = 0

    # Early stopping
    early_stopping_patience = getattr(training_config, "early_stopping_patience", None)
    epochs_without_improvement = 0

    # Gradient clipping
    gradient_clip_val = getattr(training_config, "gradient_clip_val", None)

    # Target processing
    reshape_targets = getattr(training_config, "reshape_targets", False)
    target_output_key = getattr(training_config, "target_output_key", None)

    try:
        # Notify training start
        if callback:
            callback.on_training_start()

        # Training loop
        for epoch in range(1, epochs + 1):
            # Check for cancellation
            if stop_event and stop_event.is_set():
                logger.info(f"Training cancelled at epoch {epoch}")
                return TrainingResult(
                    success=False,
                    train_losses=train_losses,
                    val_losses=val_losses if val_losses else None,
                    final_train_loss=train_losses[-1] if train_losses else None,
                    final_val_loss=val_losses[-1] if val_losses else None,
                    best_val_loss=best_val_loss if val_losses else None,
                    training_time=time.time() - start_time,
                    total_epochs=epoch - 1,
                    best_epoch=best_epoch,
                    error_message="Training cancelled by user",
                ), optimizer

            # Notify epoch start
            if callback:
                callback.on_epoch_start(epoch, epochs)

            # Train one epoch
            model.train()
            epoch_train_loss = 0.0
            num_batches = len(train_loader)

            for batch_idx, batch in enumerate(train_loader, 1):
                # Check for cancellation during epoch
                if stop_event and stop_event.is_set():
                    logger.info(f"Training cancelled during epoch {epoch}")
                    return TrainingResult(
                        success=False,
                        train_losses=train_losses,
                        val_losses=val_losses if val_losses else None,
                        final_train_loss=train_losses[-1] if train_losses else None,
                        final_val_loss=val_losses[-1] if val_losses else None,
                        best_val_loss=best_val_loss if val_losses else None,
                        training_time=time.time() - start_time,
                        total_epochs=epoch - 1,
                        best_epoch=best_epoch,
                        error_message="Training cancelled by user",
                    ), optimizer

                # Extract features and targets
                if isinstance(batch, (tuple, list)):
                    features, targets = batch[0], batch[1]
                else:
                    features, targets = batch["features"], batch["targets"]

                # Move to device
                if isinstance(features, dict):
                    features = {k: v.to(device) for k, v in features.items()}
                else:
                    features = features.to(device)
                targets = targets.to(device)

                # Reshape targets if needed
                if reshape_targets and targets.dim() == 1:
                    targets = targets.unsqueeze(1)

                # Forward pass
                optimizer.zero_grad()
                outputs = model(features)

                # Extract specific output if needed
                if target_output_key and isinstance(outputs, dict):
                    outputs = outputs[target_output_key]

                # Compute loss
                loss = loss_fn(outputs, targets)

                # Backward pass
                loss.backward()

                # Gradient clipping if configured
                if gradient_clip_val is not None:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clip_val)

                optimizer.step()

                # Track loss
                batch_loss = loss.item()
                epoch_train_loss += batch_loss

                # Notify batch end
                if callback:
                    callback.on_batch_end(batch_idx, num_batches, batch_loss)

            # Average training loss for epoch
            avg_train_loss = epoch_train_loss / num_batches
            train_losses.append(avg_train_loss)

            # Validation
            avg_val_loss = None
            if val_loader:
                model.eval()
                epoch_val_loss = 0.0
                num_val_batches = len(val_loader)

                with torch.no_grad():
                    for batch in val_loader:
                        # Extract features and targets
                        if isinstance(batch, (tuple, list)):
                            features, targets = batch[0], batch[1]
                        else:
                            features, targets = batch["features"], batch["targets"]

                        # Move to device
                        if isinstance(features, dict):
                            features = {k: v.to(device) for k, v in features.items()}
                        else:
                            features = features.to(device)
                        targets = targets.to(device)

                        # Reshape targets if needed
                        if reshape_targets and targets.dim() == 1:
                            targets = targets.unsqueeze(1)

                        # Forward pass
                        outputs = model(features)

                        # Extract specific output if needed
                        if target_output_key and isinstance(outputs, dict):
                            outputs = outputs[target_output_key]

                        # Compute loss
                        loss = loss_fn(outputs, targets)
                        epoch_val_loss += loss.item()

                avg_val_loss = epoch_val_loss / num_val_batches
                val_losses.append(avg_val_loss)

                # Track best model
                if avg_val_loss < best_val_loss:
                    best_val_loss = avg_val_loss
                    best_epoch = epoch
                    epochs_without_improvement = 0

                    # Save checkpoint if configured
                    if checkpoint_dir:
                        checkpoint_dir.mkdir(parents=True, exist_ok=True)
                        checkpoint_path = checkpoint_dir / "best_model.pt"
                        torch.save({
                            "epoch": epoch,
                            "model_state_dict": model.state_dict(),
                            "optimizer_state_dict": optimizer.state_dict(),
                            "train_loss": avg_train_loss,
                            "val_loss": avg_val_loss,
                        }, checkpoint_path)
                else:
                    epochs_without_improvement += 1

            # Epoch metrics
            metrics = {
                "epoch": epoch,
                "train_loss": avg_train_loss,
            }
            if avg_val_loss is not None:
                metrics["val_loss"] = avg_val_loss

            # Notify epoch end
            if callback:
                callback.on_epoch_end(epoch, metrics)

            # Early stopping check
            if early_stopping_patience and epochs_without_improvement >= early_stopping_patience:
                logger.info(
                    f"Early stopping triggered after {epoch} epochs "
                    f"({epochs_without_improvement} epochs without improvement)"
                )
                break

        # Training completed successfully
        training_time = time.time() - start_time

        final_metrics = {
            "final_train_loss": train_losses[-1] if train_losses else None,
            "final_val_loss": val_losses[-1] if val_losses else None,
            "best_val_loss": best_val_loss if val_losses else None,
        }

        if callback:
            callback.on_training_end(final_metrics)

        result = TrainingResult(
            success=True,
            train_losses=train_losses,
            val_losses=val_losses if val_losses else None,
            final_train_loss=train_losses[-1] if train_losses else None,
            final_val_loss=val_losses[-1] if val_losses else None,
            best_val_loss=best_val_loss if val_losses else None,
            training_time=training_time,
            total_epochs=len(train_losses),
            best_epoch=best_epoch if val_losses else None,
        )

        return result, optimizer

    except Exception as e:
        logger.error(f"Training failed with error: {e}", exc_info=True)

        if callback:
            callback.on_training_end({})

        result = TrainingResult(
            success=False,
            train_losses=train_losses,
            val_losses=val_losses if val_losses else None,
            final_train_loss=train_losses[-1] if train_losses else None,
            final_val_loss=val_losses[-1] if val_losses else None,
            best_val_loss=best_val_loss if val_losses else None,
            training_time=time.time() - start_time,
            total_epochs=len(train_losses),
            best_epoch=best_epoch if val_losses else None,
            error_message=str(e),
        )

        return result, optimizer
