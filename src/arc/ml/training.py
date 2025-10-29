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


def _normalize_device(device: str) -> str:
    """Normalize device string to valid PyTorch device.

    Args:
        device: Device string (can be "auto", "cpu", "cuda", "mps", etc.)

    Returns:
        Valid PyTorch device string
    """
    device = device.lower().strip()

    # If already a valid device, return as-is
    if device in ("cpu", "cuda", "mps", "ipu", "xpu"):
        return device

    # Handle "auto" - pick best available device
    if device == "auto":
        if torch.cuda.is_available():
            return "cuda"
        elif torch.backends.mps.is_available():
            return "mps"
        else:
            return "cpu"

    # Default to CPU for unknown devices
    logger.warning(f"Unknown device '{device}', defaulting to 'cpu'")
    return "cpu"


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
        loss_type: Type of loss function (can be simple name or full path like
                   "torch.nn.functional.binary_cross_entropy_with_logits")
        **kwargs: Additional loss parameters

    Returns:
        PyTorch loss function
    """
    # Normalize loss type: strip PyTorch path prefixes if present
    loss_type = loss_type.lower()
    # Remove various PyTorch prefixes
    for prefix in ["torch.nn.functional.", "torch.nn.", "torch.functional.", "torch."]:
        if loss_type.startswith(prefix):
            loss_type = loss_type.replace(prefix, "", 1)
            break

    # Handle both functional names and class names
    # Map class names to their functional equivalents
    class_to_functional = {
        "mseloss": "mse",
        "l1loss": "mae",
        "bceloss": "bce",
        "bcewithlogitsloss": "bce_with_logits",
        "crossentropyloss": "cross_entropy",
        "nllloss": "nll",
    }
    loss_type = class_to_functional.get(loss_type, loss_type)

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
    tensorboard_log_dir: Path | None = None,
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
        tensorboard_log_dir: Optional directory for TensorBoard logging

    Returns:
        Tuple of (training result, optimizer instance)
    """
    start_time = time.time()

    # Extract configuration
    device_str = getattr(training_config, "device", "cpu")
    device = _normalize_device(device_str)
    epochs = getattr(training_config, "epochs", 10)
    learning_rate = getattr(training_config, "learning_rate", 0.001)

    # Initialize TensorBoard writer if logging is enabled
    tensorboard_writer = None
    if tensorboard_log_dir:
        try:
            from torch.utils.tensorboard import SummaryWriter
            tensorboard_log_dir.mkdir(parents=True, exist_ok=True)
            tensorboard_writer = SummaryWriter(log_dir=str(tensorboard_log_dir))
            logger.info(f"TensorBoard logging enabled: {tensorboard_log_dir}")
        except ImportError:
            logger.warning("TensorBoard not available. Install with: pip install tensorboard")
        except Exception as e:
            logger.warning(f"Failed to initialize TensorBoard: {e}")

    # Initialize metrics tracker if validation data is provided
    metrics_tracker = None
    if val_loader:
        try:
            from arc.ml.metrics import create_metrics_for_task

            # Infer task type from training config or loss function
            loss_fn_name = getattr(training_config, "loss_function", "mse").lower()
            if "mse" in loss_fn_name or "mae" in loss_fn_name or "l1" in loss_fn_name:
                task_type = "regression"
            elif "cross_entropy" in loss_fn_name or "nll" in loss_fn_name:
                task_type = "classification"
            elif "bce" in loss_fn_name:
                task_type = "binary_classification"
            else:
                # Default to classification for unknown loss functions
                task_type = "classification"

            metrics_tracker = create_metrics_for_task(task_type)
            logger.info(f"Initialized metrics tracker for {task_type} task")
        except Exception as e:
            logger.warning(f"Failed to initialize metrics tracker: {e}")

    # Move model to device
    model = model.to(device)

    # Log model graph to TensorBoard
    if tensorboard_writer:
        try:
            # Get a sample batch to trace the model
            sample_batch = next(iter(train_loader))
            if isinstance(sample_batch, (tuple, list)):
                sample_features = sample_batch[0]
            else:
                sample_features = sample_batch["features"]

            # Move to device
            if isinstance(sample_features, dict):
                sample_features = {k: v.to(device) for k, v in sample_features.items()}
            else:
                sample_features = sample_features.to(device)

            # Log the model graph
            tensorboard_writer.add_graph(model, sample_features)
            logger.debug("Logged model architecture graph to TensorBoard")

        except Exception as e:
            logger.warning(f"Failed to log model graph: {e}")

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
    global_step = 0  # Track global step for TensorBoard logging

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

                # Log gradient statistics before clipping (every 50 steps to reduce overhead)
                if tensorboard_writer and global_step % 50 == 0:
                    try:
                        grad_norms = []
                        grad_max = []
                        grad_min = []
                        for name, param in model.named_parameters():
                            if param.grad is not None:
                                grad_norm = param.grad.norm().item()
                                grad_norms.append(grad_norm)
                                grad_max.append(param.grad.max().item())
                                grad_min.append(param.grad.min().item())

                                # Log per-layer gradient norms (for detailed debugging)
                                tensorboard_writer.add_scalar(
                                    f"gradients/{name}/norm", grad_norm, global_step
                                )

                        if grad_norms:
                            # Log aggregate gradient statistics
                            import numpy as np
                            tensorboard_writer.add_scalar(
                                "gradients/global_norm", np.sqrt(sum(g**2 for g in grad_norms)), global_step
                            )
                            tensorboard_writer.add_scalar(
                                "gradients/mean_norm", np.mean(grad_norms), global_step
                            )
                            tensorboard_writer.add_scalar(
                                "gradients/max_norm", max(grad_norms), global_step
                            )
                            tensorboard_writer.add_scalar(
                                "gradients/max_value", max(grad_max), global_step
                            )
                            tensorboard_writer.add_scalar(
                                "gradients/min_value", min(grad_min), global_step
                            )

                    except Exception as e:
                        logger.debug(f"Failed to log gradient statistics: {e}")

                # Gradient clipping if configured
                if gradient_clip_val is not None:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clip_val)

                optimizer.step()

                # Track loss
                batch_loss = loss.item()
                epoch_train_loss += batch_loss
                global_step += 1

                # Log to TensorBoard
                if tensorboard_writer and global_step % 10 == 0:  # Log every 10 steps
                    tensorboard_writer.add_scalar("train/batch_loss", batch_loss, global_step)
                    # Log learning rate
                    current_lr = optimizer.param_groups[0]["lr"]
                    tensorboard_writer.add_scalar("train/learning_rate", current_lr, global_step)

                # Notify batch end
                if callback:
                    callback.on_batch_end(batch_idx, num_batches, batch_loss)

            # Average training loss for epoch
            avg_train_loss = epoch_train_loss / num_batches
            train_losses.append(avg_train_loss)

            # Validation
            avg_val_loss = None
            val_metrics = {}
            if val_loader:
                model.eval()
                epoch_val_loss = 0.0
                num_val_batches = len(val_loader)

                # Collect all predictions and targets for metrics computation
                all_predictions = []
                all_targets = []

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

                        # Collect predictions and targets for metrics
                        if metrics_tracker:
                            all_predictions.append(outputs.cpu())
                            all_targets.append(targets.cpu())

                avg_val_loss = epoch_val_loss / num_val_batches
                val_losses.append(avg_val_loss)

                # Compute validation metrics
                if metrics_tracker and all_predictions:
                    try:
                        all_predictions_tensor = torch.cat(all_predictions, dim=0)
                        all_targets_tensor = torch.cat(all_targets, dim=0)

                        # Compute metrics
                        metrics_results = metrics_tracker.compute_metrics(
                            all_predictions_tensor, all_targets_tensor
                        )

                        # Extract metric values - ensure they're Python floats
                        for metric_name, metric_result in metrics_results.items():
                            value = metric_result.value
                            # Convert torch.Tensor to Python float if needed
                            if isinstance(value, torch.Tensor):
                                value = value.item()
                            val_metrics[metric_name] = value

                        # Log visualizations to TensorBoard for classification tasks
                        if tensorboard_writer and "accuracy" in val_metrics:
                            try:
                                from arc.ml.visualization import TensorBoardVisualizer

                                viz = TensorBoardVisualizer(tensorboard_writer)

                                # Get probabilities for binary classification
                                if all_predictions_tensor.shape[-1] == 2:
                                    # Softmax output
                                    probs = torch.softmax(all_predictions_tensor, dim=1)[:, 1]
                                elif all_predictions_tensor.shape[-1] == 1:
                                    # Sigmoid output
                                    probs = torch.sigmoid(all_predictions_tensor.squeeze())
                                else:
                                    probs = None

                                # Log PR curve for binary classification
                                if probs is not None and len(torch.unique(all_targets_tensor)) == 2:
                                    viz.log_pr_curve(
                                        all_targets_tensor,
                                        probs,
                                        tag="validation/pr_curve",
                                        step=epoch,
                                    )
                                    viz.log_roc_curve(
                                        all_targets_tensor,
                                        probs,
                                        tag="validation/roc",
                                        step=epoch,
                                    )

                                # Log confusion matrix
                                if all_predictions_tensor.shape[-1] > 1:
                                    predictions_class = torch.argmax(
                                        all_predictions_tensor, dim=1
                                    )
                                else:
                                    predictions_class = (probs > 0.5).long()

                                viz.log_confusion_matrix(
                                    all_targets_tensor,
                                    predictions_class,
                                    tag="validation/confusion_matrix",
                                    step=epoch,
                                )

                                # Log confusion matrix as heatmap image
                                viz.log_confusion_matrix_heatmap(
                                    all_targets_tensor,
                                    predictions_class,
                                    tag="validation/confusion_matrix_heatmap",
                                    step=epoch,
                                    normalize=True,  # Show percentages and counts
                                )

                                # Log per-class performance
                                viz.log_class_performance(
                                    all_targets_tensor,
                                    predictions_class,
                                    tag="validation/class_performance",
                                    step=epoch,
                                )

                            except Exception as e:
                                logger.warning(f"Failed to log visualizations: {e}")

                    except Exception as e:
                        logger.warning(f"Failed to compute validation metrics: {e}")

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

            # Add validation metrics
            for metric_name, metric_value in val_metrics.items():
                metrics[f"val_{metric_name}"] = metric_value

            # Log epoch metrics to TensorBoard
            if tensorboard_writer:
                tensorboard_writer.add_scalar("epoch/train_loss", avg_train_loss, epoch)
                if avg_val_loss is not None:
                    tensorboard_writer.add_scalar("epoch/val_loss", avg_val_loss, epoch)

                # Log validation metrics
                for metric_name, metric_value in val_metrics.items():
                    tensorboard_writer.add_scalar(
                        f"epoch/val_{metric_name}", metric_value, epoch
                    )

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

        # Log hyperparameters to TensorBoard
        if tensorboard_writer:
            try:
                # Collect hyperparameters
                hparams = {
                    "learning_rate": learning_rate,
                    "epochs": epochs,
                    "batch_size": train_loader.batch_size if hasattr(train_loader, 'batch_size') else 32,
                    "optimizer": optimizer_type,
                    "loss_function": loss_fn_name,
                    "device": device,
                }

                # Add optional hyperparameters if they exist
                if early_stopping_patience is not None:
                    hparams["early_stopping_patience"] = early_stopping_patience
                if gradient_clip_val is not None:
                    hparams["gradient_clip_val"] = gradient_clip_val
                if val_loader:
                    hparams["validation_split"] = getattr(training_config, "validation_split", 0.2)

                # Add optimizer-specific parameters
                for key, value in optimizer_params.items():
                    if isinstance(value, (int, float, bool, str)):
                        hparams[f"optimizer_{key}"] = value

                # Prepare metrics for hparams (only scalars)
                hparam_metrics = {}
                if train_losses:
                    hparam_metrics["hparam/final_train_loss"] = train_losses[-1]
                if val_losses:
                    hparam_metrics["hparam/final_val_loss"] = val_losses[-1]
                    hparam_metrics["hparam/best_val_loss"] = best_val_loss
                    hparam_metrics["hparam/best_epoch"] = best_epoch
                hparam_metrics["hparam/total_epochs"] = len(train_losses)
                hparam_metrics["hparam/training_time"] = training_time

                # Log to TensorBoard's HPARAMS tab
                tensorboard_writer.add_hparams(hparam_dict=hparams, metric_dict=hparam_metrics)
                logger.debug("Logged hyperparameters to TensorBoard")

            except Exception as e:
                logger.warning(f"Failed to log hyperparameters: {e}")

        # Close TensorBoard writer
        if tensorboard_writer:
            try:
                tensorboard_writer.close()
                logger.info("TensorBoard writer closed")
            except Exception as e:
                logger.warning(f"Failed to close TensorBoard writer: {e}")

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

        # Close TensorBoard writer
        if tensorboard_writer:
            try:
                tensorboard_writer.close()
            except Exception:
                pass  # Ignore errors during cleanup

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
