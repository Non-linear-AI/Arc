"""Training callbacks for Arc ML framework."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from torch.utils.tensorboard import SummaryWriter

logger = logging.getLogger(__name__)


class TensorBoardLogger:
    """TensorBoard logging callback for training progress.

    Logs training and validation metrics to TensorBoard for live visualization.
    Compatible with the ProgressCallback protocol.

    Args:
        log_dir: Directory to save TensorBoard logs
        enabled: Whether TensorBoard logging is enabled (default: True)

    Example:
        >>> logger = TensorBoardLogger(log_dir="~/.arc/training_logs/run_123")
        >>> logger.on_epoch_end(epoch=1, metrics={"loss": 0.5, "accuracy": 0.8})
    """

    def __init__(self, log_dir: str | Path, enabled: bool = True):
        self.log_dir = Path(log_dir).expanduser()
        self.enabled = enabled
        self.writer: SummaryWriter | None = None
        self.global_step = 0
        self.current_epoch = 0

        if self.enabled:
            try:
                from torch.utils.tensorboard import SummaryWriter

                # Create log directory
                self.log_dir.mkdir(parents=True, exist_ok=True)

                # Initialize TensorBoard writer
                self.writer = SummaryWriter(log_dir=str(self.log_dir))
                logger.info(f"TensorBoard logging enabled: {self.log_dir}")
            except ImportError:
                logger.warning(
                    "TensorBoard not available. Install with: pip install tensorboard"
                )
                self.enabled = False
            except Exception as e:
                logger.warning(f"Failed to initialize TensorBoard: {e}")
                self.enabled = False

    def on_training_start(self) -> None:
        """Called when training starts."""
        if not self.enabled or not self.writer:
            return

        logger.debug("TensorBoard: Training started")
        self.global_step = 0
        self.current_epoch = 0

    def on_epoch_start(self, epoch: int, total_epochs: int) -> None:
        """Called at the start of each epoch.

        Args:
            epoch: Current epoch number (0-indexed)
            total_epochs: Total number of epochs
        """
        if not self.enabled or not self.writer:
            return

        self.current_epoch = epoch
        logger.debug(f"TensorBoard: Epoch {epoch + 1}/{total_epochs} started")

    def on_batch_end(self, batch: int, total_batches: int, loss: float) -> None:
        """Called at the end of each batch.

        Args:
            batch: Current batch number
            total_batches: Total number of batches in epoch
            loss: Batch loss value
        """
        if not self.enabled or not self.writer:
            return

        self.global_step += 1

        # Log batch loss
        self.writer.add_scalar("batch/loss", loss, self.global_step)

        # Log progress within epoch
        progress = (batch + 1) / total_batches
        self.writer.add_scalar(
            f"epoch_{self.current_epoch}/progress", progress, self.global_step
        )

    def on_epoch_end(self, epoch: int, metrics: dict[str, float]) -> None:
        """Called at the end of each epoch.

        Args:
            epoch: Current epoch number (0-indexed)
            metrics: Dictionary of metric name -> value
                Expected keys: 'loss', 'val_loss', and any custom metrics
        """
        if not self.enabled or not self.writer:
            return

        epoch_num = epoch + 1  # Convert to 1-indexed for display

        # Log all metrics
        for metric_name, value in metrics.items():
            # Determine metric group (train vs validation)
            if metric_name.startswith("val_"):
                group = "validation"
                clean_name = metric_name[4:]  # Remove "val_" prefix
            else:
                group = "train"
                clean_name = metric_name

            # Log to TensorBoard
            self.writer.add_scalar(f"{group}/{clean_name}", value, epoch_num)

        logger.debug(
            f"TensorBoard: Logged {len(metrics)} metrics for epoch {epoch_num}"
        )

        # Flush to ensure data is written
        self.writer.flush()

    def on_training_end(self, final_metrics: dict[str, float]) -> None:
        """Called when training ends.

        Args:
            final_metrics: Final training metrics
        """
        if not self.enabled or not self.writer:
            return

        # Log final metrics with special tag
        for metric_name, value in final_metrics.items():
            self.writer.add_scalar(f"final/{metric_name}", value, 0)

        logger.info(f"TensorBoard: Training completed. Logs saved to {self.log_dir}")

        # Close writer
        self.writer.close()

    def log_hyperparameters(self, hparams: dict[str, any]) -> None:
        """Log hyperparameters to TensorBoard.

        Args:
            hparams: Dictionary of hyperparameter name -> value
        """
        if not self.enabled or not self.writer:
            return

        # Convert all values to strings for TensorBoard compatibility
        hparams_str = {k: str(v) for k, v in hparams.items()}

        # Log as text
        hparam_text = "\n".join(f"{k}: {v}" for k, v in hparams_str.items())
        self.writer.add_text("hyperparameters", hparam_text, 0)

        logger.debug(f"TensorBoard: Logged {len(hparams)} hyperparameters")

    def log_model_graph(self, model, input_tensor) -> None:
        """Log model architecture graph to TensorBoard.

        Args:
            model: PyTorch model
            input_tensor: Example input tensor for the model
        """
        if not self.enabled or not self.writer:
            return

        try:
            self.writer.add_graph(model, input_tensor)
            logger.debug("TensorBoard: Logged model graph")
        except Exception as e:
            logger.warning(f"Failed to log model graph: {e}")

    def __del__(self):
        """Cleanup: close writer on deletion."""
        if self.writer is not None:
            from contextlib import suppress

            with suppress(Exception):
                self.writer.close()
