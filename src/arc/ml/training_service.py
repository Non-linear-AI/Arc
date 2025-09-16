"""Training service for Arc Graph model training integration."""

from __future__ import annotations

import asyncio
import logging
import uuid
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from threading import Event
from typing import Any

from ..database.services.job_service import JobService
from ..database.services.ml_data_service import MLDataService
from ..graph.spec import ArcGraph, TrainingConfig
from ..jobs.models import Job, JobStatus, JobType
from .artifacts import (
    ModelArtifactManager,
    create_artifact_from_training,
)
from .builder import ArcModel, ModelBuilder
from .data import DataProcessor
from .trainer import ArcTrainer, ProgressCallback, TrainingResult

logger = logging.getLogger(__name__)


@dataclass
class TrainingJobConfig:
    """Configuration for a training job."""

    # Model configuration
    model_id: str
    model_name: str
    arc_graph: ArcGraph

    # Data configuration
    train_table: str
    target_column: str
    feature_columns: list[str] | None = None
    validation_table: str | None = None
    validation_split: float = 0.2

    # Training configuration
    training_config: TrainingConfig | None = None

    # Storage configuration
    artifacts_dir: str | None = None
    checkpoint_dir: str | None = None

    # Job metadata
    description: str | None = None
    tags: list[str] | None = None


class TrainingJobProgressCallback:
    """Progress callback that updates job status in database."""

    def __init__(self, job_service: JobService, job_id: str):
        self.job_service = job_service
        self.job_id = job_id
        self.total_epochs = 0
        self.current_epoch = 0

    def on_training_start(self) -> None:
        """Called when training starts."""
        self.job_service.update_job_status(
            self.job_id, JobStatus.RUNNING, "Training started"
        )
        logger.info(f"Training job {self.job_id} started")

    def on_epoch_start(self, epoch: int, total_epochs: int) -> None:
        """Called at the start of each epoch."""
        self.current_epoch = epoch
        self.total_epochs = total_epochs

        progress_pct = int((epoch / total_epochs) * 100)
        message = f"Epoch {epoch}/{total_epochs} ({progress_pct}%)"

        self.job_service.update_job_status(self.job_id, JobStatus.RUNNING, message)

    def on_epoch_end(self, epoch: int, metrics: dict[str, float]) -> None:
        """Called at the end of each epoch."""
        metrics_str = " | ".join(
            [
                f"{k}: {v:.4f}"
                for k, v in metrics.items()
                if isinstance(v, (int, float)) and k != "epoch"
            ]
        )

        progress_pct = int((epoch / self.total_epochs) * 100)
        message = f"Epoch {epoch}/{self.total_epochs} ({progress_pct}%) - {metrics_str}"

        self.job_service.update_job_status(self.job_id, JobStatus.RUNNING, message)

    def on_batch_end(self, batch: int, total_batches: int, loss: float) -> None:
        """Called at the end of each batch."""
        # Only update for significant progress milestones to avoid spam
        if batch % max(1, total_batches // 10) == 0:
            batch_progress = (batch / total_batches) * 100
            epoch_progress = (self.current_epoch / self.total_epochs) * 100
            message = (
                f"Epoch {self.current_epoch}/{self.total_epochs} "
                f"({epoch_progress:.1f}%) - Batch {batch}/{total_batches} "
                f"({batch_progress:.1f}%) - Loss: {loss:.4f}"
            )

            self.job_service.update_job_status(self.job_id, JobStatus.RUNNING, message)

    def on_training_end(self, final_metrics: dict[str, float]) -> None:
        """Called when training ends."""
        metrics_str = " | ".join(
            [
                f"{k}: {v:.4f}"
                for k, v in final_metrics.items()
                if isinstance(v, (int, float))
            ]
        )

        message = f"Training completed - {metrics_str}"
        self.job_service.update_job_status(self.job_id, JobStatus.COMPLETED, message)
        logger.info(f"Training job {self.job_id} completed")


class TrainingService:
    """Service for managing model training jobs."""

    def __init__(
        self,
        job_service: JobService,
        artifacts_dir: str | Path | None = None,
        max_concurrent_jobs: int = 2,
    ):
        """Initialize training service.

        Args:
            job_service: Job service for database operations
            artifacts_dir: Directory for storing model artifacts
            max_concurrent_jobs: Maximum concurrent training jobs
        """
        self.job_service = job_service
        self.artifacts_dir = Path(artifacts_dir) if artifacts_dir else Path("artifacts")
        self.artifact_manager = ModelArtifactManager(self.artifacts_dir)

        # Async execution
        self.executor = ThreadPoolExecutor(max_workers=max_concurrent_jobs)
        self.active_jobs: dict[str, asyncio.Task] = {}
        self._cancel_events: dict[str, Event] = {}

        logger.info(
            f"TrainingService initialized with artifacts_dir: {self.artifacts_dir}"
        )

    def submit_training_job(self, config: TrainingJobConfig) -> str:
        """Submit a new training job.

        Args:
            config: Training job configuration

        Returns:
            Job ID for tracking the training job
        """
        # Generate job ID
        job_id = str(uuid.uuid4())

        # Create job record
        job = Job(
            job_id=job_id,
            model_id=None,  # Will be set after model is created
            type=JobType.TRAIN_MODEL.value,
            status=JobStatus.PENDING,
            message="Training job submitted",
            sql_query=None,
            created_at=datetime.now(UTC),
            updated_at=datetime.now(UTC),
        )

        # Save job to database
        self.job_service.create_job(job)

        # Start async training
        cancel_event = Event()
        self._cancel_events[job_id] = cancel_event

        task = asyncio.create_task(
            self._execute_training_job(job_id, config, cancel_event)
        )
        self.active_jobs[job_id] = task

        logger.info(f"Training job {job_id} submitted for model {config.model_id}")
        return job_id

    async def _execute_training_job(
        self, job_id: str, config: TrainingJobConfig, cancel_event: Event
    ) -> TrainingResult:
        """Execute training job asynchronously.

        Args:
            job_id: Job identifier
            config: Training configuration
            cancel_event: Cancellation event shared with executor task

        Returns:
            Training result
        """
        try:
            # Setup progress callback
            progress_callback = TrainingJobProgressCallback(self.job_service, job_id)

            # Run training in executor to avoid blocking event loop
            loop = asyncio.get_event_loop()
            result = await loop.run_in_executor(
                self.executor,
                self._run_training,
                job_id,
                config,
                progress_callback,
                cancel_event,
            )

            return result

        except Exception as e:
            # Update job status on failure
            error_message = f"Training failed: {str(e)}"
            self.job_service.update_job_status(job_id, JobStatus.FAILED, error_message)
            logger.error(f"Training job {job_id} failed: {e}")
            raise

        finally:
            # Clean up
            if job_id in self.active_jobs:
                del self.active_jobs[job_id]
            if job_id in self._cancel_events:
                del self._cancel_events[job_id]

    def _run_training(
        self,
        job_id: str,
        config: TrainingJobConfig,
        progress_callback: ProgressCallback,
        cancel_event: Event,
    ) -> TrainingResult:
        """Run training synchronously.

        Args:
            job_id: Job identifier
            config: Training configuration
            progress_callback: Progress callback
            cancel_event: Cancellation event

        Returns:
            Training result
        """
        # Setup training configuration - extract from Arc-Graph or use provided config
        if config.training_config:
            training_config = config.training_config
        else:
            # Extract training configuration from Arc-Graph specification without
            # overriding graph-defined parameters unless explicitly provided
            training_config = config.arc_graph.to_training_config()

        # Persist the effective training configuration for downstream use
        config.training_config = training_config

        # Build model from Arc Graph
        builder = ModelBuilder()
        model = builder.build_model(config.arc_graph)

        # Create data processor with ML data service
        ml_data_service = MLDataService(self.job_service.db_manager)
        data_processor = DataProcessor(ml_data_service=ml_data_service)

        # Create data loaders
        train_loader = data_processor.create_dataloader_from_dataset(
            dataset_name=config.train_table,
            feature_columns=config.feature_columns,
            target_columns=[config.target_column],
            batch_size=training_config.batch_size,
            shuffle=training_config.shuffle,
        )

        val_loader = None
        if config.validation_table:
            val_loader = data_processor.create_dataloader_from_dataset(
                dataset_name=config.validation_table,
                feature_columns=config.feature_columns,
                target_columns=[config.target_column],
                batch_size=training_config.batch_size,
                shuffle=False,
            )

        # Setup checkpoint directory
        checkpoint_dir = None
        if config.checkpoint_dir:
            checkpoint_dir = Path(config.checkpoint_dir)
        else:
            checkpoint_dir = self.artifacts_dir / config.model_id / "checkpoints"

        # Create trainer and run training
        trainer = ArcTrainer(training_config)
        result = trainer.train(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            callback=progress_callback,
            checkpoint_dir=checkpoint_dir,
            stop_event=cancel_event,
        )

        if cancel_event.is_set() and not result.success:
            # Ensure job status reflects cancellation even if trainer returned
            self.job_service.update_job_status(
                job_id, JobStatus.CANCELLED, "Training cancelled"
            )
            return result

        # Save trained model artifact
        if result.success:
            self._save_training_artifact(job_id, config, model, trainer, result)

        return result

    def _save_training_artifact(
        self,
        job_id: str,
        config: TrainingJobConfig,
        model: ArcModel,
        trainer: ArcTrainer,
        result: TrainingResult,
    ) -> None:
        """Save training artifacts.

        Args:
            job_id: Job identifier
            config: Training configuration
            model: Trained model
            trainer: Trainer instance
            result: Training result
        """
        # Determine version (increment from existing)
        try:
            latest_version = self.artifact_manager.get_latest_version(config.model_id)
            version = latest_version + 1
        except FileNotFoundError:
            version = 1

        # Create artifact metadata
        artifact = create_artifact_from_training(
            model_id=config.model_id,
            model_name=config.model_name,
            version=version,
            training_config=config.training_config or TrainingConfig(),
            training_result=result,
            arc_graph=config.arc_graph,
            model_info={
                "model_class": model.__class__.__name__,
                "input_names": model.input_names,
                "output_mapping": model.output_mapping,
                "execution_order": model.execution_order,
            },
            description=config.description,
            tags=config.tags,
        )

        # Prepare training history
        training_history = {
            "job_id": job_id,
            "train_losses": result.train_losses,
            "val_losses": result.val_losses,
            "metrics_history": result.metrics_history,
            "training_time": result.training_time,
            "config": asdict(config.training_config or TrainingConfig()),
        }

        # Save artifact
        artifact_dir = self.artifact_manager.save_model_artifact(
            model=model,
            artifact=artifact,
            optimizer=trainer.optimizer,
            training_history=training_history,
            arc_graph=config.arc_graph,
            overwrite=False,
        )

        logger.info(f"Training artifacts saved to: {artifact_dir}")

    def get_job_status(self, job_id: str) -> dict[str, Any]:
        """Get status of a training job.

        Args:
            job_id: Job identifier

        Returns:
            Job status information
        """
        # Get job from database
        job_info = self.job_service.get_job_by_id(job_id)

        # Add runtime information
        status = {
            "job_id": job_id,
            "status": job_info.status.value if job_info else "NOT_FOUND",
            "message": job_info.message if job_info else "Job not found",
            "created_at": job_info.created_at.isoformat() if job_info else None,
            "updated_at": job_info.updated_at.isoformat() if job_info else None,
            "is_active": job_id in self.active_jobs,
        }

        return status

    def cancel_job(self, job_id: str) -> bool:
        """Cancel a running training job.

        Args:
            job_id: Job identifier

        Returns:
            True if job was cancelled, False if not found or already completed
        """
        if job_id in self.active_jobs:
            cancel_event = self._cancel_events.get(job_id)
            if cancel_event:
                cancel_event.set()

            # Update job status immediately to reflect cancellation request
            self.job_service.update_job_status(
                job_id, JobStatus.CANCELLED, "Job cancellation requested"
            )

            logger.info(f"Training job {job_id} cancellation requested")
            return True

        return False

    def list_active_jobs(self) -> list[str]:
        """List active training job IDs.

        Returns:
            List of active job IDs
        """
        return list(self.active_jobs.keys())

    def cleanup_completed_jobs(self) -> None:
        """Clean up completed/failed job tasks."""
        completed_jobs = []

        for job_id, task in self.active_jobs.items():
            if task.done():
                completed_jobs.append(job_id)

        for job_id in completed_jobs:
            del self.active_jobs[job_id]
            if job_id in self._cancel_events:
                del self._cancel_events[job_id]

        if completed_jobs:
            logger.info(f"Cleaned up {len(completed_jobs)} completed jobs")

    async def wait_for_job(
        self, job_id: str, timeout: float | None = None
    ) -> TrainingResult | None:
        """Wait for a training job to complete.

        Args:
            job_id: Job identifier
            timeout: Optional timeout in seconds

        Returns:
            Training result if completed, None if timeout or not found
        """
        if job_id not in self.active_jobs:
            return None

        task = self.active_jobs[job_id]

        try:
            if timeout:
                return await asyncio.wait_for(task, timeout=timeout)
            else:
                return await task
        except TimeoutError:
            return None

    def shutdown(self) -> None:
        """Shutdown the training service."""
        # Cancel all active jobs
        for job_id, task in self.active_jobs.items():
            if not task.done():
                task.cancel()
                self.job_service.update_job_status(
                    job_id, JobStatus.CANCELLED, "Service shutdown"
                )

        # Shutdown executor
        self.executor.shutdown(wait=True)
        logger.info("Training service shutdown completed")
