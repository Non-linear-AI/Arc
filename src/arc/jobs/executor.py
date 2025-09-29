"""Async job execution engine."""

import asyncio
import logging
import threading
from collections.abc import Awaitable, Callable
from concurrent.futures import Future, ThreadPoolExecutor

from arc.jobs.manager import JobManager
from arc.jobs.models import Job, JobStatus, JobType

logger = logging.getLogger(__name__)

# Type for job execution functions
JobExecutorFunc = Callable[[Job], None]
AsyncJobExecutorFunc = Callable[[Job], Awaitable[None]]


class JobExecutor:
    """Executes jobs asynchronously with progress tracking."""

    def __init__(self, job_manager: JobManager, max_workers: int = 4):
        """Initialize JobExecutor.

        Args:
            job_manager: JobManager instance for job persistence
            max_workers: Maximum number of concurrent workers
        """
        self.job_manager = job_manager
        self.max_workers = max_workers
        self._executor = ThreadPoolExecutor(max_workers=max_workers)
        self._running = False
        self._job_handlers: dict[JobType, JobExecutorFunc] = {}
        self._async_job_handlers: dict[JobType, AsyncJobExecutorFunc] = {}
        self._worker_thread: threading.Thread | None = None
        self._stop_event = threading.Event()
        # Event to wake the worker immediately when a new job is submitted
        self._new_job_event = threading.Event()
        self._futures: dict[str, Future] = {}
        # Condition to signal when a job Future is registered
        self._futures_cond = threading.Condition()

    def register_handler(self, job_type: JobType, handler: JobExecutorFunc) -> None:
        """Register a synchronous job handler.

        Args:
            job_type: Type of job this handler processes
            handler: Function that executes the job
        """
        self._job_handlers[job_type] = handler
        logger.info(f"Registered sync handler for job type: {job_type}")

    def register_async_handler(
        self, job_type: JobType, handler: AsyncJobExecutorFunc
    ) -> None:
        """Register an asynchronous job handler.

        Args:
            job_type: Type of job this handler processes
            handler: Async function that executes the job
        """
        self._async_job_handlers[job_type] = handler
        logger.info(f"Registered async handler for job type: {job_type}")

    def start(self) -> None:
        """Start the job executor worker thread."""
        if self._running:
            logger.warning("JobExecutor is already running")
            return

        self._running = True
        self._stop_event.clear()
        self._worker_thread = threading.Thread(target=self._worker_loop, daemon=True)
        self._worker_thread.start()
        logger.info("JobExecutor started")

    def stop(self, timeout: float = 30.0) -> None:
        """Stop the job executor and wait for current jobs to finish.

        Args:
            timeout: Maximum time to wait for shutdown
        """
        if not self._running:
            return

        logger.info("Stopping JobExecutor...")
        self._running = False
        self._stop_event.set()
        # Wake the worker loop if it's waiting for new jobs
        self._new_job_event.set()

        if self._worker_thread and self._worker_thread.is_alive():
            self._worker_thread.join(timeout=timeout)

        self._executor.shutdown(wait=True)
        logger.info("JobExecutor stopped")

    def submit_job(
        self,
        job_type: JobType,
        model_id: int | None = None,
        message: str = "",
        sql_query: str | None = None,
    ) -> Job:
        """Submit a new job for execution.

        Args:
            job_type: Type of job to execute
            model_id: ID of the model being processed (for train/predict jobs)
            message: Initial job message
            sql_query: SQL query for data (for train jobs: "on <table>")

        Returns:
            Created Job object

        Raises:
            ValueError: If no handler is registered for the job type
        """
        if (
            job_type not in self._job_handlers
            and job_type not in self._async_job_handlers
        ):
            raise ValueError(f"No handler registered for job type: {job_type}")

        job = self.job_manager.create_job(job_type, model_id, message, sql_query)
        logger.info(f"Submitted job {job.job_id} for execution")
        # Wake the worker loop immediately to schedule this job
        self._new_job_event.set()
        return job

    def cancel_job(self, job_id: str) -> bool:
        """Cancel a job by ID.

        Args:
            job_id: Job ID to cancel

        Returns:
            True if job was cancelled, False otherwise
        """
        return self.job_manager.cancel_job(job_id)

    def _worker_loop(self) -> None:
        """Main worker loop that processes pending jobs."""
        logger.info("Worker loop started")

        while self._running:
            try:
                # Get pending jobs
                pending_jobs = [
                    job
                    for job in self.job_manager.get_active_jobs()
                    if job.status == JobStatus.PENDING
                ]

                for job in pending_jobs:
                    if not self._running:
                        break

                    # Store the future before submission to avoid race condition
                    # where wait_for_job is called before the future is registered
                    with self._futures_cond:
                        # Submit job for execution
                        fut = self._executor.submit(self._execute_job, job)
                        # Store the future and notify any waiter
                        self._futures[job.job_id] = fut
                        self._futures_cond.notify_all()
                    # Don't wait for completion - let it run async

                # Block until a new job arrives or stop is requested
                if self._stop_event.is_set():
                    break
                # Clear any previous signal and wait indefinitely for the next
                self._new_job_event.clear()
                # This wait removes polling delays entirely
                self._new_job_event.wait()

            except Exception as e:
                logger.error(f"Error in worker loop: {e}")
                # On error, wait for either a new job signal or stop
                if self._stop_event.is_set():
                    break
                self._new_job_event.clear()
                self._new_job_event.wait()

        logger.info("Worker loop stopped")

    def _execute_job(self, job: Job) -> None:
        """Execute a single job.

        Args:
            job: Job to execute
        """
        try:
            # Mark job as started
            job.start("Executing job")
            self.job_manager.update_job(job)
            logger.info(f"Started executing job {job.job_id}")

            # Get the appropriate handler
            if job.type in self._job_handlers:
                handler = self._job_handlers[job.type]
                handler(job)
            elif job.type in self._async_job_handlers:
                handler = self._async_job_handlers[job.type]
                # Run async handler in new event loop
                asyncio.run(handler(job))
            else:
                raise ValueError(f"No handler found for job type: {job.type}")

            # Mark job as completed
            current_job = self.job_manager.get_job(job.job_id)
            if current_job and current_job.status != JobStatus.CANCELLED:
                job.complete("Job completed successfully")
                self.job_manager.update_job(job)
                logger.info(f"Completed job {job.job_id}")

        except Exception as e:
            logger.error(f"Job {job.job_id} failed: {e}")
            job.fail(str(e))
            self.job_manager.update_job(job)

    def wait_for_job(self, job_id: str, timeout: float = 30.0) -> None:
        """Block until the specified job's handler has finished executing.

        This waits on the executor Future rather than using timeouts or sleeps.
        If the job hasn't been scheduled yet, it spins briefly on the worker
        loop handoff without sleeping, but returns as soon as the Future exists
        and completes.

        Args:
            job_id: Job ID to wait for
            timeout: Maximum time to wait for job completion (default 30s)
        """
        # Try to get an existing future; if not present, wait until the
        # worker loop schedules it without busy-waiting.
        with self._futures_cond:
            fut = self._futures.get(job_id)
            while fut is None and self._running:
                # Use timeout to prevent indefinite blocking in CI environments
                if not self._futures_cond.wait(timeout=timeout):
                    logger.warning(f"Timeout waiting for job {job_id} to be scheduled")
                    return
                fut = self._futures.get(job_id)
        if fut is not None:
            # Propagate any exception from the job handler
            try:
                fut.result(timeout=timeout)
            except Exception:
                # Re-raise the exception from the job handler
                raise

    def update_job_message(self, job_id: str, message: str) -> None:
        """Update job progress message.

        Args:
            job_id: Job ID to update
            message: Progress message
        """
        job = self.job_manager.get_job(job_id)
        if job:
            job.update_message(message)
            self.job_manager.update_job(job)

    @property
    def is_running(self) -> bool:
        """Check if the executor is currently running."""
        return self._running

    def get_stats(self) -> dict:
        """Get executor statistics.

        Returns:
            Dictionary with executor stats
        """
        stats = self.job_manager.get_job_statistics()
        stats.update(
            {
                "max_workers": self.max_workers,
                "is_running": self._running,
            }
        )
        return stats
