"""Arc Jobs Module - Async job management system."""

from arc.jobs.models import Job, JobStatus, JobType

# Avoid circular imports by using lazy imports for manager and executor
__all__ = ["Job", "JobStatus", "JobType", "JobManager", "JobExecutor"]


def __getattr__(name: str):
    """Lazy import for JobManager and JobExecutor to avoid circular imports."""
    if name == "JobManager":
        from arc.jobs.manager import JobManager

        return JobManager
    elif name == "JobExecutor":
        from arc.jobs.executor import JobExecutor

        return JobExecutor
    raise AttributeError(f"module '{__name__}' has no attribute '{name}'")
