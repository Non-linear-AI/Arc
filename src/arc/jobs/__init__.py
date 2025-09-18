"""Arc Jobs Module - Async job management system."""

from .models import Job, JobStatus, JobType

# Avoid circular imports by using lazy imports for manager and executor
__all__ = ["Job", "JobStatus", "JobType", "JobManager", "JobExecutor"]


def __getattr__(name: str):
    """Lazy import for JobManager and JobExecutor to avoid circular imports."""
    if name == "JobManager":
        from .manager import JobManager

        return JobManager
    elif name == "JobExecutor":
        from .executor import JobExecutor

        return JobExecutor
    raise AttributeError(f"module '{__name__}' has no attribute '{name}'")
