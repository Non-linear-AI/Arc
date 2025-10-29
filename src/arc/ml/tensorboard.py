"""TensorBoard process management for Arc training runs."""

import socket
import subprocess
from pathlib import Path


class TensorBoardError(Exception):
    """Exception raised for TensorBoard-related errors."""

    pass


class TensorBoardManager:
    """Manages TensorBoard processes for training runs.

    Provides functionality to launch, stop, and track TensorBoard instances
    for visualizing training metrics. Each instance is associated with a
    specific job ID and runs on a dedicated port.

    This is a singleton class - all instances share the same process registry.
    """

    _instance: "TensorBoardManager | None" = None
    _processes: dict[str, dict] = {}

    def __new__(cls):
        """Create or return the singleton instance."""
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self):
        """Initialize the TensorBoard manager (no-op for singleton)."""
        # Initialization only happens once when _instance is created
        pass

    def launch_for_job(self, job_id: str, port: int = 6006) -> tuple[str, int]:
        """Launch TensorBoard for a training job using default log directory.

        Automatically determines the log directory from the job ID using the
        standard Arc convention: ~/.arc/tensorboard/run_{job_id}

        Args:
            job_id: Training job identifier
            port: Preferred port (will find available if taken)

        Returns:
            Tuple of (url, pid) for the launched TensorBoard instance

        Raises:
            TensorBoardError: If TensorBoard fails to launch
        """
        # Use standard Arc tensorboard directory structure
        logdir = Path.home() / ".arc" / "tensorboard" / f"run_{job_id}"
        return self.launch(job_id, logdir, port)

    def launch(self, job_id: str, logdir: Path, port: int = 6006) -> tuple[str, int]:
        """Launch TensorBoard for a training job.

        Args:
            job_id: Training job identifier
            logdir: Path to TensorBoard logs directory
            port: Preferred port (will find available if taken)

        Returns:
            Tuple of (url, pid) for the launched TensorBoard instance

        Raises:
            TensorBoardError: If TensorBoard fails to launch
        """
        # Check if already running
        if job_id in self._processes:
            info = self._processes[job_id]
            return info["url"], info["pid"]

        # Create logdir if it doesn't exist
        logdir.mkdir(parents=True, exist_ok=True)

        # Find available port
        actual_port = self._find_available_port(port)

        # Launch TensorBoard process
        try:
            process = subprocess.Popen(
                [
                    "tensorboard",
                    "--logdir",
                    str(logdir),
                    "--port",
                    str(actual_port),
                    "--bind_all",
                ],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                start_new_session=True,  # Detach from parent
            )

            url = f"http://localhost:{actual_port}"
            self._processes[job_id] = {
                "process": process,
                "pid": process.pid,
                "port": actual_port,
                "url": url,
                "logdir": str(logdir),
            }

            return url, process.pid

        except FileNotFoundError as exc:
            raise TensorBoardError(
                "TensorBoard not found. Install with: pip install tensorboard"
            ) from exc
        except Exception as exc:
            raise TensorBoardError(f"Failed to launch TensorBoard: {exc}") from exc

    def stop(self, job_id: str) -> bool:
        """Stop TensorBoard for a specific job.

        Args:
            job_id: Training job identifier

        Returns:
            True if stopped, False if not running
        """
        if job_id not in self._processes:
            return False

        info = self._processes[job_id]
        process = info["process"]

        try:
            process.terminate()
            process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            process.kill()
        except Exception:  # noqa: S110
            pass

        del self._processes[job_id]
        return True

    def stop_all(self) -> int:
        """Stop all TensorBoard processes.

        Returns:
            Number of processes stopped
        """
        job_ids = list(self._processes.keys())
        count = 0
        for job_id in job_ids:
            if self.stop(job_id):
                count += 1
        return count

    def list_running(self) -> list[dict]:
        """List all running TensorBoard instances.

        Returns:
            List of dicts with job_id, url, pid, port, logdir for each instance
        """
        # Clean up dead processes
        self._cleanup_dead_processes()

        result = []
        for job_id, info in self._processes.items():
            result.append(
                {
                    "job_id": job_id,
                    "url": info["url"],
                    "pid": info["pid"],
                    "port": info["port"],
                    "logdir": info["logdir"],
                }
            )
        return result

    def is_running(self, job_id: str) -> bool:
        """Check if TensorBoard is running for a job.

        Args:
            job_id: Training job identifier

        Returns:
            True if running, False otherwise
        """
        if job_id not in self._processes:
            return False

        info = self._processes[job_id]
        process = info["process"]

        # Check if process is still alive
        if process.poll() is not None:
            # Process died, clean up
            del self._processes[job_id]
            return False

        return True

    def get_url(self, job_id: str) -> str | None:
        """Get TensorBoard URL for a job.

        Args:
            job_id: Training job identifier

        Returns:
            URL string if running, None otherwise
        """
        if not self.is_running(job_id):
            return None
        return self._processes[job_id]["url"]

    def _find_available_port(self, start_port: int, max_attempts: int = 10) -> int:
        """Find an available port starting from start_port.

        Args:
            start_port: Port to start searching from
            max_attempts: Maximum number of ports to try

        Returns:
            Available port number

        Raises:
            TensorBoardError: If no available port found
        """
        for offset in range(max_attempts):
            port = start_port + offset
            if self._is_port_available(port):
                return port

        raise TensorBoardError(
            f"No available ports found in range "
            f"{start_port}-{start_port + max_attempts - 1}"
        )

    def _is_port_available(self, port: int) -> bool:
        """Check if a port is available.

        Args:
            port: Port number to check

        Returns:
            True if available, False otherwise
        """
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
                sock.bind(("localhost", port))
                return True
        except OSError:
            return False

    def _cleanup_dead_processes(self):
        """Remove dead processes from tracking."""
        dead_jobs = []
        for job_id, info in self._processes.items():
            process = info["process"]
            if process.poll() is not None:
                dead_jobs.append(job_id)

        for job_id in dead_jobs:
            del self._processes[job_id]
