"""Tests for TensorBoard path management."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from arc.ml.tensorboard import TensorBoardError, TensorBoardManager


@pytest.fixture
def tb_manager():
    """Create a fresh TensorBoard manager instance for each test."""
    # Since TensorBoardManager is a singleton, we need to reset it
    manager = TensorBoardManager()
    # Clear any existing processes
    manager._processes.clear()
    yield manager
    # Cleanup after test
    manager.stop_all()
    manager._processes.clear()


def test_tensorboard_log_dir_project_local(tb_manager):
    """Test that TensorBoard logs default to project-local .arc/tensorboard directory."""
    job_id = "test_job_123"

    # Mock subprocess.Popen to avoid actually launching tensorboard
    with patch("arc.ml.tensorboard.subprocess.Popen") as mock_popen:
        mock_process = MagicMock()
        mock_process.pid = 12345
        mock_popen.return_value = mock_process

        # Launch for job (uses default path)
        url, pid = tb_manager.launch_for_job(job_id, port=6006)

        # Verify the logdir argument passed to tensorboard command
        call_args = mock_popen.call_args
        cmd_list = call_args[0][0]

        # Find logdir in command
        logdir_idx = cmd_list.index("--logdir")
        actual_logdir = cmd_list[logdir_idx + 1]

        # Should be project-local .arc/tensorboard/run_{job_id}
        expected_logdir = str(Path(".arc") / "tensorboard" / f"run_{job_id}")
        assert actual_logdir == expected_logdir

        # Verify URL and PID
        assert url.startswith("http://localhost:")
        assert pid == 12345


def test_tensorboard_log_dir_custom_path(tb_manager, tmp_path):
    """Test that TensorBoard can use custom log directory path."""
    job_id = "custom_job_456"
    custom_logdir = tmp_path / "custom_logs" / "tensorboard"

    with patch("arc.ml.tensorboard.subprocess.Popen") as mock_popen:
        mock_process = MagicMock()
        mock_process.pid = 67890
        mock_popen.return_value = mock_process

        # Launch with custom path
        url, pid = tb_manager.launch(job_id, logdir=custom_logdir, port=6007)

        # Verify custom logdir was used
        call_args = mock_popen.call_args
        cmd_list = call_args[0][0]
        logdir_idx = cmd_list.index("--logdir")
        actual_logdir = cmd_list[logdir_idx + 1]

        assert actual_logdir == str(custom_logdir)


def test_tensorboard_creates_logdir_if_missing(tb_manager, tmp_path):
    """Test that TensorBoard creates log directory if it doesn't exist."""
    job_id = "mkdir_test"
    logdir = tmp_path / "new_logs" / "tensorboard"

    # Verify directory doesn't exist yet
    assert not logdir.exists()

    with patch("arc.ml.tensorboard.subprocess.Popen") as mock_popen:
        mock_process = MagicMock()
        mock_process.pid = 11111
        mock_popen.return_value = mock_process

        tb_manager.launch(job_id, logdir=logdir, port=6008)

    # Verify directory was created
    assert logdir.exists()
    assert logdir.is_dir()


def test_tensorboard_process_tracking(tb_manager):
    """Test that TensorBoard manager tracks running processes."""
    job_id = "tracking_test"

    with patch("arc.ml.tensorboard.subprocess.Popen") as mock_popen:
        mock_process = MagicMock()
        mock_process.pid = 22222
        mock_process.poll.return_value = None  # Process is running
        mock_popen.return_value = mock_process

        # Launch tensorboard
        url, pid = tb_manager.launch_for_job(job_id, port=6009)

        # Verify it's tracked
        assert tb_manager.is_running(job_id)
        assert tb_manager.get_url(job_id) == url

        # Verify it's in the running list
        running = tb_manager.list_running()
        assert len(running) == 1
        assert running[0]["job_id"] == job_id
        assert running[0]["pid"] == pid


def test_tensorboard_singleton_behavior():
    """Test that TensorBoardManager is a singleton."""
    manager1 = TensorBoardManager()
    manager2 = TensorBoardManager()

    # Should be the same instance
    assert manager1 is manager2

    # Should share process registry
    with patch("arc.ml.tensorboard.subprocess.Popen") as mock_popen:
        mock_process = MagicMock()
        mock_process.pid = 33333
        mock_process.poll.return_value = None
        mock_popen.return_value = mock_process

        manager1.launch_for_job("singleton_test", port=6010)

    # Process should be visible from both references
    assert manager2.is_running("singleton_test")

    # Cleanup
    manager1.stop_all()


def test_tensorboard_not_in_home_directory(tb_manager):
    """Test that TensorBoard logs are NOT placed in home directory."""
    job_id = "no_home_test"

    with patch("arc.ml.tensorboard.subprocess.Popen") as mock_popen:
        mock_process = MagicMock()
        mock_process.pid = 44444
        mock_popen.return_value = mock_process

        tb_manager.launch_for_job(job_id, port=6011)

        # Get the logdir from the call
        call_args = mock_popen.call_args
        cmd_list = call_args[0][0]
        logdir_idx = cmd_list.index("--logdir")
        actual_logdir = cmd_list[logdir_idx + 1]

        # Should NOT contain home directory path
        home_str = str(Path.home())
        assert home_str not in actual_logdir

        # Should be relative path starting with .arc
        assert actual_logdir.startswith(".arc")


def test_tensorboard_stop_cleanup(tb_manager):
    """Test that stopping TensorBoard cleans up process tracking."""
    job_id = "cleanup_test"

    with patch("arc.ml.tensorboard.subprocess.Popen") as mock_popen:
        mock_process = MagicMock()
        mock_process.pid = 55555
        mock_process.poll.return_value = None
        mock_popen.return_value = mock_process

        # Launch
        tb_manager.launch_for_job(job_id, port=6012)
        assert tb_manager.is_running(job_id)

        # Stop
        stopped = tb_manager.stop(job_id)
        assert stopped is True

        # Should no longer be running
        assert not tb_manager.is_running(job_id)
        assert tb_manager.get_url(job_id) is None


def test_tensorboard_port_allocation(tb_manager):
    """Test that TensorBoard uses correct port."""
    job_id = "port_test"
    custom_port = 7777

    with patch("arc.ml.tensorboard.subprocess.Popen") as mock_popen:
        mock_process = MagicMock()
        mock_process.pid = 66666
        mock_popen.return_value = mock_process

        # Mock port availability check to always return True
        with patch.object(tb_manager, "_is_port_available", return_value=True):
            url, _ = tb_manager.launch_for_job(job_id, port=custom_port)

        # Verify port is in URL
        assert f":{custom_port}" in url

        # Verify port was passed to tensorboard command
        call_args = mock_popen.call_args
        cmd_list = call_args[0][0]
        port_idx = cmd_list.index("--port")
        actual_port = cmd_list[port_idx + 1]
        assert actual_port == str(custom_port)


def test_tensorboard_error_when_not_installed(tb_manager):
    """Test that appropriate error is raised when tensorboard is not installed."""
    job_id = "install_test"

    # Mock subprocess.Popen to raise FileNotFoundError (tensorboard not found)
    with (
        patch("arc.ml.tensorboard.subprocess.Popen", side_effect=FileNotFoundError()),
        pytest.raises(
            TensorBoardError, match="TensorBoard not found.*pip install tensorboard"
        ),
    ):
        tb_manager.launch_for_job(job_id, port=6013)
