"""Tests for the main module and CLI functionality.

This module contains tests for the main entry point of the Arc CLI application
and its command-line interface. It verifies that:

1. The main function correctly calls the CLI function
2. The CLI version option works as expected
3. The directory option in the chat command works correctly
4. The API key option in the chat command works correctly

These tests use mocking to isolate the components being tested and avoid
actual API calls or file system changes.
"""

from unittest.mock import MagicMock, patch

import pytest
from click.testing import CliRunner

from arc import main
from arc.ui.cli import cli


def test_main_calls_cli() -> None:
    """Test that main calls the CLI function.

    This test verifies that the main entry point correctly calls the CLI function
    from the arc.ui.cli module, which is the primary interface for the application.
    """
    with patch("arc.ui.cli.cli") as mock_cli:
        main()
        mock_cli.assert_called_once()


def test_cli_version():
    """Test that the CLI version option works correctly.

    This test verifies that the --version flag correctly displays the application
    version (0.1.0) and exits with a success code.
    """
    runner = CliRunner()
    result = runner.invoke(cli, ["--version"])
    assert result.exit_code == 0
    assert "version 0.1.0" in result.output


@pytest.mark.parametrize(
    "directory,should_change",
    [
        (".", True),  # Valid directory
        ("non_existent_dir", False),  # Invalid directory
    ],
)
def test_chat_directory_option(directory, should_change):
    """Test that the directory option in chat command works correctly.

    Args:
        directory: The directory to change to
        should_change: Whether the directory change should succeed
    """
    runner = CliRunner()

    with (
        patch("os.chdir") as mock_chdir,
        patch("arc.ui.cli.SettingsManager") as mock_settings,
        patch("arc.ui.cli.run_interactive_mode"),
    ):
        # Setup mock settings manager
        mock_settings_instance = MagicMock()
        mock_settings_instance.get_api_key.return_value = "test-api-key"
        mock_settings.return_value = mock_settings_instance

        # If directory is invalid, os.chdir will raise OSError
        if not should_change:
            mock_chdir.side_effect = OSError("Directory not found")

        result = runner.invoke(cli, ["chat", "-d", directory])

        if should_change:
            mock_chdir.assert_called_once_with(directory)
            assert result.exit_code == 0
        else:
            mock_chdir.assert_called_once_with(directory)
            assert result.exit_code == 1
            assert "Error changing directory" in result.output


def test_chat_api_key_option():
    """Test that the API key option in chat command works correctly.

    This test verifies that the API key provided via command line option
    takes precedence over the one from settings.
    """
    runner = CliRunner()
    test_api_key = "test-api-key-from-option"

    with (
        patch("arc.ui.cli.SettingsManager") as mock_settings,
        patch("arc.ui.cli.run_interactive_mode") as mock_run,
    ):
        # Setup mock settings manager
        mock_settings_instance = MagicMock()
        mock_settings_instance.get_api_key.return_value = "test-api-key-from-settings"
        mock_settings.return_value = mock_settings_instance

        result = runner.invoke(cli, ["chat", "-k", test_api_key])

        # Verify run_interactive_mode was called with the correct API key
        mock_run.assert_called_once()
        args, _ = mock_run.call_args
        assert args[0] == test_api_key
        assert result.exit_code == 0
