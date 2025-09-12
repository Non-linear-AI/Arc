"""Tests for the main module."""

from unittest.mock import patch

from arc import main


def test_main_prints_hello() -> None:
    """Test that main prints the expected message."""
    with patch("builtins.print") as mock_print:
        main()
        mock_print.assert_called_once_with("Hello from arc!")
