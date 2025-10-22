"""Tests for Snowflake database integration."""

from unittest.mock import MagicMock, patch

import pytest

from arc.database.duckdb import DuckDBDatabase


def test_snowflake_extensions_setup_called_on_connect():
    """Test that Snowflake extension setup is called during connection."""
    with patch("arc.database.duckdb.duckdb.connect") as mock_connect:
        mock_connection = MagicMock()
        mock_connect.return_value = mock_connection

        # Create database instance (triggers _connect)
        with (
            patch.object(DuckDBDatabase, "_setup_s3_extensions") as mock_s3_setup,
            patch.object(
                DuckDBDatabase, "_setup_snowflake_extensions"
            ) as mock_snowflake_setup,
        ):
            _ = DuckDBDatabase(":memory:")

            # Verify both extension setups were called
            mock_s3_setup.assert_called_once()
            mock_snowflake_setup.assert_called_once()


def test_snowflake_extension_installation():
    """Test Snowflake extension installation gracefully handles errors."""
    # Mock SettingsManager before creating database
    with patch("arc.core.config.SettingsManager") as mock_settings_class:
        mock_settings = MagicMock()
        mock_settings.get_snowflake_config.return_value = None
        mock_settings_class.return_value = mock_settings

        # Create database - should not raise errors even if extension unavailable
        try:
            db = DuckDBDatabase(":memory:")
            db.close()
        except Exception as e:
            pytest.fail(
                f"Should handle missing Snowflake extension gracefully, but raised: {e}"
            )


def test_snowflake_credentials_configuration_with_valid_config():
    """Test Snowflake credentials are configured when valid config exists."""
    mock_config = {
        "account": "test.snowflakecomputing.com",
        "user": "testuser",
        "password": "testpass",
        "database": "TESTDB",
        "warehouse": "TESTWH",
        "schema": "PUBLIC",
    }

    with patch("arc.core.config.SettingsManager") as mock_settings_class:
        mock_settings = MagicMock()
        mock_settings.get_snowflake_config.return_value = mock_config
        mock_settings_class.return_value = mock_settings

        # Create database - should attempt to configure Snowflake
        # (will fail to attach since not real Snowflake, but should not raise)
        try:
            db = DuckDBDatabase(":memory:")
            db.close()
        except Exception as e:
            pytest.fail(
                f"Should handle Snowflake configuration gracefully, but raised: {e}"
            )


def test_snowflake_credentials_skipped_when_no_config():
    """Test Snowflake credentials configuration is skipped when no config."""
    with patch("arc.core.config.SettingsManager") as mock_settings_class:
        mock_settings = MagicMock()
        mock_settings.get_snowflake_config.return_value = None
        mock_settings_class.return_value = mock_settings

        # Create database - should skip Snowflake configuration
        try:
            db = DuckDBDatabase(":memory:")
            db.close()
        except Exception as e:
            pytest.fail(f"Should skip Snowflake when no config, but raised: {e}")


def test_snowflake_setup_handles_extension_unavailable():
    """Test graceful handling when Snowflake extension is unavailable."""
    # Even without mocking, should handle extension unavailable gracefully
    try:
        db = DuckDBDatabase(":memory:")
        db.close()
    except Exception as e:
        pytest.fail(f"Should handle missing extension gracefully, but raised: {e}")


def test_snowflake_attach_handles_connection_failure():
    """Test graceful handling when Snowflake connection fails."""
    mock_config = {
        "account": "invalid.snowflakecomputing.com",
        "user": "invaliduser",
        "password": "invalidpass",
        "database": "INVALIDDB",
        "warehouse": "INVALIDWH",
        "schema": "PUBLIC",
    }

    with patch("arc.core.config.SettingsManager") as mock_settings_class:
        mock_settings = MagicMock()
        mock_settings.get_snowflake_config.return_value = mock_config
        mock_settings_class.return_value = mock_settings

        # Create database - should handle invalid credentials gracefully
        try:
            db = DuckDBDatabase(":memory:")
            db.close()
        except Exception as e:
            pytest.fail(f"Should handle connection failure gracefully, but raised: {e}")


def test_snowflake_configuration_values():
    """Test that Snowflake configuration is correctly read and used."""
    mock_config = {
        "account": "myaccount.snowflakecomputing.com",
        "user": "myuser",
        "password": "mypass",
        "database": "MYDB",
        "warehouse": "MYWH",
        "schema": "MYSCHEMA",
    }

    with patch("arc.core.config.SettingsManager") as mock_settings_class:
        mock_settings = MagicMock()
        mock_settings.get_snowflake_config.return_value = mock_config
        mock_settings_class.return_value = mock_settings

        # Verify configuration is read
        db = DuckDBDatabase(":memory:")

        # Verify SettingsManager was instantiated and get_snowflake_config was called
        assert mock_settings_class.called
        assert mock_settings.get_snowflake_config.called

        db.close()
