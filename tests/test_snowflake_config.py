"""Tests for Snowflake configuration management."""

from tempfile import TemporaryDirectory

import pytest

from arc.core.config import SettingsManager


@pytest.fixture
def temp_settings_dir():
    """Create a temporary settings directory for testing."""
    with TemporaryDirectory() as tmpdir:
        yield tmpdir


@pytest.fixture
def settings_manager(temp_settings_dir):
    """Create a SettingsManager with temporary settings directory."""
    return SettingsManager(settings_dir=temp_settings_dir)


def test_get_snowflake_config_from_env_vars(settings_manager, monkeypatch):
    """Test get_snowflake_config() reads from environment variables."""
    # Set environment variables
    monkeypatch.setenv("SNOWFLAKE_ACCOUNT", "test.snowflakecomputing.com")
    monkeypatch.setenv("SNOWFLAKE_USER", "testuser")
    monkeypatch.setenv("SNOWFLAKE_PASSWORD", "testpass")
    monkeypatch.setenv("SNOWFLAKE_DATABASE", "TESTDB")
    monkeypatch.setenv("SNOWFLAKE_WAREHOUSE", "TESTWH")
    monkeypatch.setenv("SNOWFLAKE_SCHEMA", "TESTSCHEMA")

    config = settings_manager.get_snowflake_config()

    assert config is not None
    assert config["account"] == "test.snowflakecomputing.com"
    assert config["user"] == "testuser"
    assert config["password"] == "testpass"
    assert config["database"] == "TESTDB"
    assert config["warehouse"] == "TESTWH"
    assert config["schema"] == "TESTSCHEMA"


def test_get_snowflake_config_from_settings_file(settings_manager):
    """Test get_snowflake_config() reads from settings file."""
    # Save settings to file
    settings = {
        "snowflakeAccount": "file.snowflakecomputing.com",
        "snowflakeUser": "fileuser",
        "snowflakePassword": "filepass",
        "snowflakeDatabase": "FILEDB",
        "snowflakeWarehouse": "FILEWH",
        "snowflakeSchema": "FILESCHEMA",
    }
    settings_manager.save_user_settings(settings)

    config = settings_manager.get_snowflake_config()

    assert config is not None
    assert config["account"] == "file.snowflakecomputing.com"
    assert config["user"] == "fileuser"
    assert config["password"] == "filepass"
    assert config["database"] == "FILEDB"
    assert config["warehouse"] == "FILEWH"
    assert config["schema"] == "FILESCHEMA"


def test_get_snowflake_config_env_takes_precedence(settings_manager, monkeypatch):
    """Test environment variables take precedence over settings file."""
    # Save settings to file
    settings_manager.save_user_settings(
        {
            "snowflakeAccount": "file.snowflakecomputing.com",
            "snowflakeUser": "fileuser",
            "snowflakePassword": "filepass",
            "snowflakeDatabase": "FILEDB",
            "snowflakeWarehouse": "FILEWH",
        }
    )

    # Set environment variables (should override)
    monkeypatch.setenv("SNOWFLAKE_ACCOUNT", "env.snowflakecomputing.com")
    monkeypatch.setenv("SNOWFLAKE_USER", "envuser")
    monkeypatch.setenv("SNOWFLAKE_PASSWORD", "envpass")
    monkeypatch.setenv("SNOWFLAKE_DATABASE", "ENVDB")
    monkeypatch.setenv("SNOWFLAKE_WAREHOUSE", "ENVWH")

    config = settings_manager.get_snowflake_config()

    # Environment variables should take precedence
    assert config["account"] == "env.snowflakecomputing.com"
    assert config["user"] == "envuser"
    assert config["password"] == "envpass"
    assert config["database"] == "ENVDB"
    assert config["warehouse"] == "ENVWH"


def test_get_snowflake_config_default_schema(settings_manager, monkeypatch):
    """Test default schema is PUBLIC when not specified."""
    # Set required env vars without schema
    monkeypatch.setenv("SNOWFLAKE_ACCOUNT", "test.snowflakecomputing.com")
    monkeypatch.setenv("SNOWFLAKE_USER", "testuser")
    monkeypatch.setenv("SNOWFLAKE_PASSWORD", "testpass")
    monkeypatch.setenv("SNOWFLAKE_DATABASE", "TESTDB")
    monkeypatch.setenv("SNOWFLAKE_WAREHOUSE", "TESTWH")

    config = settings_manager.get_snowflake_config()

    assert config is not None
    assert config["schema"] == "PUBLIC"


def test_get_snowflake_config_missing_required_fields(settings_manager, monkeypatch):
    """Test returns None when required fields are missing."""
    # Only set some required fields
    monkeypatch.setenv("SNOWFLAKE_ACCOUNT", "test.snowflakecomputing.com")
    monkeypatch.setenv("SNOWFLAKE_USER", "testuser")
    # Missing: password, database, warehouse

    config = settings_manager.get_snowflake_config()
    assert config is None


def test_get_snowflake_config_no_configuration(settings_manager):
    """Test returns None when no configuration exists."""
    config = settings_manager.get_snowflake_config()
    assert config is None


def test_get_snowflake_config_partial_env_partial_settings(
    settings_manager, monkeypatch
):
    """Test combining environment variables and settings file."""
    # Set some values in settings file
    settings_manager.save_user_settings(
        {
            "snowflakePassword": "filepass",
            "snowflakeDatabase": "FILEDB",
            "snowflakeWarehouse": "FILEWH",
        }
    )

    # Set other values in environment (should be combined)
    monkeypatch.setenv("SNOWFLAKE_ACCOUNT", "env.snowflakecomputing.com")
    monkeypatch.setenv("SNOWFLAKE_USER", "envuser")

    config = settings_manager.get_snowflake_config()

    # Should combine both sources
    assert config is not None
    assert config["account"] == "env.snowflakecomputing.com"  # from env
    assert config["user"] == "envuser"  # from env
    assert config["password"] == "filepass"  # from settings
    assert config["database"] == "FILEDB"  # from settings
    assert config["warehouse"] == "FILEWH"  # from settings
