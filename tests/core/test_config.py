"""Tests for configuration path management."""

from __future__ import annotations

from pathlib import Path

import pytest

from arc.core.config import SettingsManager


@pytest.fixture
def temp_settings_dir(tmp_path):
    """Create a temporary settings directory for testing."""
    return tmp_path / "test_arc_settings"


@pytest.fixture
def settings_manager(temp_settings_dir):
    """Create a SettingsManager instance with temporary settings directory."""
    return SettingsManager(settings_dir=str(temp_settings_dir))


def test_settings_dir_in_home_by_default():
    """Test that settings directory defaults to ~/.arc when not specified."""
    manager = SettingsManager()
    expected_path = Path.home() / ".arc"
    assert manager.settings_dir == expected_path


def test_settings_dir_custom(temp_settings_dir):
    """Test that settings directory can be customized."""
    manager = SettingsManager(settings_dir=str(temp_settings_dir))
    assert manager.settings_dir == temp_settings_dir


def test_system_database_path_project_local():
    """Test that system database path defaults to project-local .arc/ directory."""
    manager = SettingsManager()
    db_path = manager.get_system_database_path()

    # Should be relative path in project-local .arc/
    expected = str(Path(".arc") / "arc_system.db")
    assert db_path == expected


def test_user_database_path_project_local():
    """Test that user database path defaults to project-local .arc/ directory."""
    manager = SettingsManager()
    db_path = manager.get_user_database_path()

    # Should be relative path in project-local .arc/
    expected = str(Path(".arc") / "arc_user.db")
    assert db_path == expected


def test_system_database_path_env_override(monkeypatch):
    """Test that ARC_SYSTEM_DATABASE_PATH environment variable overrides default."""
    custom_path = "/custom/path/system.db"
    monkeypatch.setenv("ARC_SYSTEM_DATABASE_PATH", custom_path)

    manager = SettingsManager()
    db_path = manager.get_system_database_path()

    assert db_path == custom_path


def test_user_database_path_env_override(monkeypatch):
    """Test that ARC_USER_DATABASE_PATH environment variable overrides default."""
    custom_path = "/custom/path/user.db"
    monkeypatch.setenv("ARC_USER_DATABASE_PATH", custom_path)

    manager = SettingsManager()
    db_path = manager.get_user_database_path()

    assert db_path == custom_path


def test_database_path_settings_override(settings_manager, temp_settings_dir):
    """Test that settings file can override database paths."""
    # Set custom database paths in settings
    custom_system_path = "/custom/system.db"
    custom_user_path = "/custom/user.db"

    settings_manager.update_user_setting("systemDatabasePath", custom_system_path)
    settings_manager.update_user_setting("userDatabasePath", custom_user_path)

    # Create new manager to load settings
    manager = SettingsManager(settings_dir=str(temp_settings_dir))

    assert manager.get_system_database_path() == custom_system_path
    assert manager.get_user_database_path() == custom_user_path


def test_env_overrides_settings_for_database_paths(
    settings_manager, temp_settings_dir, monkeypatch
):
    """Test that environment variables take precedence over settings file."""
    # Set paths in settings file
    settings_manager.update_user_setting("systemDatabasePath", "/settings/system.db")
    settings_manager.update_user_setting("userDatabasePath", "/settings/user.db")

    # Set paths in environment (should override)
    env_system_path = "/env/system.db"
    env_user_path = "/env/user.db"
    monkeypatch.setenv("ARC_SYSTEM_DATABASE_PATH", env_system_path)
    monkeypatch.setenv("ARC_USER_DATABASE_PATH", env_user_path)

    # Create new manager
    manager = SettingsManager(settings_dir=str(temp_settings_dir))

    # Environment should win
    assert manager.get_system_database_path() == env_system_path
    assert manager.get_user_database_path() == env_user_path


def test_settings_file_location_unchanged(settings_manager, temp_settings_dir):
    """Test that settings file is still stored in settings directory, not .arc/."""
    # Save a setting
    settings_manager.update_user_setting("model", "test-model")

    # Verify settings file is in settings directory
    expected_file = temp_settings_dir / "user-settings.json"
    assert expected_file.exists()

    # Verify it's NOT in .arc/
    arc_settings = Path(".arc") / "user-settings.json"
    assert not arc_settings.exists()


def test_database_paths_independent_of_settings_dir(tmp_path):
    """Test that database paths are independent of settings directory location."""
    # Create manager with custom settings dir
    custom_settings = tmp_path / "custom_settings"
    manager = SettingsManager(settings_dir=str(custom_settings))

    # Database paths should still be in project-local .arc/
    system_db = manager.get_system_database_path()
    user_db = manager.get_user_database_path()

    assert system_db == str(Path(".arc") / "arc_system.db")
    assert user_db == str(Path(".arc") / "arc_user.db")

    # But settings dir should be custom
    assert manager.settings_dir == custom_settings
