"""Configuration management for Arc CLI."""

import json
import os
from pathlib import Path
from typing import Any


class SettingsManager:
    """Manages user settings and configuration."""

    def __init__(self, settings_dir: str | None = None):
        self.settings_dir = Path(settings_dir or Path.home() / ".arc")
        self.settings_file = self.settings_dir / "user-settings.json"
        self.settings_dir.mkdir(exist_ok=True)

    def load_user_settings(self) -> dict[str, Any]:
        """Load user settings from file."""
        if not self.settings_file.exists():
            return {}

        try:
            with open(self.settings_file, encoding="utf-8") as f:
                return json.load(f)
        except (OSError, json.JSONDecodeError):
            return {}

    def save_user_settings(self, settings: dict[str, Any]) -> None:
        """Save user settings to file."""
        try:
            with open(self.settings_file, "w", encoding="utf-8") as f:
                json.dump(settings, f, indent=2)
        except OSError as e:
            print(f"Warning: Could not save settings: {e}")

    def update_user_setting(self, key: str, value: Any) -> None:
        """Update a single user setting."""
        settings = self.load_user_settings()
        settings[key] = value
        self.save_user_settings(settings)

    def get_api_key(self) -> str | None:
        """Get API key from environment or settings."""
        # First check environment
        api_key = os.getenv("ARC_API_KEY")
        if api_key:
            return api_key

        # Then check settings file
        settings = self.load_user_settings()
        return settings.get("apiKey")

    def get_base_url(self) -> str:
        """Get base URL from environment or settings."""
        # First check environment
        base_url = os.getenv("ARC_BASE_URL")
        if base_url:
            return base_url

        # Then check settings file
        settings = self.load_user_settings()
        return settings.get("baseURL", "https://api.openai.com/v1")

    def get_current_model(self) -> str | None:
        """Get current model from environment or settings."""
        # First check environment
        model = os.getenv("ARC_MODEL")
        if model:
            return model

        # Then check settings file
        settings = self.load_user_settings()
        return settings.get("model")

    def get_system_database_path(self) -> str:
        """Get system database path from environment or settings."""
        # First check environment
        db_path = os.getenv("ARC_SYSTEM_DATABASE_PATH")
        if db_path:
            return db_path

        # Then check settings file
        settings = self.load_user_settings()
        db_path = settings.get("systemDatabasePath")
        if db_path:
            return db_path

        # Default to system database file in settings directory
        return str(self.settings_dir / "arc_system.db")

    def get_user_database_path(self) -> str:
        """Get user database path from environment or settings."""
        # First check environment
        db_path = os.getenv("ARC_USER_DATABASE_PATH")
        if db_path:
            return db_path

        # Then check settings file
        settings = self.load_user_settings()
        db_path = settings.get("userDatabasePath")
        if db_path:
            return db_path

        # Default to user database file in settings directory
        return str(self.settings_dir / "arc_user.db")

    def set_user_database_path(self, db_path: str) -> None:
        """Set user database path in settings."""
        self.update_user_setting("userDatabasePath", db_path)

    def get_tensorboard_mode(self) -> str | None:
        """Get TensorBoard launch mode preference.

        Returns:
            One of "always", "ask", "never", or None if not set
        """
        settings = self.load_user_settings()
        return settings.get("tensorboardMode")

    def set_tensorboard_mode(self, mode: str) -> None:
        """Set TensorBoard launch mode preference.

        Args:
            mode: One of "always", "ask", "never"

        Raises:
            ValueError: If mode is not valid
        """
        if mode not in ("always", "ask", "never"):
            raise ValueError(
                f"Invalid tensorboard mode: {mode}. Must be 'always', 'ask', or 'never'"
            )
        self.update_user_setting("tensorboardMode", mode)

    def get_tensorboard_port(self) -> int:
        """Get preferred TensorBoard port.

        Returns:
            Port number (default: 6006)
        """
        settings = self.load_user_settings()
        return settings.get("tensorboardPort", 6006)

    def set_tensorboard_port(self, port: int) -> None:
        """Set preferred TensorBoard port.

        Args:
            port: Port number

        Raises:
            ValueError: If port is not valid
        """
        if not (1024 <= port <= 65535):
            raise ValueError(f"Invalid port: {port}. Must be between 1024 and 65535")
        self.update_user_setting("tensorboardPort", port)
