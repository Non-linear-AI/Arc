"""Configuration management for Arc CLI."""

import json
import os
from pathlib import Path
from typing import Any

from arc.utils.validation import ValidationError, validate_api_key, validate_url


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
        """Update a single user setting with validation.

        Args:
            key: Setting key to update
            value: Setting value

        Raises:
            ValidationError: If the value is invalid for the given key

        Note:
            API key can be empty to support enterprise gateway environments.
        """
        # Validate based on key
        if key == "apiKey":
            # Allow empty API keys for enterprise gateway mode
            value = validate_api_key(value)
        elif key == "baseURL":
            value = validate_url(value)
        elif key == "model":
            # Validate model name (non-empty string)
            if not value or not isinstance(value, str):
                raise ValidationError("Model name must be a non-empty string")
            value = value.strip()
            if not value:
                raise ValidationError("Model name cannot be only whitespace")
        elif key == "systemDatabasePath" or key == "userDatabasePath":
            # Validate database path is non-empty string
            if not value or not isinstance(value, str):
                raise ValidationError(f"{key} must be a non-empty string")
            value = value.strip()
            if not value:
                raise ValidationError(f"{key} cannot be only whitespace")
        elif key == "tensorboardMode":
            # Already validated in set_tensorboard_mode(), but validate here too
            if value not in ("always", "ask", "never"):
                raise ValidationError(
                    f"Invalid tensorboard mode: {value}. "
                    "Must be 'always', 'ask', or 'never'"
                )
        elif key == "tensorboardPort" and (
            not isinstance(value, int) or not (1024 <= value <= 65535)
        ):
            # Already validated in set_tensorboard_port(), but validate here too
            raise ValidationError(
                f"Invalid port: {value}. Must be an integer between 1024 and 65535"
            )

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
        """Get base URL from environment or settings.

        Returns:
            Base URL, defaults to "https://api.openai.com/v1" if not configured

        Note:
            Empty strings from environment variables are treated as not configured.
        """
        # First check environment
        base_url = os.getenv("ARC_BASE_URL")
        if base_url and base_url.strip():
            return base_url.strip()

        # Then check settings file
        settings = self.load_user_settings()
        base_url = settings.get("baseURL")
        if base_url and base_url.strip():
            return base_url.strip()

        # Return default if nothing configured
        return "https://api.openai.com/v1"

    def get_current_model(self) -> str:
        """Get current model from environment or settings.

        Returns:
            Model name, defaults to "gpt-4o" if not configured
        """
        # First check environment
        model = os.getenv("ARC_MODEL")
        if model:
            return model

        # Then check settings file
        settings = self.load_user_settings()
        model = settings.get("model")

        # Return default model if not configured
        if not model:
            return "gpt-4o"

        return model

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

        # Default to project-local .arc directory
        return str(Path(".arc") / "arc_system.db")

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

        # Default to project-local .arc directory
        return str(Path(".arc") / "arc_user.db")

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

    def get_verbose_mode(self) -> bool:
        """Get verbose/debug mode setting.

        Returns:
            True if verbose mode is enabled via environment variable or settings

        Note:
            Checks ARC_VERBOSE environment variable first, then settings file.
            Accepts: "1", "true", "yes" (case-insensitive)
        """
        # First check environment variable
        verbose_env = os.getenv("ARC_VERBOSE", "").lower()
        if verbose_env in ("1", "true", "yes"):
            return True

        # Then check settings file
        settings = self.load_user_settings()
        verbose_setting = settings.get("verbose", False)
        return bool(verbose_setting)

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

    def get_s3_config(self) -> dict[str, str] | None:
        """Get S3 configuration from environment variables or settings.

        Returns:
            Dict with S3 credentials (access_key_id, secret_access_key, region,
                endpoint)
            or None if no credentials configured.

        Priority:
            1. Environment variables (AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY, etc.)
            2. Arc user settings file (~/.arc/user-settings.json)
        """
        # Check environment variables first
        env_access_key = os.getenv("AWS_ACCESS_KEY_ID")
        env_secret_key = os.getenv("AWS_SECRET_ACCESS_KEY")
        env_region = os.getenv("AWS_REGION") or os.getenv("AWS_DEFAULT_REGION")
        env_endpoint = os.getenv("AWS_ENDPOINT_URL")

        # Check settings file
        settings = self.load_user_settings()
        settings_access_key = settings.get("awsAccessKeyId")
        settings_secret_key = settings.get("awsSecretAccessKey")
        settings_region = settings.get("awsRegion")
        settings_endpoint = settings.get("s3Endpoint")

        # Combine (environment takes precedence)
        access_key = env_access_key or settings_access_key
        secret_key = env_secret_key or settings_secret_key
        region = env_region or settings_region or "us-east-1"  # Default region
        endpoint = env_endpoint or settings_endpoint

        # Return None if no credentials configured
        if not access_key or not secret_key:
            return None

        config = {
            "access_key_id": access_key,
            "secret_access_key": secret_key,
            "region": region,
        }

        if endpoint:
            config["endpoint"] = endpoint

        return config

    def get_snowflake_config(self) -> dict[str, str] | None:
        """Get Snowflake configuration from environment variables or settings.

        Returns:
            Dict with Snowflake credentials (account, user, password, database,
                warehouse, schema) or None if incomplete configuration.

        Priority:
            1. Environment variables (SNOWFLAKE_*)
            2. Arc user settings file (~/.arc/user-settings.json)
        """
        # Check environment variables first
        env_account = os.getenv("SNOWFLAKE_ACCOUNT")
        env_user = os.getenv("SNOWFLAKE_USER")
        env_password = os.getenv("SNOWFLAKE_PASSWORD")
        env_database = os.getenv("SNOWFLAKE_DATABASE")
        env_warehouse = os.getenv("SNOWFLAKE_WAREHOUSE")
        env_schema = os.getenv("SNOWFLAKE_SCHEMA")

        # Check settings file
        settings = self.load_user_settings()
        settings_account = settings.get("snowflakeAccount")
        settings_user = settings.get("snowflakeUser")
        settings_password = settings.get("snowflakePassword")
        settings_database = settings.get("snowflakeDatabase")
        settings_warehouse = settings.get("snowflakeWarehouse")
        settings_schema = settings.get("snowflakeSchema")

        # Combine (environment takes precedence)
        account = env_account or settings_account
        user = env_user or settings_user
        password = env_password or settings_password
        database = env_database or settings_database
        warehouse = env_warehouse or settings_warehouse
        schema = env_schema or settings_schema or "PUBLIC"  # Default schema

        # Return None if required credentials are missing
        if not all([account, user, password, database, warehouse]):
            return None

        return {
            "account": account,
            "user": user,
            "password": password,
            "database": database,
            "warehouse": warehouse,
            "schema": schema,
        }
