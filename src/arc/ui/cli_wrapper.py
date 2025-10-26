"""
CLI wrapper that auto-configures native library paths for Arc.

This module ensures that ADBC Snowflake driver native libraries are
discoverable before Arc starts, enabling seamless Snowflake integration
without manual environment variable setup.

The wrapper automatically detects the platform and sets the appropriate
library path environment variable (LD_LIBRARY_PATH on Linux, DYLD_LIBRARY_PATH
on macOS, PATH on Windows) if needed, then restarts the Python process to
ensure the dynamic linker picks up the changes.
"""

import os
import sys
from pathlib import Path

import adbc_driver_snowflake

from arc.ui.cli import cli


def _get_library_env_var() -> str:
    """Get the platform-specific library path environment variable name."""
    if sys.platform == "darwin":
        return "DYLD_LIBRARY_PATH"
    elif sys.platform == "win32":
        return "PATH"
    else:
        return "LD_LIBRARY_PATH"


def _should_restart() -> tuple[bool, str | None]:
    """
    Check if we need to restart Python to configure library paths.

    Returns:
        Tuple of (should_restart, adbc_lib_dir)
    """
    if os.environ.get("_ARC_RESTARTED"):
        return False, None

    adbc_lib_dir = str(Path(adbc_driver_snowflake.__file__).parent)
    lib_env_var = _get_library_env_var()
    current_path = os.environ.get(lib_env_var, "")

    if adbc_lib_dir in current_path:
        return False, None

    return True, adbc_lib_dir


def auto_configure_and_restart() -> None:
    """
    Auto-configure library paths for ADBC and restart if needed.

    This function checks if the ADBC library path is properly configured.
    If not, it sets the appropriate environment variable and restarts
    Python with the updated environment using os.execv().

    The restart is necessary because the dynamic linker only reads library
    path environment variables at process startup.
    """
    should_restart, adbc_lib_dir = _should_restart()

    if not should_restart:
        return

    lib_env_var = _get_library_env_var()
    separator = ";" if sys.platform == "win32" else ":"

    current_path = os.environ.get(lib_env_var, "")
    new_path = (
        f"{adbc_lib_dir}{separator}{current_path}" if current_path else adbc_lib_dir
    )
    os.environ[lib_env_var] = new_path
    os.environ["_ARC_RESTARTED"] = "1"

    try:
        os.execv(sys.executable, [sys.executable] + sys.argv)
    except Exception as e:
        print(
            f"Warning: Could not auto-configure library paths: {e}",
            file=sys.stderr,
        )
        print(
            f"Snowflake integration may not work. Please manually set {lib_env_var}.",
            file=sys.stderr,
        )


def cli_main() -> None:
    """
    Main entry point for Arc CLI with automatic library path configuration.

    This wrapper ensures native libraries for Snowflake integration are
    discoverable before starting Arc. If needed, it automatically restarts
    the Python process with proper environment variables set.
    """
    auto_configure_and_restart()
    cli()


if __name__ == "__main__":
    cli_main()
