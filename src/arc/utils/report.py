"""GitHub issue reporter utilities for Arc CLI.

This module prepares system context and constructs a prefilled GitHub issue URL.
It intentionally avoids UI dependencies; callers (e.g., CLI) handle prompts.
"""

from __future__ import annotations

import platform
import sys
import urllib.parse
import webbrowser

GITHUB_ISSUES_URL = "https://github.com/non-linear-ai/arc/issues/new"


def get_python_info() -> str:
    implementation = platform.python_implementation()
    is_venv = sys.prefix != getattr(sys, "base_prefix", sys.prefix)
    return (
        f"Python implementation: {implementation}\n"
        f"Virtual environment: {'Yes' if is_venv else 'No'}"
    )


def get_os_info() -> str:
    try:
        arch = platform.architecture()[0]
    except Exception:
        arch = "unknown"
    return f"OS: {platform.system()} {platform.release()} ({arch})"


def get_torch_info() -> str:
    try:
        import torch  # type: ignore

        ver = getattr(torch, "__version__", None) or "unknown"
        return f"PyTorch version: {ver}"
    except Exception:
        return "PyTorch version: Unavailable"


def get_database_info() -> str:
    """Return basic database engine information, if available."""
    try:
        import duckdb  # type: ignore

        ver = getattr(duckdb, "__version__", None) or "unknown"
        return f"Database engine: DuckDB {ver}"
    except Exception:
        return "Database engine: Unavailable"


def get_arc_version() -> str:
    """Best-effort retrieval of Arc version.

    Tries importlib.metadata when installed, otherwise falls back to CLI version
    reported via environment or a static default.
    """
    try:
        import importlib.metadata as im  # type: ignore

        return im.version("arc")
    except Exception:
        # Fall back to a static default; CLI also shows version option.
        return "unknown"


def compose_issue_body(user_text: str, model: str | None = None) -> str:
    """Compose a GitHub-friendly issue body with clear sections.

    - System details are grouped in a fenced block under a header.
    - User message (if provided) is placed in a separate section.
    """
    version_info = f"Arc version: {get_arc_version()}\n"
    python_version = f"Python version: {sys.version.split()[0]}\n"
    platform_info = f"Platform: {platform.platform()}\n"
    model_info = f"Model: {model or 'unknown'}\n"
    python_info = get_python_info() + "\n"
    os_info = get_os_info() + "\n"
    torch_info = get_torch_info() + "\n"
    db_info = get_database_info() + "\n"

    system_info = (
        version_info
        + python_version
        + platform_info
        + model_info
        + python_info
        + os_info
        + torch_info
        + db_info
    )

    sys_section = "### System Information\n\n```\n" + system_info.strip() + "\n```\n"

    user_section = ""
    if user_text and user_text.strip():
        user_section = "\n\n### Description\n\n" + user_text.strip() + "\n"

    return sys_section + user_section


def build_issue_url(title: str | None, body: str) -> str:
    """Construct the GitHub issue URL with prefilled title and body."""
    params = {"body": body}
    params["title"] = title or "Bug report"
    return f"{GITHUB_ISSUES_URL}?{urllib.parse.urlencode(params)}"


def open_in_browser(url: str) -> bool:
    """Attempt to open the given URL in the default web browser."""
    try:
        return bool(webbrowser.open(url))
    except Exception:
        return False
