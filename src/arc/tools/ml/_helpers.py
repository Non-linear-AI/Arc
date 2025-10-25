"""Shared helper functions for ML tool parameter validation and conversion.

This module provides utility functions for converting and validating
parameters passed to ML tools, ensuring type safety and providing
clear error messages.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any


def _as_optional_int(value: Any, field_name: str) -> int | None:
    """Convert value to optional int with validation.

    Args:
        value: Value to convert (can be None, int, or string)
        field_name: Name of field (used in error messages)

    Returns:
        Integer value or None if input is None

    Raises:
        ValueError: If value cannot be converted to int

    Examples:
        >>> _as_optional_int(None, "epochs")
        None
        >>> _as_optional_int(42, "epochs")
        42
        >>> _as_optional_int("42", "epochs")
        42
    """
    if value is None:
        return None
    if isinstance(value, int):
        return value
    try:
        return int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{field_name} must be an integer") from exc


def _as_optional_float(value: Any, field_name: str) -> float | None:
    """Convert value to optional float with validation.

    Args:
        value: Value to convert (can be None, int, float, or string)
        field_name: Name of field (used in error messages)

    Returns:
        Float value or None if input is None

    Raises:
        ValueError: If value cannot be converted to float

    Examples:
        >>> _as_optional_float(None, "learning_rate")
        None
        >>> _as_optional_float(0.01, "learning_rate")
        0.01
        >>> _as_optional_float(42, "learning_rate")
        42.0
    """
    if value is None:
        return None
    if isinstance(value, (float, int)):
        return float(value)
    try:
        return float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{field_name} must be a number") from exc


def _as_string_list(value: Any, field_name: str) -> list[str] | None:
    """Convert value to list of strings with validation.

    Accepts comma-separated strings or sequences and converts them
    to a list of strings, stripping whitespace.

    Args:
        value: Value to convert (can be None, string, or sequence)
        field_name: Name of field (used in error messages)

    Returns:
        List of strings or None if input is None or empty

    Raises:
        ValueError: If value cannot be converted to string list

    Examples:
        >>> _as_string_list(None, "columns")
        None
        >>> _as_string_list("a,b,c", "columns")
        ['a', 'b', 'c']
        >>> _as_string_list(["a", "b", "c"], "columns")
        ['a', 'b', 'c']
        >>> _as_string_list(" a , b , c ", "columns")
        ['a', 'b', 'c']
    """
    if value is None:
        return None
    if isinstance(value, str):
        items = [item.strip() for item in value.split(",") if item.strip()]
        return items or None
    if isinstance(value, Sequence):
        cleaned = [str(item).strip() for item in value if str(item).strip()]
        return cleaned or None
    raise ValueError(f"{field_name} must be an array of strings or comma-separated")
