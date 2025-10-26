"""Shared utilities for ML tools."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any


def _as_optional_int(value: Any, field_name: str) -> int | None:
    if value is None:
        return None
    if isinstance(value, int):
        return value
    try:
        return int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{field_name} must be an integer") from exc


def _as_optional_float(value: Any, field_name: str) -> float | None:
    if value is None:
        return None
    if isinstance(value, (float, int)):
        return float(value)
    try:
        return float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{field_name} must be a number") from exc


def _as_string_list(value: Any, field_name: str) -> list[str] | None:
    if value is None:
        return None
    if isinstance(value, str):
        items = [item.strip() for item in value.split(",") if item.strip()]
        return items or None
    if isinstance(value, Sequence):
        cleaned = [str(item).strip() for item in value if str(item).strip()]
        return cleaned or None
    raise ValueError(f"{field_name} must be an array of strings or comma-separated")


def _load_ml_plan(services, plan_id: str) -> tuple[dict, Any] | tuple[None, str]:
    """Load ML plan from database.

    Args:
        services: ServiceContainer with ml_plans service
        plan_id: ML plan ID to load

    Returns:
        Tuple of (ml_plan_dict, MLPlan_object) on success,
        or (None, error_message) on failure
    """
    try:
        from arc.core.ml_plan import MLPlan

        ml_plan = services.ml_plans.get_plan_content(plan_id)
        plan = MLPlan.from_dict(ml_plan)
        return (ml_plan, plan)
    except ValueError as e:
        return (None, f"Failed to load ML plan '{plan_id}': {e}")
    except Exception as e:
        return (None, f"Unexpected error loading ML plan '{plan_id}': {e}")
