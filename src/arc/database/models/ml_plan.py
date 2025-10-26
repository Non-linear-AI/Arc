"""ML Plan database model."""

from dataclasses import dataclass
from datetime import datetime


@dataclass
class MLPlan:
    """Represents a machine learning plan in the database.

    An ML plan contains the strategic decisions for building a model,
    including feature engineering, architecture, training, and evaluation.
    """

    plan_id: str  # Unique identifier (e.g., "diabetes-classifier-plan-v1")
    name: str  # User-friendly plan name (e.g., "diabetes-classifier-plan")
    version: int  # Version number for the same base plan
    user_context: str  # Original user intent/description
    source_tables: str  # Comma-separated source table names
    plan_yaml: str  # Full plan as YAML string
    status: str  # 'draft', 'approved', 'implemented'
    created_at: datetime
    updated_at: datetime
