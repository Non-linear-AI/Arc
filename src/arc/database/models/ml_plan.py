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
    version: int  # Version number for the same base plan
    user_context: str  # Original user intent/description
    data_table: str  # Source data table
    target_column: str | None  # Prediction target column
    plan_yaml: str  # Full plan as YAML string
    status: str  # 'draft', 'approved', 'implemented'
    created_at: datetime
    updated_at: datetime
