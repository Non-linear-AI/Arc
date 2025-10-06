from dataclasses import dataclass
from datetime import datetime


@dataclass
class Model:
    """Data class representing a model in the Arc system."""

    id: str
    type: str
    name: str
    version: int
    description: str
    spec: str
    created_at: datetime
    updated_at: datetime
    plan_id: str | None = None
