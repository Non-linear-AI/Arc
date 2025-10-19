from dataclasses import dataclass
from datetime import datetime


@dataclass
class Evaluator:
    """Data class representing an evaluator in the Arc system."""

    id: str
    name: str
    version: int
    trainer_id: str
    trainer_version: int
    spec: str
    description: str
    plan_id: str | None = None
    created_at: datetime | None = None
    updated_at: datetime | None = None
