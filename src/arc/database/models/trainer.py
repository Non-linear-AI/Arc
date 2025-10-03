from dataclasses import dataclass
from datetime import datetime


@dataclass
class Trainer:
    """Data class representing a trainer in the Arc system."""

    id: str
    name: str
    version: int
    model_id: str
    model_version: int
    spec: str
    description: str
    created_at: datetime
    updated_at: datetime
