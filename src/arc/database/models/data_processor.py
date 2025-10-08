from dataclasses import dataclass
from datetime import datetime


@dataclass
class DataProcessor:
    """Data class representing a data processor in the Arc system."""

    id: str
    name: str
    version: int
    spec: str
    description: str
    created_at: datetime
    updated_at: datetime
