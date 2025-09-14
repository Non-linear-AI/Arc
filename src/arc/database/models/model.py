from dataclasses import dataclass
from datetime import datetime


@dataclass
class Model:
    """Data class representing a model in the Arc system.

    Mirrors the C++ Model struct with exact field mapping.
    """

    id: str
    type: str
    name: str
    version: int
    description: str
    base_model_id: str | None
    spec: str
    arc_graph: str
    created_at: datetime
    updated_at: datetime
