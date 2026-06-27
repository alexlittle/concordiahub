from dataclasses import dataclass
from typing import Optional

@dataclass
class DataSchema:
    instance_id_column: str
    target_column: str
    prediction_column: str
    feature_columns: Optional[list[str]] = None
    class_names: Optional[list] = None

@dataclass
class ResponseSchema:
    contributions_field: str
    contribution_key: str | None


@dataclass
class AlignmentLevel:
    level: int
    label: str
    colour: str
    description: str

ALIGNMENT_LEVELS = [
    AlignmentLevel(1, "Fully Conflicting", "#d32f2f", "..."),
    AlignmentLevel(2, "Mostly Conflicting", "#ef6c00", "..."),
    AlignmentLevel(3, "Mixed", "#f9a825", "..."),
    AlignmentLevel(4, "Mostly Agreeing", "#7cb342",  "..."),
    AlignmentLevel(5, "Fully Agreeing", "#2e7d32", "..."),
]