from dataclasses import dataclass
from typing import Optional

@dataclass
class DataSchema:
    instance_id_column: str
    target_column: str
    prediction_column: str
    feature_columns: Optional[list[str]] = None
    class_names: Optional[list] = None
    # for models with very large feature space, set a threshold to ignore any
    # normalised absolute values below this value (eg 0.01)
    # can be set separately for global and local usage
    hide_zero_global_threshold: float | None = None
    hide_zero_local_threshold: float | None = None

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