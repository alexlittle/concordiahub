from dataclasses import dataclass
from typing import Optional

@dataclass
class DataSchema:
    sample_id_column: str
    target_column: str
    prediction_column: str
    feature_columns: Optional[list[str]] = None
    class_names: Optional[list] = None
    # for models with very large feature space, set a threshold to ignore any
    # normalised absolute values below this value (eg 0.01)
    hide_zero_threshold: float | None = None

@dataclass
class ResponseSchema:
    contributions_field: str
    contribution_key: str | None