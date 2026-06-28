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
