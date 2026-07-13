"""
Dataclasses for DataSchema and ResponseSchema
"""

from dataclasses import dataclass
from typing import Optional

@dataclass
class DataSchema:
    """
    DataSchema - defines format of the data provided by the model developer
    """
    instance_id_column: str
    target_column: str
    prediction_column: str
    feature_columns: Optional[list[str]] = None
    class_names: Optional[list] = None

@dataclass
class ResponseSchema:
    """
    ResponseSchema - defines format of an LLM response
    """
    contributions_field: str
    contribution_key: str | None
