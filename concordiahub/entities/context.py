"""
Context dataclasses for XAI-Compare
"""

import pandas as pd
from dataclasses import dataclass
from typing import Any, Optional


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
class ModelData:
    """
    Dataclass structure for ModelData to be passed to runner
    """
    data_schema: DataSchema
    raw_instances: pd.DataFrame
    x_test: pd.DataFrame
    y_test: pd.DataFrame
    y_pred: pd.DataFrame
    y_exp: Optional[pd.DataFrame] = None

@dataclass
class ResponseSchema:
    """
    ResponseSchema - defines format of an LLM response
    """
    contributions_field: str
    contribution_key: str | None

@dataclass
class ModelContext:
    """
    Dataclass structure for ModelContext to be passed to runner
    """
    model_type: str
    xai_methods: list[str]
    dataset_id: str
    data: ModelData
    model_display_name: str
    model: Optional[Any] = None
    model_subtype: Optional[str] = None
    model_dir_name: Optional[str] = None
    response_schema: Optional[ResponseSchema] = None




