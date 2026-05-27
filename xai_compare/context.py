'''
Context dataclasses for XAI-Compare
'''



import pandas as pd
from dataclasses import dataclass
from typing import Any, Optional

@dataclass
class DataSchema:
    sample_id_column: str
    target_column: str
    feature_columns: Optional[list[str]] = None
    class_names: Optional[list] = None


@dataclass
class RunData:
    """
    Dataclass structure for RunData to be passed to runner
    """
    data_schema: DataSchema
    x: Optional[pd.DataFrame] = None
    y: Optional[pd.DataFrame] = None
    x_train: Optional[pd.DataFrame] = None
    x_test: Optional[pd.DataFrame] = None
    y_train: Optional[pd.DataFrame] = None
    y_test: Optional[pd.DataFrame] = None
    y_pred: Optional[pd.DataFrame] = None
    y_exp: Optional[pd.DataFrame] = None


@dataclass
class RunContext:
    """
    Dataclass structure for RunContext to be passed to runner
    """
    model_type: str
    xai_methods: list[str]
    dataset_id: str
    data: RunData
    model: Optional[Any] = None
    model_subtype: Optional[str] = None
    run_name: Optional[str] = None
    model_id: Optional[str] = None