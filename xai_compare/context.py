'''
Context dataclasses for XAI-Compare
'''

import pandas as pd
from dataclasses import dataclass, field
from typing import Any, Optional

from xai_compare import constants
from xai_compare.services.preprocess import normalise_values
from xai_compare.run import Run

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


@dataclass
class ComparisonContext:
    """
    Dataclass structure for passing data to comparison visualisations
    """
    run_a: Run
    run_b: Run
    explanation_a: pd.Series
    explanation_b: pd.Series
    axis_range: tuple | None = None

    features_a: pd.Series = field(init=False)
    features_b: pd.Series = field(init=False)

    def __post_init__(self):
        self.features_a = normalise_values(self.explanation_a, self.run_a.get(constants.DATA_SCHEMA_ID_COLUMN))
        self.features_b = normalise_values(self.explanation_b, self.run_b.get(constants.DATA_SCHEMA_ID_COLUMN))