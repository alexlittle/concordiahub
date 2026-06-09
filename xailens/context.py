'''
Context dataclasses for XAI-Compare
'''

import pandas as pd
from dataclasses import dataclass, field
from typing import Any, Optional

from xailens import constants
from xailens.services.preprocess import normalise_by_sum
from xailens.run import Run

@dataclass
class DataSchema:
    sample_id_column: str
    target_column: str
    prediction_column: str
    feature_columns: Optional[list[str]] = None
    class_names: Optional[list] = None

@dataclass
class ResponseSchema:
    contributions_field: str
    contribution_key: str | None

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
    response_schema: Optional[ResponseSchema] = None


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
        self.features_a = normalise_by_sum(self.explanation_a, self.run_a.get(constants.DATA_SCHEMA_ID_COLUMN))
        self.features_b = normalise_by_sum(self.explanation_b, self.run_b.get(constants.DATA_SCHEMA_ID_COLUMN))


@dataclass
class VizOption:
    display_name: str
    method: str | None = None