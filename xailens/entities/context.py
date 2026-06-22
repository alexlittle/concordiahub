'''
Context dataclasses for XAI-Compare
'''

import pandas as pd
from dataclasses import dataclass
from typing import Any, Optional

from xailens.entities.modeleval import ModelEval
from xailens.schema import DataSchema, ResponseSchema


@dataclass
class ModelMetrics:
    accuracy: Optional[float] = None
    f1_macro: Optional[float] = None
    f1_weighted: Optional[float] = None
    n_instances: Optional[int] = None
    n_failed: Optional[int] = None


@dataclass
class ModelData:
    """
    Dataclass structure for ModelData to be passed to runner
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
    metrics: Optional[ModelMetrics] = None


@dataclass
class ComparisonContext:
    """
    Dataclass structure for passing data to comparison visualisations
    """
    modeleval_a: ModelEval
    modeleval_b: ModelEval
    explanation_a: pd.Series
    explanation_b: pd.Series
    axis_range: tuple | None = None


@dataclass
class VizOption:
    display_name: str
    method: str | None = None