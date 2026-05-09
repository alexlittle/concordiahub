import pandas as pd
import numpy as np
from dataclasses import dataclass
from typing import Any

EMBEDDING_ENCODER ="all-MiniLM-L6-v2"


@dataclass
class RunContext:
    # identity
    model_type: str
    xai_method: str
    dataset_id: str

    # data
    X_train: pd.DataFrame
    X_test: pd.DataFrame
    y_train: pd.Series
    y_test: pd.Series
    y_pred: np.ndarray

    # optional
    model: Any = None
    run_id: str = None
    model_id: str = None
    y_proba: np.ndarray = None