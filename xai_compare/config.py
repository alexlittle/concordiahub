import pandas as pd
import numpy as np
from dataclasses import dataclass
from typing import Any

EMBEDDING_ENCODER ="all-MiniLM-L6-v2"


@dataclass
class RunContext:
    # core model/method/identifiers
    model_type: str
    xai_methods: list[str]
    dataset_id: str

    # optional
    X_train: pd.DataFrame
    X_test: pd.DataFrame
    y_train: pd.Series
    y_test: pd.Series
    y_pred: np.ndarray


    model: Any = None
    run_id: str = None
    model_id: str = None
    y_proba: np.ndarray = None