'''
Configuration file for XAI-Compare
'''

import pandas as pd
import numpy as np
from dataclasses import dataclass
from typing import Any, Final

# Constants for artifact filename/structure
RUNS_DIR_NAME: Final[str] = "runs"
META_INFO_FILENAME: Final[str] = "meta.json"
MODEL_FILENAME: Final[str] = "model.joblib"


# Configuration & Constants for NLP tasks
EMBEDDING_ENCODER: Final[str] ="all-MiniLM-L6-v2"


@dataclass
class RunContext:
    '''
    Context definition for runner parameters
    '''
    # core model/method/identifiers
    model_type: str
    xai_methods: list[str]
    dataset_id: str

    # optional
    X_train: pd.DataFrame = None
    X_test: pd.DataFrame = None
    y_train: pd.Series = None
    y_test: pd.Series = None
    y_pred: np.ndarray = None

    model: Any = None
    run_id: str = None
    model_id: str = None
    y_proba: np.ndarray = None