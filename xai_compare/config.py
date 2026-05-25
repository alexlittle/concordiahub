'''
Configuration file for XAI-Compare
'''

import pandas as pd
import numpy as np
from dataclasses import dataclass
from typing import Any, Final, Optional

# Constants for artifact filename/structure
RUNS_DIR_NAME: Final[str] = "runs"
META_INFO_FILENAME: Final[str] = "meta.json"
MODEL_FILENAME: Final[str] = "model.joblib"


# Configuration & Constants for NLP tasks
EMBEDDING_ENCODER: Final[str] ="all-MiniLM-L6-v2"

@dataclass
class RunData:
    """
    Dataclass structure for RunData to be passed to runner
    """
    x: Optional[pd.DataFrame] = None
    y: Optional[pd.DataFrame] = None
    x_train: Optional[pd.DataFrame] = None
    x_test: Optional[pd.DataFrame] = None
    y_train: Optional[pd.DataFrame] = None
    y_test: Optional[pd.DataFrame] = None
    y_pred: Optional[pd.DataFrame] = None
    y_exp: Optional[pd.DataFrame] = None
    class_names: Optional[list] = None

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