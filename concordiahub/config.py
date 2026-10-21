"""
Configuration file for ConcordiaHub
"""

import os

from typing import Final
from dataclasses import dataclass
from pathlib import Path
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor, GradientBoostingClassifier, GradientBoostingRegressor
from xgboost import XGBClassifier, XGBRegressor

from concordiahub import strings


DATA_DIR_ENV_VAR: Final[str] = "CONCORDIAHUB_DATA_DIR"
DEFAULT_DATA_DIR_NAME: Final[str] = "data_store"


def resolve_data_dir() -> str:
    """
    Works out where model data is stored. Uses the CONCORDIAHUB_DATA_DIR
    environment variable if set (the CLI sets this from --data-dir), otherwise
    a data_store folder in the current working directory, so that it sits in
    the root of the project where the package is being used.

    Returns:
        str: absolute path of the data directory
    """
    configured = os.environ.get(DATA_DIR_ENV_VAR)
    if configured:
        return str(Path(configured).expanduser().resolve())
    return str(Path.cwd() / DEFAULT_DATA_DIR_NAME)


# Constants for artifact filename/structure
MODEL_DATA_DIR: Final[str] = resolve_data_dir()
META_INFO_FILENAME: Final[str] = "meta"
MODEL_FILENAME: Final[str] = "model"

DATA_SCHEMA_ID_COLUMN: Final[str] = 'data_schema.instance_id_column'
DATA_SCHEMA_TARGET_COLUMN: Final[str] = 'data_schema.target_column'
DATA_SCHEMA_PREDICTION_COLUMN: Final[str] = 'data_schema.prediction_column'


type TreeModel = (XGBClassifier | XGBRegressor | RandomForestClassifier |
             RandomForestRegressor | GradientBoostingClassifier | GradientBoostingRegressor)

# for now only treemodels supported, when other options are available, add here
type ScikitLearnModel = TreeModel

# Which metric to base the best model on
BEST_MODEL_METRIC = "f1_weighted"


# a cut-off of number of features to keep when generating heatmap, if there are
# 1000's of features (for models using TF-IDF), this cut-off prevents the page
# being very slow to load
KEEP_NUM_FEATURES = 500

@dataclass
class VizOption:
    """
    Helper dataclass for VizOption - with a display name and method
    """
    display_name: str
    method: str | None = None

#------------------------
# XAI Method default visualizations & defaults
# with current adapters all are compatible with each other, but will not always be the case with future XAI method and
# visualization combinations
#------------------------
COEFFS_DEFAULT_VIZ = "barchart"
COEFFS_COMPATIBLE_VIZ = {
        "barchart": VizOption(display_name=strings.BARCHART_DISPLAY_NAME),
        "tabular": VizOption(display_name=strings.TABULAR_DISPLAY_NAME),
}


TREE_SHAP_DEFAULT_VIZ = "barchart"
TREE_SHAP_COMPATIBLE_VIZ = {
        "barchart": VizOption(display_name=strings.BARCHART_DISPLAY_NAME),
        "tabular": VizOption(display_name=strings.TABULAR_DISPLAY_NAME),
    }

LLM_REASONING_DEFAULT_VIZ = "barchart"
LLM_REASONING_COMPATIBLE_VIZ = {
        "barchart": VizOption(display_name=strings.BARCHART_DISPLAY_NAME),
        "tabular": VizOption(display_name=strings.TABULAR_DISPLAY_NAME),
    }

#------------------------
# Visualization compatibility with XAI methods
# as with XAI method, currently identical
#------------------------
BARCHART_COMPATIBLE_XAI = ["tree_shap", "coefficients", "llm_reasoning"]
TABULAR_COMPATIBLE_XAI = ["llm_reasoning", "tree_shap", "coefficients"]

#------------------------
# Display config
#------------------------

PLOTLY_CONFIG = {"displayModeBar": False}

# maximum num features to show - esp for TF-IDF based models
PLOT_MAX_FEATURES = 20

# how many pixels to add for each row on a plot
# so doesn't get squashed and hide axis labels
PLOT_PER_ROW_MULTIPLIER = 25

#------------------------
# Graph/Plot colour scheme
#------------------------

PLOT_SECONDARY_COLOR = "#FE6100"
PLOT_PRIMARY_COLOR = "teal"

PLOT_ZERO_LINE= {"zeroline": True, "zerolinewidth": 2, "zerolinecolor": "black"}

PLOT_UPSET_CORRECT_COLOR = "#4C8C8C"
PLOT_UPSET_INCORRECT_COLOR = "#D97B3F"