"""
Configuration file for XAI-Compare
"""

from typing import Final

from xailens import strings
from xailens.viz_option import VizOption

# Constants for artifact filename/structure
MODEL_DATA_DIR_NAME: Final[str] = "model_data"
META_INFO_FILENAME: Final[str] = "meta"
MODEL_FILENAME: Final[str] = "model"

DATA_SCHEMA_ID_COLUMN: Final[str] = 'data_schema.instance_id_column'
DATA_SCHEMA_TARGET_COLUMN: Final[str] = 'data_schema.target_column'
DATA_SCHEMA_PREDICTION_COLUMN: Final[str] = 'data_schema.prediction_column'

# Which metric to base the best model on
BEST_MODEL_METRIC = "f1_weighted"


# a cut-off of number of features to keep when generating heatmap, if there are
# 1000's of features (for models using TF-IDF), this cut-off prevents the page
# being very slow to load
KEEP_NUM_FEATURES = 500


#------------------------
# XAI Method default visualisations & defaults
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
# Visualisation compatibility with XAI methods
#------------------------
BARCHART_COMPATIBLE_XAI = ["tree_shap", "coefficients"]
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