"""
Configuration file for XAI-Compare
"""

from typing import Final

from xailens import strings
from xailens.entities import context

# Constants for artifact filename/structure
MODEL_DATA_DIR_NAME: Final[str] = "model_data"
META_INFO_FILENAME: Final[str] = "meta"
MODEL_FILENAME: Final[str] = "model"


#------------------------
# XAI Method default visualisations & defaults
#------------------------
COEFFS_DEFAULT_VIZ = "barchart"
COEFFS_COMPATIBLE_VIZ = {
        "barchart": context.VizOption(display_name=strings.BARCHART_DISPLAY_NAME),
        "waterfall": context.VizOption(display_name=strings.WATERFALL_DISPLAY_NAME),}


TREE_SHAP_DEFAULT_VIZ = "barchart"
TREE_SHAP_COMPATIBLE_VIZ = {
        "barchart": context.VizOption(display_name=strings.BARCHART_DISPLAY_NAME),
        "waterfall": context.VizOption(display_name=strings.WATERFALL_DISPLAY_NAME),
    }

LLM_REASONING_DEFAULT_VIZ = "barchart"
LLM_REASONING_COMPATIBLE_VIZ = {
        "barchart": context.VizOption(display_name=strings.BARCHART_DISPLAY_NAME),
        "tabular": context.VizOption(display_name=strings.TABULAR_DISPLAY_NAME),
    }

#------------------------
# Visualisation compatibility with XAI methods
#------------------------
BARCHART_COMPATIBLE_XAI = ["tree_shap", "coefficients"]
WATERFALL_COMPATIBLE_XAI = ["tree_shap", "coefficients", "llm_reasoning"]
TABULAR_COMPATIBLE_XAI = ["llm_reasoning"]

#------------------------
# Display config
#------------------------
# maximum no features to show - esp for TF-IDF based models
PLOT_MAX_FEATURES = 25

# how many pixels to add for each row on a plot (so doesn't get squashed)
PLOT_PER_ROW_MULTIPLIER = 25

#------------------------
# Graph/Plot colour scheme
#------------------------

PLOT_LEFT_COLOR = "brown"
PLOT_RIGHT_COLOR = "teal"

PLOT_ZERO_LINE= {"zeroline": True, "zerolinewidth": 2, "zerolinecolor": "black"}