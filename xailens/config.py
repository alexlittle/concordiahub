"""
Configuration file for XAI-Compare
"""

from typing import Final

from xailens import context, strings

# Constants for artifact filename/structure
RUNS_DIR_NAME: Final[str] = "runs"
META_INFO_FILENAME: Final[str] = "meta"
MODEL_FILENAME: Final[str] = "model"


# Configuration & Constants for NLP tasks
EMBEDDING_ENCODER: Final[str] ="all-MiniLM-L6-v2"

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

COR_DEFAULT_VIZ = "barchart"
COR_COMPATIBLE_VIZ = {
        "barchart": context.VizOption(display_name=strings.BARCHART_DISPLAY_NAME),
        "tabular": context.VizOption(display_name=strings.TABULAR_DISPLAY_NAME),
    }

#------------------------
# Visualisation compatibility with XAI methods
#------------------------
BARCHART_COMPATIBLE_XAI = ["tree_shap", "coefficients"]
WATERFALL_COMPATIBLE_XAI = ["tree_shap", "coefficients", "cor"]
TABULAR_COMPATIBLE_XAI = ["cor"]

#------------------------
# Comparison Visualisation compatibility combinations & configuration
#------------------------
DEFAULT_COMPARISON_VISUALISATION = "dumbbell"
DEFAULT_COMPARISON_VISUALISATION_OPTIONS = {
        "diff_bar": context.VizOption(display_name="Difference (bar)", method="viz_diff_bar"),
        "dumbbell": context.VizOption(display_name="Dumbbell", method="viz_dumbbell"),
        "alignment_scale": context.VizOption(display_name="Alignment Scale", method="viz_alignment_scale")
    }

ALIGNMENT_SCALE_COMPARISON_VISUALISATION = "alignment_scale"
ALIGNMENT_SCALE_COMPARISON_VISUALISATION_OPTIONS = {
        "alignment_scale": context.VizOption(display_name="Alignment Scale", method="viz_alignment_scale")
    }
ALIGNMENT_THRESHOLDS = [0.25, 0.5, 0.75, 1.0]
ALIGNMENT_ABSENT_PENALTY = 1.0

#-------------------------
# Filtering/search defaults
#---------------------------
ALL_DATASET_OPTIONS = ("all", "All datasets")
ALL_COMP_OPTIONS = ("all", "All comparisons")


#------------------------
# Graph/Plot colour scheme
#------------------------

PLOT_LEFT_COLOR = "brown"
PLOT_RIGHT_COLOR = "teal"

PLOT_ZERO_LINE= {"zeroline": True, "zerolinewidth": 2, "zerolinecolor": "black"}