"""
Configuration file for XAI-Compare
"""

from typing import Final

# Constants for artifact filename/structure
RUNS_DIR_NAME: Final[str] = "runs"
META_INFO_FILENAME: Final[str] = "meta"
MODEL_FILENAME: Final[str] = "model"


# Configuration & Constants for NLP tasks
EMBEDDING_ENCODER: Final[str] ="all-MiniLM-L6-v2"


# XAI Method/Visualisation compatability & defaults
COEFFICIENTS_DEFAULT_VIZ = "forceplot"
COEFFICIENTS_COMPATIBLE_VIZ = ["waterfall", "forceplot"]

COR_DEFAULT_VIZ = None
COR_COMPATIBLE_VIZ = []

KERNEL_SHAP_DEFAULT_VIZ = None
KERNEL_SHAP_COMPATIBLE_VIZ = []

TREE_SHAP_DEFAULT_VIZ = "forceplot"
TREE_SHAP_COMPATIBLE_VIZ = ["waterfall", "forceplot", "beeswarm"]


FORCE_PLOT_COMPATIBLE_XAI = ["shap_tree", "coefficients"]
WATERFALL_COMPATIBLE_XAI = ["shap_tree","coefficients"]


# Graph/Plot colour scheme
PLOT_LEFT_COLOR = "brown"
PLOT_RIGHT_COLOR = "teal"

PLOT_ZERO_LINE= {"zeroline": True, "zerolinewidth": 2, "zerolinecolor": "black"}