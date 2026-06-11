
# ----------------
# App/page layout
# ----------------
APP_NAME = "XAI Lens"

# sidebar
SIDEBAR_FILTER_TITLE = "Quick Filter:"

# homepage
HOMEPAGE_TITLE = "Comparison browser"
HOMEPAGE_HELPER = "Select a comparison to explore, or use the filters to narrow down."
BTN_GO_TO_COMPARISON = "Go to comparison &#10132;"



# ----------------
# Error/Warning messages
# ----------------
ERR_NO_RUNS = "No runs selected"
ERR_NOT_2_RUNS = "You need exactly 2 runs to compare"

# ----------------
# MODEL_TYPES
# ----------------
class DisplayNameDict(dict):
    def __missing__(self, key: str) -> str:
        return key.replace("_", " ").title()

MODEL_DISPLAY_NAME = DisplayNameDict({
    "logistic_regression": "Logistic Regression",
    "xgboost": "XGBoost",
    "llm": "LLM",
})

# ----------------
# XAI METHODS
# ----------------
COEFFICIENTS_DISPLAY_NAME = "Coefficients"
COR_DISPLAY_NAME = "Chain of Reasoning"
KERNEL_SHAP_DISPLAY_NAME = "Kernel Shap"
TREE_SHAP_DISPLAY_NAME = "Tree Shap"

# ----------------
# MODELS
# ----------------
LLM_DISPLAY_NAME = "LLM"
LOGISTIC_REGRESSION_DISPLAY_NAME = "Logistic Regression"
XGBOOST_DISPLAY_NAME = "XGBoost"

# ----------------
# VISUALISATIONS
# ----------------
# Force Plot Visualisation
FORCE_PLOT_DISPLAY_NAME = "Force Plot"
FORCE_PLOT_TITLE = "Feature Contributions (normalised)"
FORCE_PLOT_X_AXIS = "Normalised contribution"

# Waterfall Visualisation
WATERFALL_DISPLAY_NAME = "Waterfall"
WATERFALL_TITLE = "Feature Contributions"
WATERFALL_X_AXIS = "Contribution"

# Tabular Visualisation
TABULAR_DISPLAY_NAME = "Tabular (text)"