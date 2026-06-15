
# ----------------
# App/page layout
# ----------------
APP_NAME = "XAI Lens"

# sidebar
SIDEBAR_FILTER_TITLE = "Quick Filter:"

# homepage
HOMEPAGE_TITLE = "Comparison Browser"
HOMEPAGE_HELPER = "Select a comparison to explore, or use the filters to narrow down by dataset and/or model type."
BTN_GO_TO_COMPARISON = "Go to comparison &#10132;"
OVERVIEW_TITLE = "Comparison Overview"
DETAIL_TITLE = "Comparison Detail"

# Filtering/search defaults
ALL_DATASET_OPTIONS = ("all", "All datasets")
ALL_COMP_OPTIONS = ("all", "All model types")


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
XAI_METHOD_DISPLAY_NAME = DisplayNameDict({
    "coefficients": "Coefficients",
    "tree_shap": "Tree SHAP",
    "cor": "Chain of Reasoning",
})

COEFFICIENTS_DISPLAY_NAME = "Coefficients"
COR_DISPLAY_NAME = "Chain of Reasoning"
KERNEL_SHAP_DISPLAY_NAME = "Kernel Shap"
TREE_SHAP_DISPLAY_NAME = "Tree Shap"

COEFFICIENTS_NOTES = """
    Coefficients are considered inherently interpretable for linear models, meaning the values directly reflect each 
    feature's contribution to the model's decision.
    """
COR_NOTES = """
    Chain of Reasoning explanations may be post-rationalised by the model, and so may not 
    truly reflect the decision process the model used to arrive at the explanation.
    """
TREE_SHAP_NOTES = """
    SHAP values estimate each feature's contribution to a prediction relative to a baseline, but when features are 
    correlated, the model may distribute credit across them in ways that don't clearly correspond to which feature 
    is truly responsible.
    """

# ----------------
# MODELS
# ----------------
LLM_DISPLAY_NAME = "LLM"
LOGISTIC_REGRESSION_DISPLAY_NAME = "Logistic Regression"
XGBOOST_DISPLAY_NAME = "XGBoost"

# ----------------
# VISUALISATIONS
# ----------------
# Bar Chart Visualisation
BARCHART_DISPLAY_NAME = "Bar Chart"
BARCHART_TITLE = "Feature Contributions (normalised)"
BARCHART_X_AXIS = "Normalised contribution"

# Waterfall Visualisation
WATERFALL_DISPLAY_NAME = "Waterfall"
WATERFALL_TITLE = "Feature Contributions"
WATERFALL_X_AXIS = "Contribution"

# Tabular Visualisation
TABULAR_DISPLAY_NAME = "Tabular (text)"