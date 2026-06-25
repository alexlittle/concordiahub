# ---------------------
# Runner messages
# ---------------------
RUN_STARTING = "XAI Lens starting..."
RUN_COMPUTING_METRICS = "Computing metrics"
RUN_SAVING_METADATA = "Saving model metadata"
RUN_SAVING_MODEL_DATA = "Saving model data"
RUN_RUNNING_XAI_METHODS = "Processing XAI methods"
RUN_GLOBAL_EXPLANATIONS = "Processing global explanations for {xai_method}"
RUN_LOCAL_EXPLANATIONS = "Processing local explanations for {xai_method}"
RUN_COMPLETE = """
XAI Lens run completed, all the data and explanations have now been stored.    
You can now run the dashboard with `xailens-dash` to view the results.
"""



# ----------------
# App/page layout
# ----------------
APP_NAME = "XAI Lens"

# sidebar
SIDEBAR_FILTER_TITLE = "Quick Filter:"

# Page titles
HOMEPAGE_TITLE = "Select a dataset"
DATASET_TITLE = "Dataset Comparison"
INSTANCE_TITLE = "Instance Comparison"
INSTANCE_DETAIL_TITLE = "Instance Detail"

# Homepage elements
HOME_DATASET_NUM_MODELS = "{count} models"

# Unused?
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
ERR_NO_MODELS = "No model_data selected"
ERR_NOT_2_MODELS = "You need exactly 2 models to compare"

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