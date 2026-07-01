
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
#------------------------
SIDEBAR_MODEL_FILTER_TITLE = "Include models:"
SIDEBAR_INSTANCE_GROUP_FILTER_TITLE = "Instances:"

# Page titles
HOMEPAGE_TITLE = "Select a dataset"
DATASET_TITLE = "Model Comparison for {dataset} Dataset"
INSTANCE_TITLE = "Instance Comparison for {dataset} Dataset"
INSTANCE_DETAIL_TITLE = "Instance Detail"

# Homepage elements
#-------------------------------
HOME_DATASET_NUM_MODELS = "{count} models"

# Dataset page elements
#-------------------------------



# section titles
RECOMMENDED_MODEL_TITLE = "Recommended Model"
RECOMMENDED_MODELS_TITLE = "Recommended Models"
RECOMMENDED_MODEL_EXPLANATION = "Based on the highest {metric} score"

PERF_FI_WEIGHTED_TITLE = "F1 weighted"
PERF_METRICS_TITLE = "Accuracy, Precision & Recall Metrics"
GLOBAL_EXPLANATION_TITLE = "Global Feature Importance Heatmap"
INSTANCE_CONCENSUS_TITLE = "Instance Consensus"

# performance metrics graphs
METRICS_NAMES = {
    'accuracy': 'Accuracy',
    'f1_weighted': "F1 weighted",
    'precision_weighted': "Precision weighted",
    'recall_weighted': "Recall weighted",
}


# instance consensus metrics
INSTANCES_TOTAL_TITLE = "Total Instances"
INSTANCES_ALL_CORRECT_TITLE = "All Models Correct"
INSTANCES_ALL_INCORRECT_TITLE = "All Models Incorrect"
INSTANCES_DISAGREE_TITLE = "Models Disagree"

# Instance groups page
#-------------------------------

INSTANCE_GROUPS = {
    "all": "All Instances",
    "allcorrect": "All Correct Instances",
    "allincorrect": "All Incorrect Instances",
    "disagree": "Models Disagree"
}

DISPLAY_NAMES = {
    "instance_uuid": "Instance ID",
    "true_label": "True Value",
}


# Instance detail page
#-------------------------------
INSTANCE_DETAIL_PREDICTIONS_TITLE = "Model Predictions"
INSTANCE_DETAIL_HEATMAP_TITLE = "Local Feature Importance Heatmap"
INSTANCE_DETAIL_DATA_TITLE = "Instance Data"
INSTANCE_DETAIL_MODEL_EXPLANATIONS = "Model Explanations"

# ----------------
# Error/Warning messages
# ----------------

WARNING_GLOBAL_NOT_AVAILABLE = "Global explanations are not available for {adapter}"

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
    "llm_reasoning": "LLM Reasoning",
})

COEFFICIENTS_DISPLAY_NAME = "Coefficients"
LLM_REASONING_DISPLAY_NAME = "LLM Reasoning"
TREE_SHAP_DISPLAY_NAME = "Tree Shap"

COEFFICIENTS_NOTES = """
    Coefficients are considered inherently interpretable for linear models, meaning the values directly reflect each 
    feature's contribution to the model's decision.
    """
LLM_REASONING_NOTES = """
    LLM Reasoning explanations may be post-rationalised by the model, and so may not 
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
TABULAR_SCORE_TITLE = "Score (1 to -1)"