"""
Strings used throughout the framework, including dashboard.
These are all user facing, so having in one place makes it easier to update or
to translate to alternative languages
"""

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

# common across pages
#------------------------
CLICK_TO_VIEW = "Click to view"

# Page titles
#-----------------
HOMEPAGE_TITLE = "Select a dataset"
DATASET_TITLE = "Model Comparison for {dataset} Dataset"
INSTANCE_TITLE = "Instance Comparison for {dataset} Dataset"
INSTANCE_DETAIL_TITLE = "Instance Detail"

# Homepage elements
#-------------------------------
HOME_DATASET_NUM_MODELS = "{count} models"

# Dataset page elements
#-------------------------------

# tabs
DATASET_TAB_OVERVIEW = "Overview"
DATASET_TAB_MORE_METRICS = "More Metrics"
DATASET_TAB_INSTANCE_CONSENSUS = "Instance Consensus"
DATASET_TABS = [DATASET_TAB_OVERVIEW, DATASET_TAB_MORE_METRICS, DATASET_TAB_INSTANCE_CONSENSUS]

# section titles
RECOMMENDED_MODEL_TITLE = "Recommended Model"
RECOMMENDED_MODELS_TITLE = "Recommended Models"

# functions to help display correctly pluralised text

def recommended_title(count: int) -> str:
    """
    Returns the correct pluralised title

    Args:
        count (int): how many models are recommended

    Returns:
        str: the recommended pluralised title
    """
    return RECOMMENDED_MODEL_TITLE if count == 1 else RECOMMENDED_MODELS_TITLE

def format_recommended_models(recommended_models: list[str]) -> str:
    """
    Format the model names for display.

    Args:
        recommended_models (list[str]): list of the model names

    Returns:
        str: joined list of model names for display
    """
    if len(recommended_models) == 1:
        return recommended_models[0]
    return ', '.join(recommended_models)


PERF_F1_WEIGHTED_TITLE = "F1 Weighted Score"
PERF_METRICS_TITLE = "Accuracy, Precision & Recall Metrics"
GLOBAL_EXPLANATION_TITLE = "Global Feature Importance Heatmap"
INSTANCE_CONSENSUS_TITLE = "Instance Consensus"

# performance metrics graphs
METRICS_NAMES = {
    'accuracy': 'Accuracy',
    'f1_weighted': "F1 weighted",
    'precision_weighted': "Precision weighted",
    'recall_weighted': "Recall weighted",
}

METRICS_EXPLANATIONS = {
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

MISCLASSIFIED = "Misclassifications"

INSTANCE_GROUPS = {
    "all": "All Instances",
    "allcorrect": "All Correct Instances",
    "allincorrect": "All Incorrect Instances",
    "disagree": "Models Disagree"
}

MISCLASSIFIED_INSTANCES_TITLE = "Misclassified by: {models}"

DISPLAY_NAMES = {
    "instance_uuid": "Instance ID",
    "true_label": "True Value",
    "view_detail": "View full details"
}


# Instance detail page
#-------------------------------

# tabs
INSTANCE_DETAIL_TAB_OVERVIEW = "Instance Overview"
INSTANCE_DETAIL_TAB_EXPLANATIONS = "Model Explanations"
INSTANCE_DETAIL_TABS = [INSTANCE_DETAIL_TAB_OVERVIEW, INSTANCE_DETAIL_TAB_EXPLANATIONS]

INSTANCE_DETAIL_PREDICTIONS_TITLE = "Model Predictions"
INSTANCE_DETAIL_HEATMAP_TITLE = "Local Feature Importance Heatmap"
INSTANCE_DETAIL_DATA_TITLE = "Instance Data"
INSTANCE_DETAIL_MODEL_EXPLANATIONS = "Model Explanations"

# ----------------
# Error/Warning messages
# ----------------

ERROR_MISSING_MODELDATA = "{method} requires {data} in ModelData"
ERROR_INSTANCE_NOT_FOUND = "Instance '{instance_id}' not found in explanations for {key}"

WARNING_GLOBAL_NOT_AVAILABLE = "Global explanations are not available for {adapter}"
WARNING_INVALID_URL_PARAMS = "Invalid combination of url parameters"

# ----------------
# MODEL_TYPES
# ----------------
class DisplayNameDict(dict):
    """
    Extends the dict class, to manage missing keys
    """
    def __missing__(self, key: str) -> str:
        """
        Defines what to return is a key is missing from the dict
        
        Args:
            key (str):

        Returns:
            str: 
        """
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
# VISUALIZATIONS
# ----------------
# Bar Chart Visualization
BARCHART_DISPLAY_NAME = "Bar Chart"
BARCHART_TITLE = "Feature Contributions (normalized)"
BARCHART_X_AXIS = "Normalized contribution"


# Tabular Visualization
TABULAR_DISPLAY_NAME = "Tabular (text)"
TABULAR_FEATURE_TITLE = "Feature"
TABULAR_SCORE_TITLE = "Normalized<br/>Score (1 to -1)"
TABULAR_OVERALL_REASON_TITLE = "Overall Reasoning"
TABULAR_OVERALL_CONF_TITLE= "Overall confidence"


# Heatmaps
HEATMAP_COLOUR_BAR_TITLE = "Importance"
HEATMAP_HOVER_TEMPLATE = "<b>%{x}</b><br>%{y}<br>Importance: %{z:.4f}<extra></extra>"
HEATMAP_HOVER_MISSING_TEMPLATE = "<b>%{x}</b><br>%{y}<br>No data<extra></extra>"
HEATMAP_NO_DATA = "No data"

# Matrix set plot
MATRIX_SET_PLOT_TITLE = "Sets of Misclassified Instances"

# ----------------
# HELP TEXT
# ----------------

HELP_RECOMMENDED_MODEL = """
    The recommended model is the model with highest F1 weighted score,
    out of the models specified in the sidebar"""

HELP_METRICS = {
    'accuracy': "Proportion of all predictions the model got right.",
    'f1_weighted': "Harmonic mean of precision and recall, averaged across classes weighted by class size.",
    'precision_weighted': "Of the predictions made for each class, the proportion that were correct, averaged and weighted by class size.",
    'recall_weighted': "Of the actual instances of each class, the proportion the model correctly identified, averaged and weighted by class size.",
}

HELP_GLOBAL_EXPLANATION = """
    The normalized importance each model assigned to each feature in the dataset. Ordered by the average across all models in descending order.
"""

HELP_INSTANCES = """
    Shows where models agree and disagree on individual predictions.
"""

