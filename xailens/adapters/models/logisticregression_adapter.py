"""
LogisticRegressionAdapter class - implementation of ModelAdapter
"""

from xailens.registry.model_registry import register_model
from xailens.adapters.models.model_base import ModelAdapter
from xailens import strings

@register_model("logistic_regression")
class LogisticRegressionAdapter(ModelAdapter):
    """
    LogisticRegressionAdapter class - implementation of ModelAdapter
    """

    display_name = strings.LOGISTIC_REGRESSION_DISPLAY_NAME
