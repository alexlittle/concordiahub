"""
LogisticRegressionAdapter class - implementation of ModelAdapter
"""

from concordiahub.registry.model_registry import register_model
from concordiahub.adapters.models.model_base import ModelAdapter
from concordiahub import strings

@register_model("logistic_regression")
class LogisticRegressionAdapter(ModelAdapter):
    """
    LogisticRegressionAdapter class - implementation of ModelAdapter
    """

    display_name = strings.LOGISTIC_REGRESSION_DISPLAY_NAME
