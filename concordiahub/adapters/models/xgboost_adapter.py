"""
XGBoostAdapter class - implementation of ModelAdapter
"""


from concordiahub.registry.model_registry import register_model
from concordiahub.adapters.models.model_base import ModelAdapter
from concordiahub import strings

@register_model("xgboost")
class XGBoostAdapter(ModelAdapter):
    """
    XGBoostAdapter class - implementation of ModelAdapter
    """
    
    display_name = strings.XGBOOST_DISPLAY_NAME
