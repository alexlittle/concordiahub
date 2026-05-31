from xai_compare.registry.model_registry import register_model
from xai_compare.adapters.models.model_base import ModelAdapter
from xai_compare import strings

@register_model("xgboost")
class XGBoostAdapter(ModelAdapter):
    display_name = strings.XGBOOST_DISPLAY_NAME
    pass