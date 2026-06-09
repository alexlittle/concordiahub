from xailens.registry.model_registry import register_model
from xailens.adapters.models.model_base import ModelAdapter
from xailens import strings

@register_model("xgboost")
class XGBoostAdapter(ModelAdapter):

    display_name = strings.XGBOOST_DISPLAY_NAME
