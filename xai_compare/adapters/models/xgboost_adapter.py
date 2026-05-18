from xai_compare.registry.model_registry import register_model
from xai_compare.adapters.models.model_base import ModelAdapter


@register_model("xgboost")
class XGBoostAdapter(ModelAdapter):

    def prepare(self):
        return self.model