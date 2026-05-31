from xai_compare.registry.model_registry import register_model
from xai_compare.adapters.models.model_base import ModelAdapter
from xai_compare import strings

@register_model("logistic_regression")
class LogisticRegressionAdapter(ModelAdapter):

    display_name = strings.LOGISTIC_REGRESSION_DISPLAY_NAME
