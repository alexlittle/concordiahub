import shap
import pandas as pd

from xai_compare.config import RunData
from xai_compare.registry.xai_registry import register_xai
from xai_compare.adapters.explainers.xaimethod_base import XAIMethodAdapter


@register_xai("shap_kernel")
class KernalSHAPAdapter(XAIMethodAdapter):

    def explain(self, model, data: RunData) -> pd.DataFrame:
        explainer = shap.KernelExplainer(model.predict_proba, data.x_train)
        return explainer.shap_values(data.x_test)