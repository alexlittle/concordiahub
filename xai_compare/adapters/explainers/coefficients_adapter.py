
import pandas as pd

from xai_compare.config import RunData
from xai_compare.registry.xai_registry import register_xai
from xai_compare.adapters.explainers.xaimethod_base import XAIMethodAdapter



@register_xai("coefficients")
class CoefficientsAdapter(XAIMethodAdapter):



    def explain(self, model, data: RunData) -> pd.DataFrame:
        coef = model.coef_[0]
        local_explanations = data.x_test * coef
        return local_explanations