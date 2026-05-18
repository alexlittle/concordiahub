import shap
import pandas as pd
from xai_compare.registry.xai_registry import register_xai
from xai_compare.adapters.explainers.xaimethod_base import XAIMethodAdapter



@register_xai("shap_tree")
class TreeSHAPAdapter(XAIMethodAdapter):


    def explain(self, model, X_train, X_test) -> pd.DataFrame:
        explainer = shap.TreeExplainer(model)
        return explainer.shap_values(X_test)