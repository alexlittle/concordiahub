import shap
import pandas as pd

from xai_compare.exceptions import MissingDataError
from xai_compare.config import RunData
from xai_compare.registry.xai_registry import register_xai
from xai_compare.adapters.explainers.xaimethod_base import XAIMethodAdapter



@register_xai("shap_tree")
class TreeSHAPAdapter(XAIMethodAdapter):
    """Tree Shap Explainer"""

    def explain_local(self, model, data: RunData ) -> pd.DataFrame:
        """Computes the local explanation from the coefficients of the model
        Args:
            model: Trained model object
            data: RunData object

        Returns:
            pd.DataFrame: dataframe containing local explanations

        Raises:
            MissingDataError: if data.x_test is not provided
        """
        if data.x_test is None:
            raise MissingDataError("CoefficientsAdapter requires x_test in RunData")
        explainer = shap.TreeExplainer(model)
        return explainer.shap_values(data.x_test)