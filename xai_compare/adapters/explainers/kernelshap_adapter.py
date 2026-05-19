import shap
import pandas as pd

from xai_compare.exceptions import MissingDataError
from xai_compare.config import RunData
from xai_compare.registry.xai_registry import register_xai
from xai_compare.adapters.explainers.xaimethod_base import XAIMethodAdapter


@register_xai("shap_kernel")
class KernalSHAPAdapter(XAIMethodAdapter):
    """Model-agnostic Kernel Shap Explainer"""

    def explain_local(self, data: RunData, model=None) -> pd.DataFrame:
        """Computes the local explanation using KernelShap method
        Args:
            model: Trained model object
            data: RunData object

        Returns:
            pd.DataFrame: dataframe containing local explanations

        Raises:
            MissingDataError: if x_train or x_test is not provided
        """
        if data.x_train is None or data.x_test is None:
            raise MissingDataError("CoefficientsAdapter requires x_train and x_test in RunData")
        explainer = shap.KernelExplainer(model.predict_proba, data.x_train)
        return explainer.shap_values(data.x_test)