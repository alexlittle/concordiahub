import shap
import pandas as pd

from xai_compare.exceptions import MissingDataError
from xai_compare.context import RunData
from xai_compare.registry.xai_registry import register_xai
from xai_compare.adapters.explainers.xaimethod_base import XAIMethodAdapter
from xai_compare import config, strings

@register_xai("shap_kernel")
class KernalSHAPAdapter(XAIMethodAdapter):
    """Model-agnostic Kernel Shap Explainer"""

    display_name = strings.KERNEL_SHAP_DISPLAY_NAME
    default_visualisation = config.KERNEL_SHAP_DEFAULT_VIZ
    compatible_visualisations = config.KERNEL_SHAP_COMPATIBLE_VIZ

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
            raise MissingDataError("KernalSHAPAdapter requires x_train and x_test in RunData")
        explainer = shap.KernelExplainer(model.predict_proba, data.x_train)
        return explainer.shap_values(data.x_test)