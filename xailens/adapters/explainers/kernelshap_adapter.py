import shap
import pandas as pd
from typing import Any

from xailens.exceptions import MissingDataError
from xailens.context import RunContext
from xailens.registry.xai_registry import register_xai
from xailens.adapters.explainers.xaimethod_base import XAIMethodAdapter
from xailens import config, strings

@register_xai("shap_kernel")
class KernalSHAPAdapter(XAIMethodAdapter):
    """Model-agnostic Kernel Shap Explainer"""

    display_name = strings.KERNEL_SHAP_DISPLAY_NAME

    default_visualisation = None
    compatible_visualisations = {}

    def explain_local_df(self, ctx: RunContext, model=None) -> pd.DataFrame:
        """Computes the local explanation using KernelShap method
        Args:
            ctx: RunContext object
            model: Trained model object

        Returns:
            pd.DataFrame: dataframe containing local explanations

        Raises:
            MissingDataError: if x_train or x_test is not provided
        """
        if ctx.data.x_train is None or ctx.data.x_test is None:
            raise MissingDataError("KernalSHAPAdapter requires x_train and x_test in RunData")
        explainer = shap.KernelExplainer(model.predict_proba, ctx.data.x_train)
        return explainer.shap_values(ctx.data.x_test)

    def explain_local_json(self, data: RunContext) -> dict[str, Any]:
        raise NotImplementedError