
import pandas as pd

from xai_compare.context import RunData
from xai_compare.exceptions import MissingDataError
from xai_compare.registry.xai_registry import register_xai
from xai_compare.adapters.explainers.xaimethod_base import XAIMethodAdapter


@register_xai("coefficients")
class CoefficientsAdapter(XAIMethodAdapter):
    """For models with native interpretability"""

    default_visualisation = "forceplot"
    compatible_visualisations = ["waterfall", "forceplot"]

    def explain_local(self, data: RunData, model=None) -> pd.DataFrame:
        """Computes the local explanation from the coefficients of the model
        Args:
            model: Trained model object
            data: RunData object

        Returns:
            pd.DataFrame: dataframe containing local explanations

        Raises:
            MissingDataError: if data.x is not provided
        """
        if data.x is None:
            raise MissingDataError("CoefficientsAdapter requires x in RunData")
        coef = model.coef_[0]
        local_explanations = data.x * coef
        return local_explanations