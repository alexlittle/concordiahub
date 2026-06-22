
import pandas as pd
from typing import Any

from xailens.exceptions import MissingDataError
from xailens.registry.xai_registry import register_xai
from xailens.adapters.explainers.xaimethod_base import XAIMethodAdapter

from xailens import strings, config
from xailens.entities.context import ModelContext


@register_xai("coefficients")
class CoefficientsAdapter(XAIMethodAdapter):
    """For models with native interpretability"""

    display_name = strings.COEFFICIENTS_DISPLAY_NAME
    display_notes = strings.COEFFICIENTS_NOTES
    default_visualisation = config.COEFFS_DEFAULT_VIZ
    compatible_visualisations = config.COEFFS_COMPATIBLE_VIZ


    def explain_local_df(self, ctx: ModelContext, model=None) -> pd.DataFrame:
        """Computes the local explanation from the coefficients of the model
        Args:
            ctx: ModelContext object
            model: Trained model object

        Returns:
            pd.DataFrame: dataframe containing local explanations

        Raises:
            MissingDataError: if data.x is not provided
        """
        if ctx.data.x is None:
            raise MissingDataError("CoefficientsAdapter requires x in ModelData")
        coef = model.coef_[0]
        instance_id = ctx.data.data_schema.instance_id_column
        x = ctx.data.x.drop(columns=instance_id)
        local_explanations = x * coef
        local_explanations.insert(0, instance_id, ctx.data.x[instance_id].values)
        return local_explanations

    def explain_local_json(self, ctx: ModelContext) -> dict[str, Any]:
        raise NotImplementedError