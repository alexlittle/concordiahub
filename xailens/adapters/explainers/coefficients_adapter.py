
import pandas as pd
from typing import Any

from xailens.exceptions import MissingDataError
from xailens.registry.xai_registry import register_xai
from xailens.adapters.explainers.xaimethod_base import XAIMethodAdapter
from xailens.run import Run
from xailens import constants, strings, config, context
from xailens.context import RunContext


@register_xai("coefficients")
class CoefficientsAdapter(XAIMethodAdapter):
    """For models with native interpretability"""

    display_name = strings.COEFFICIENTS_DISPLAY_NAME
    default_visualisation = config.COEFFS_DEFAULT_VIZ

    compatible_visualisations = config.COEFFS_COMPATIBLE_VIZ

    @classmethod
    def load_explanation(cls, run: Run, sample_id: str) -> pd.Series:
        df = run.store.load_dataframe(f"explanations_{cls.key}")
        row = df[df[run.get(constants.DATA_SCHEMA_ID_COLUMN)] == sample_id]
        if row.empty:
            raise MissingDataError(f"Sample '{sample_id}' not found in explanations for {cls.key}")
        return row.iloc[0]

    def explain_local_df(self, ctx: RunContext, model=None) -> pd.DataFrame:
        """Computes the local explanation from the coefficients of the model
        Args:
            ctx: RunContext object
            model: Trained model object

        Returns:
            pd.DataFrame: dataframe containing local explanations

        Raises:
            MissingDataError: if data.x is not provided
        """
        if ctx.data.x is None:
            raise MissingDataError("CoefficientsAdapter requires x in RunData")
        coef = model.coef_[0]
        local_explanations = ctx.data.x * coef
        return local_explanations

    def explain_local_json(self, ctx: RunContext) -> dict[str, Any]:
        raise NotImplementedError