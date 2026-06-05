
import pandas as pd
from typing import Any

from xai_compare.context import RunData
from xai_compare.exceptions import MissingDataError
from xai_compare.registry.xai_registry import register_xai
from xai_compare.adapters.explainers.xaimethod_base import XAIMethodAdapter
from xai_compare.run import Run
from xai_compare import constants, strings, config


@register_xai("coefficients")
class CoefficientsAdapter(XAIMethodAdapter):
    """For models with native interpretability"""

    display_name = strings.COEFFICIENTS_DISPLAY_NAME
    default_visualisation = config.COEFFICIENTS_DEFAULT_VIZ
    compatible_visualisations = config.COEFFICIENTS_COMPATIBLE_VIZ

    @classmethod
    def load_explanation(cls, run: Run, sample_id: str):
        df = run.store.load_dataframe(f"explanations_{cls.key}")
        row = df[df[run.get(constants.DATA_SCHEMA_ID_COLUMN)] == sample_id]
        if row.empty:
            raise MissingDataError(f"Sample '{sample_id}' not found in explanations for {cls.key}")
        return row.iloc[0]

    def explain_local_df(self, data: RunData, model=None) -> pd.DataFrame:
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

    def explain_local_json(self, data: RunData) -> dict[str, Any]:
        raise NotImplementedError