
import pandas as pd

from xai_compare.context import RunData
from xai_compare.exceptions import MissingDataError
from xai_compare.registry.xai_registry import register_xai
from xai_compare.adapters.explainers.xaimethod_base import XAIMethodAdapter
from xai_compare.run import Run
from xai_compare import constants

@register_xai("coefficients")
class CoefficientsAdapter(XAIMethodAdapter):
    """For models with native interpretability"""

    default_visualisation = "forceplot"
    compatible_visualisations = ["waterfall", "forceplot"]

    @classmethod
    def load_explanation(cls, run: Run, sample_id: str):
        df = run.store.load_dataframe(f"explanations_{cls.key}")
        row = df[df[run.get(constants.DATA_SCHEMA_ID_COLUMN)] == sample_id]
        if row.empty:
            raise ValueError(f"Sample '{sample_id}' not found in explanations for {cls.key}")
        return row.iloc[0]

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