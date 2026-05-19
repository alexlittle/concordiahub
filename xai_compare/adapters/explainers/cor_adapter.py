
import pandas as pd

from xai_compare.config import RunData
from xai_compare.exceptions import MissingDataError
from xai_compare.registry.xai_registry import register_xai
from xai_compare.adapters.explainers.xaimethod_base import XAIMethodAdapter



@register_xai("cor")
class CoRAdapter(XAIMethodAdapter):
    """For chain-of-reasoning/chain-of-thought explanations"""

    # no features as such so will process the explanations in this adapter
    requires_encoded_features: bool = False

    def explain_local(self, data: RunData, model=None) -> pd.DataFrame:
        """Processes the CoR local explanations
        Args:
            model:  Trained model object
            data: RunData object

        Returns:
            pd.DataFrame: dataframe containing local explanations

        Raises:
            MissingDataError: if data.x is not provided
        """
        if data.y_exp is None:
            raise MissingDataError("CoRAdapter requires y_exp in RunData")

        # process the explanation