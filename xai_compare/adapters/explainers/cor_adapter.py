
import pandas as pd
from typing import Any

from xai_compare.context import RunData
from xai_compare.exceptions import MissingDataError
from xai_compare.registry.xai_registry import register_xai
from xai_compare.adapters.explainers.xaimethod_base import XAIMethodAdapter
from xai_compare import strings, config


@register_xai("cor")
class CoRAdapter(XAIMethodAdapter):
    """For chain-of-reasoning/chain-of-thought explanations"""

    display_name = strings.COR_DISPLAY_NAME
    default_visualisation = config.COR_DEFAULT_VIZ
    compatible_visualisations = config.COR_COMPATIBLE_VIZ

    # no features as such so will process the explanations in this adapter
    requires_encoded_features: bool = False

    # uses a json file for explanation storage
    explanation_file_type: str = "json"

    def explain_local_df(self, data: RunData, model=None) -> pd.DataFrame:
        raise NotImplementedError

    def explain_local_json(self, data: RunData) -> dict[str, Any]:
        """Processes the CoR local explanations
        Args:
            data: RunData object

        Returns:
            pd.DataFrame: dataframe containing local explanations

        Raises:
            MissingDataError: if data.x is not provided
        """
        if data.y_exp is None:
            raise MissingDataError("CoRAdapter requires y_exp in RunData")

        # process the explanation