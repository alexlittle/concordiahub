import pandas as pd
from abc import ABC, abstractmethod
from typing import Any

from xai_compare.context import RunData
from xai_compare.services.preprocess import normalise_values


class XAIMethodAdapter(ABC):
    """
    Abstract class for XAI method adapter.
    """
    display_name: str | None = None
    default_visualisation: str | None = None
    compatible_visualisations: list[str] = []

    # default to true for most explanation adapters
    requires_encoded_features: bool = True

    # type of explanation storage - df or json
    explanation_file_type: str = "df"

    @classmethod
    def load_explanation(cls, store, sample_id: str):
        raise NotImplementedError

    @classmethod
    def get_abs_max(cls, explanation: pd.Series, sample_id_column) -> float:
        """ calculate absolute max value from explanation
        required for ensuring charts use the same scales """
        features = normalise_values(explanation, sample_id_column)
        return pd.to_numeric(features, errors='coerce').abs().max()

    @abstractmethod
    def explain_local(self, data: RunData, type="df", model=None) -> pd.DataFrame | dict[str, Any]:
        """Computes the local explanation for the model
        Args:
            data: RunData object
            model: Trained model object

        Returns:
            pd.DataFrame: dataframe containing local explanations

        Raises:
            MissingDataError: if data.x_test is not provided
        """
        pass



