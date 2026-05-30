import pandas as pd
from abc import ABC, abstractmethod

from xai_compare.context import RunData

class XAIMethodAdapter(ABC):
    """
    Abstract class for XAI method adapter.
    """
    default_visualisation: str | None = None
    compatible_visualisations: list[str] = []

    # default to true for most explanation adapters
    requires_encoded_features: bool = True

    @classmethod
    def load_explanation(cls, store, sample_id: str):
        raise NotImplementedError

    @abstractmethod
    def explain_local(self, data: RunData, model=None) -> pd.DataFrame:
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



