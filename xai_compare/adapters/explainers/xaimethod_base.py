import pandas as pd
from abc import ABC, abstractmethod

from xai_compare.config import RunData

class XAIMethodAdapter(ABC):
    """
    Abstract class for XAI method adapter.
    """

    @abstractmethod
    def explain_local(self, model, data: RunData) -> pd.DataFrame:
        """Computes the local explanation for the model
        Args:
            model: Trained model object
            data: RunData object

        Returns:
            pd.DataFrame: dataframe containing local explanations

        Raises:
            MissingDataError: if data.x_test is not provided
        """
        pass



