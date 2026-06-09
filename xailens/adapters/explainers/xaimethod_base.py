import pandas as pd
from abc import ABC, abstractmethod
from typing import Any, Optional

from xailens.context import RunContext


class XAIMethodAdapter(ABC):
    """
    Abstract class for XAI method adapter.
    """
    display_name: str | None = None
    default_visualisation: Optional[str]
    compatible_visualisations: Optional[dict]

    # default to true for most explanation adapters
    requires_encoded_features: bool = True

    # type of explanation storage - df or JSON, default to df
    explanation_file_type: str = "df"

    @classmethod
    def load_explanation(cls, store, sample_id: str) -> pd.Series:
        raise NotImplementedError

    @classmethod
    def get_abs_max(cls, explanation: pd.Series, sample_id_column) -> float:
        features = explanation.drop(index=sample_id_column, errors='ignore')
        return pd.to_numeric(features, errors='coerce').abs().max()


    def get_compatible_visualisations(self):
        """Gets the compatible visualisations for the instantiated class"""
        return self.compatible_visualisations

    def get_default_visualisation(self):
        """Gets the default visualisation for the instantiated class"""
        return self.default_visualisation

    @abstractmethod
    def explain_local_df(self, ctx: RunContext, model=None) -> pd.DataFrame:
        """Computes the local explanation for the model
        Args:
            ctx: RunContext object
            model: Trained model object

        Returns:
            pd.DataFrame: dataframe containing local explanations

        Raises:
            MissingDataError: if required data is not provided
        """
        pass

    @abstractmethod
    def explain_local_json(self, ctx: RunContext) -> list[dict[str, Any]]:
        """Computes the local explanation for the model
        Args:
            ctx: RunContext object

        Returns:
            list of dicts - JSON containing local explanations

        Raises:
            MissingDataError: if required data is not provided
        """
        pass



