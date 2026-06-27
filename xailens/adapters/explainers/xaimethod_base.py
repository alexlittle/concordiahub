import pandas as pd
from abc import ABC, abstractmethod
from typing import Any, Optional

from xailens.entities.context import ModelContext
from xailens.entities.modeleval import ModelEval
from xailens.exceptions import MissingDataError
from xailens import constants


class XAIMethodAdapter(ABC):
    """
    Abstract class for XAI method adapter.
    """
    display_name: str | None = None
    display_notes: str | None = None
    default_visualisation: Optional[str]
    compatible_visualisations: Optional[dict]

    # type of explanation storage - df or JSON, default to df
    explanation_file_type: str = "df"

    @classmethod
    def load_global_explanation(cls, modeleval: ModelEval) -> pd.DataFrame:
        df = modeleval.store.load_dataframe(f"exp_global_{cls.key}")
        explanation = df.groupby("feature")["importance"].mean()

        explanation_abs = explanation.abs()
        abs_sum = explanation.abs().sum()
        abs_max = explanation.abs().max()
        return pd.DataFrame({
            "feature_name": explanation.index,
            "score": explanation.values,
            "score_normalised_sum": explanation_abs.values / abs_sum if abs_sum != 0 else explanation_abs.values,
            "score_normalised_max": explanation_abs.values / abs_max if abs_max != 0 else explanation_abs.values,
        })

    @classmethod
    def load_local_explanation(cls, modeleval: ModelEval, instance_id: str) -> pd.DataFrame:
        df = modeleval.store.load_dataframe(f"exp_local_{cls.key}")
        row = df[df[modeleval.get(constants.DATA_SCHEMA_ID_COLUMN)] == instance_id]
        if row.empty:
            raise MissingDataError(f"Instance '{instance_id}' not found in explanations for {cls.key}")
        explanation = row.iloc[0].drop(modeleval.get(constants.DATA_SCHEMA_ID_COLUMN), errors='coerce')

        abs_sum = explanation.abs().sum()
        abs_max = explanation.abs().max()
        return pd.DataFrame({
            "feature_name": explanation.index,
            "score": explanation.values,
            "score_normalised_sum": explanation.values / abs_sum if abs_sum != 0 else explanation.values,
            "score_normalised_max": explanation.values / abs_max if abs_max != 0 else explanation.values,
            "reasoning": None
        })

    def get_compatible_visualisations(self):
        """Gets the compatible visualisations for the instantiated class"""
        return self.compatible_visualisations

    def get_default_visualisation(self):
        """Gets the default visualisation for the instantiated class"""
        return self.default_visualisation

    @abstractmethod
    def explain_global_df(self, ctx: ModelContext, model=None) -> pd.DataFrame:
        """Computes the global explanation for the model
        Args:
            ctx: ModelContext object
            model: Trained model object

        Returns:
            pd.DataFrame: dataframe containing global explanation

        Raises:
            MissingDataError: if required data is not provided
        """
        pass

    @abstractmethod
    def explain_local_df(self, ctx: ModelContext, model=None) -> pd.DataFrame:
        """Computes the local explanation for the model
        Args:
            ctx: ModelContext object
            model: Trained model object

        Returns:
            pd.DataFrame: dataframe containing local explanations

        Raises:
            MissingDataError: if required data is not provided
        """
        pass

    @abstractmethod
    def explain_local_json(self, ctx: ModelContext) -> list[dict[str, Any]]:
        """Computes the local explanation for the model
        Args:
            ctx: ModelContext object

        Returns:
            list of dicts - JSON containing local explanations

        Raises:
            MissingDataError: if required data is not provided
        """
        pass



