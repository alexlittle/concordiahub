import pandas as pd
from abc import ABC, abstractmethod
from typing import Any, Optional

from xailens.entities.context import RunContext
from xailens.entities.model import ModelEval
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

    # default to true for most explanation adapters
    requires_encoded_features: bool = True

    # type of explanation storage - df or JSON, default to df
    explanation_file_type: str = "df"

    @classmethod
    def load_explanation(cls, run: ModelEval, sample_id: str) -> pd.DataFrame:
        df = run.store.load_dataframe(f"explanations_{cls.key}")
        row = df[df[run.get(constants.DATA_SCHEMA_ID_COLUMN)] == sample_id]
        if row.empty:
            raise MissingDataError(f"Sample '{sample_id}' not found in explanations for {cls.key}")
        explanation = row.iloc[0].drop(run.get(constants.DATA_SCHEMA_ID_COLUMN), errors='coerce')

        # filter out near-zero valued features if a threshold is set
        schema = run.get_data_schema()
        if schema.hide_zero_threshold is not None:
            explanation = explanation[explanation.abs() > schema.hide_zero_threshold]

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



