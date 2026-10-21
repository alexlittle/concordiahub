"""
Base abstract class for all XAI method adapters
"""

import pandas as pd
from abc import ABC, abstractmethod
from typing import Any, Optional, ClassVar

from concordiahub.entities.context import ModelContext
from concordiahub.entities.model import Model
from concordiahub.exceptions import MissingDataError
from concordiahub import config, strings
from concordiahub.config import TreeModel

class XAIMethodAdapter(ABC):
    """
    Abstract class for XAI method adapter.
    """
    display_name: str | None = None
    key: ClassVar[str] = None

    display_notes: str | None = None
    default_visualization: Optional[str]
    compatible_visualizations: Optional[dict]

    # type of explanation storage - df or JSON, default to df
    explanation_file_type: str = "df"

    @classmethod
    def load_global_explanation(cls, model: Model) -> pd.DataFrame:
        """
        Loads the global explanation for the instantiated class

        Args:
            model (Model): Model object

        Returns:
            pd.DataFrame: dataframe containing global explanation
        """
        df = model.store.load_dataframe(f"exp_global_{cls.key}")
        explanation = df.groupby("feature")["importance"].mean()

        explanation_abs = explanation.abs()
        abs_max = explanation.abs().max()
        return pd.DataFrame({
            "feature_name": explanation.index,
            "score": explanation.values,
            "score_normalized_max": explanation_abs.values / abs_max if abs_max != 0 else explanation_abs.values,
        })

    @classmethod
    def load_local_explanation(cls, model: Model, instance_id: str) -> pd.DataFrame:
        """

        Args:
            model (Model): Model object:
            instance_id (str): instance id to get the explanation for

        Returns:
            pd.DataFrame: dataframe containing local explanation

        Raises:
            MissingDataError: requested instance is not in the dataset

        """
        df = model.store.load_dataframe(f"exp_local_{cls.key}")
        row = df[df[model.get(config.DATA_SCHEMA_ID_COLUMN)] == instance_id]
        if row.empty:
            raise MissingDataError(strings.ERROR_INSTANCE_NOT_FOUND.format(
                instance_id=instance_id, key=cls.key))
        explanation = row.iloc[0].drop(model.get(config.DATA_SCHEMA_ID_COLUMN), errors='coerce')

        abs_max = explanation.abs().max()
        return pd.DataFrame({
            "feature_name": explanation.index,
            "score": explanation.values,
            "score_normalized_max": explanation.values / abs_max if abs_max != 0 else explanation.values,
            "reasoning": None
        })

    def get_compatible_visualizations(self) -> dict:
        """
        Gets the compatible visualizations for the instantiated class

        Returns:
            dict: compatible visualizations
        """
        return self.compatible_visualizations

    def get_default_visualization(self) -> str:
        """
        Gets the default visualization for the instantiated class

        Returns:
            str: default visualization
        """
        return self.default_visualization

    @abstractmethod
    def explain_global_df(self, ctx: ModelContext, model: Optional[TreeModel] = None) -> pd.DataFrame:
        """Computes the global explanation for the model

        Args:
            ctx (ModelContext): ModelContext object
            model (Optional[TreeModel]): Trained model object

        Returns:
            pd.DataFrame: dataframe containing global explanation

        Raises:
            MissingDataError: if required data is not provided
        """
        pass

    @abstractmethod
    def explain_local_df(self, ctx: ModelContext, model: Optional[TreeModel] = None) -> pd.DataFrame:
        """Computes the local explanation for the model

        Args:
            ctx (ModelContext): ModelContext object
            model (Optional[TreeModel]): Trained model object

        Returns:
            pd.DataFrame: dataframe containing local explanations

        """
        pass

    @abstractmethod
    def explain_local_json(self, ctx: ModelContext) -> list[dict[str, Any]]:
        """Computes the local explanation for the model

        Args:
            ctx (ModelContext): ModelContext object

        Returns:
            list[dict[str, Any]]: JSON containing local explanations

        Raises:
            MissingDataError: if required data is not provided
        """
        pass



