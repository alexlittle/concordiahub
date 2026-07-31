"""
CoefficientsAdapter class - for model with native interpretability
"""

import pandas as pd
import numpy as np

from typing import Any, Optional, Protocol

from xailens.exceptions import MissingDataError
from xailens.registry.xai_registry import register_xai
from xailens.adapters.explainers.xaimethod_base import XAIMethodAdapter

from xailens import strings, config
from xailens.entities.context import ModelContext

class HasCoefficients(Protocol):
    """
    Helper class for type hinting - the model used for the CoefficientsAdapter
    must have a "coef_" attribute
    """
    coef_: np.ndarray

@register_xai("coefficients")
class CoefficientsAdapter(XAIMethodAdapter):
    """For models with native interpretability"""

    display_name = strings.COEFFICIENTS_DISPLAY_NAME
    display_notes = strings.COEFFICIENTS_NOTES
    default_visualisation = config.COEFFS_DEFAULT_VIZ
    compatible_visualisations = config.COEFFS_COMPATIBLE_VIZ


    def explain_local_df(self, ctx: ModelContext, model: Optional[HasCoefficients] = None ) -> pd.DataFrame:
        """Computes the local explanation from the coefficients of the model

        Args:
            ctx (ModelContext) : ModelContext object
            model (Optional[HasCoefficients]): model exposing a coef_ attribute

        Returns:
            pd.DataFrame: dataframe containing local explanations

        Raises:
            MissingDataError: if data.x is not provided
        """
        if ctx.data.x_test is None:
            raise MissingDataError(strings.ERROR_MISSING_MODELDATA.format(method="cls.key", data="x_test"))
        coef = model.coef_[0]
        instance_id = ctx.data.data_schema.instance_id_column
        x_test = ctx.data.x_test.drop(columns=instance_id)
        local_explanations = x_test * coef
        local_explanations.insert(0, instance_id, ctx.data.x_test[instance_id].values)
        return local_explanations

    def explain_global_df(self, ctx: ModelContext, model: Optional[HasCoefficients] = None) -> pd.DataFrame:
        """
        Computes the global explanation from the coefficients of the model

        Args:
            ctx (ModelContext): ModelContext object
            model (Optional[HasCoefficients]): model exposing a coef_ attribute

        Returns:
            pd.DataFrame: dataframe containing global explanation, with two columns, feature and importance
        """

        coef = model.coef_[0]
        feature_names = ctx.data.x_test.drop(columns=ctx.data.data_schema.instance_id_column).columns
        importance = pd.DataFrame({
            "feature": feature_names,
            "importance": coef,
        })
        return importance


    def explain_local_json(self, ctx: ModelContext) -> dict[str, Any]:
        """
        Not implemented for CoefficientsAdapter

        Args:
            ctx (ModelContext): ModelContext object

        Returns:
            dict[str, Any]:

        Raises:
            NotImplementedError:
        """
        raise NotImplementedError