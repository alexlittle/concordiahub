"""
TreeSHAPAdapter class - for processing SHAP explanations for tree based models
"""

import shap
import pandas as pd
from typing import Any, Optional

from xailens.exceptions import MissingDataError
from xailens.entities.context import ModelContext
from xailens.registry.xai_registry import register_xai
from xailens.adapters.explainers.xaimethod_base import XAIMethodAdapter
from xailens import strings, config
from xailens.entities.model import Model
from xailens.config import TreeModel


@register_xai("tree_shap")
class TreeSHAPAdapter(XAIMethodAdapter):
    """
    Tree SHAP Explainer
    """

    display_name = strings.TREE_SHAP_DISPLAY_NAME
    display_notes = strings.TREE_SHAP_NOTES
    default_visualization = config.TREE_SHAP_DEFAULT_VIZ
    compatible_visualizations = config.TREE_SHAP_COMPATIBLE_VIZ

    # SHAP adds a "class" column, so this needs to be removed
    EXCLUDED_CONTRIBUTION_KEYS = {"class"}

    @classmethod
    def load_local_explanation(cls, model: Model, instance_id: str) -> pd.DataFrame:
        """
        Loads local explanation for given instance_id

        Args:
            model (Model): Model object
            instance_id (str): Instance to get the explanation for

        Returns:
            pd.DataFrame: dataframe containing local explanations
        """

        df = super().load_local_explanation(model, instance_id)
        return df[~df["feature_name"].isin(cls.EXCLUDED_CONTRIBUTION_KEYS)]

    def explain_global_df(self, ctx: ModelContext, model: Optional[TreeModel] = None) -> pd.DataFrame:
        """
        Computes the global explanation from SHAP values

        Args:
            ctx (ModelContext): ModelContext object
            model (Optional[TreeModel]): trained tree-based model object

        Returns:
            pd.DataFrame: dataframe containing global explanations

        Raises:
            MissingDataError: if data.x_test is not provided
            DataProcessingError: if errors in processing SHAP values
        """
        if not hasattr(ctx.data, "x_test") or ctx.data.x_test is None:
            raise MissingDataError(strings.ERROR_MISSING_MODELDATA.format(
                method="cls.key", data="x_test"))
        instance_id = ctx.data.data_schema.instance_id_column
        data = ctx.data.x_test.copy()
        data = data.drop(columns=[instance_id]).select_dtypes(include=['number'])
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(data)

        if shap_values.ndim == 2:
            importance = pd.DataFrame(shap_values, columns=data.columns).abs().mean(axis=0)
            result = importance.reset_index()
            result.columns = ['feature', 'importance']
            return result

        else:
            n_classes = shap_values.shape[2]
            dfs = []
            for i in range(n_classes):
                importance = pd.DataFrame(shap_values[:, :, i], columns=data.columns).abs().mean(axis=0)
                df = importance.reset_index()
                df.columns = ['feature', 'importance']
                dfs.append(df)
            return pd.concat(dfs, ignore_index=True)


    def explain_local_df(self, ctx: ModelContext, model: Optional[TreeModel] = None) -> pd.DataFrame:
        """
        Computes the local explanations from SHAP values of all the instances in the test set

        Args:
            ctx (ModelContext): ModelContext object
            model (Optional[TreeModel]): trained tree-based model object

        Returns:
            pd.DataFrame: dataframe containing local explanations

        Raises:
            MissingDataError: if data.x_test is not provided
            DataProcessingError: if errors in processing SHAP values
        """

        if not hasattr(ctx.data, "x_test") or ctx.data.x_test is None:
            raise MissingDataError(strings.ERROR_MISSING_MODELDATA.format(
                method="cls.key", data="x_test"))

        instance_id = ctx.data.data_schema.instance_id_column
        data = ctx.data.x_test.copy()
        uuids = data[instance_id].values
        data = data.drop(columns=[instance_id]).select_dtypes(include=['number'])

        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(data)

        # (n_instances, 1, n_features)
        if shap_values.ndim == 2:
            result = pd.DataFrame(shap_values, columns=data.columns)
            result.insert(0, instance_id, uuids)
            result['class'] = 0

        # SHAP returns (n_instances, n_features, n_classes) for multiclass tree models
        # slice last axis per class to get (n_instances, n_features) per DataFrame
        else:
            n_classes = shap_values.shape[2]
            dfs = []
            keys = ctx.data.data_schema.class_names if ctx.data.data_schema.class_names is not None else range(
                n_classes)
            for i in range(n_classes):
                df = pd.DataFrame(shap_values[:, :, i], columns=data.columns)
                df.insert(0, instance_id, uuids)
                df['class'] = keys[i]
                dfs.append(df)
            result = pd.concat(dfs, ignore_index=True)


        # filter by abs value
        return result


    def explain_local_json(self, ctx: ModelContext) -> dict[str, Any]:
        """
        Not implemented for TreeSHAPAdapter

        Args:
            ctx (ModelContext): ModelContext object

        Returns:
            dict[str, Any]:

        Raises:
            NotImplementedError: will always return this as not implemented
        """
        raise NotImplementedError