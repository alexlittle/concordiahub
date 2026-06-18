import shap
import pandas as pd
from typing import Any

from xailens.exceptions import MissingDataError, DataProcessingError
from xailens.context import RunContext
from xailens.registry.xai_registry import register_xai
from xailens.adapters.explainers.xaimethod_base import XAIMethodAdapter
from xailens import strings, config
from xailens.run import Run

@register_xai("tree_shap")
class TreeSHAPAdapter(XAIMethodAdapter):
    """Tree SHAP Explainer"""

    display_name = strings.TREE_SHAP_DISPLAY_NAME
    display_notes = strings.TREE_SHAP_NOTES
    default_visualisation = config.TREE_SHAP_DEFAULT_VIZ
    compatible_visualisations = config.TREE_SHAP_COMPATIBLE_VIZ

    # SHAP adds a "class" column, so this needs to be removed
    EXCLUDED_CONTRIBUTION_KEYS = {"class"}
    
    @classmethod
    def load_explanation(cls, run: Run, sample_id: str) -> pd.DataFrame:
        df = super().load_explanation(run, sample_id)
        return df[~df["feature_name"].isin(cls.EXCLUDED_CONTRIBUTION_KEYS)]

    def explain_local_df(self, ctx: RunContext, model=None) -> pd.DataFrame:
        """Computes the local explanation from the coefficients of the model
        Args:
            ctx: RunContext object
            model: Trained model object

        Returns:
            pd.DataFrame: dataframe containing local explanations

        Raises:
            MissingDataError: if data.x_test is not provided
        """
        if not hasattr(ctx.data, "x_test") or ctx.data.x_test is None:
            raise MissingDataError("CoefficientsAdapter requires x_test in RunData")

        data = ctx.data.x_test
        if ctx.data.data_schema.instance_id_column in data.columns:
            data = data.set_index(ctx.data.data_schema.instance_id_column)
        data = data.select_dtypes(include=['number'])

        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(data)

        # (n_samples, 1, n_features)
        if shap_values.ndim == 2:
            dfs = [pd.DataFrame(shap_values, index=data.index, columns=ctx.data.x_test.columns)]
            dfs[0]['class'] = 0
            return dfs[0].set_index('class', append=True)

        # SHAP returns (n_samples, n_features, n_classes) for multiclass tree models
        # slice last axis per class to get (n_samples, n_features) per DataFrame
        elif shap_values.ndim == 3:
            n_classes = shap_values.shape[2]
            dfs = []

            for i in range(n_classes):
                # Slice the last axis [:, :, i] to get (n_samples, n_features)
                df = pd.DataFrame(
                    shap_values[:, :, i],
                    index=data.index,
                    columns=data.columns
                )
                dfs.append(df)

            keys = ctx.data.data_schema.class_names if ctx.data.data_schema.class_names is not None else range(n_classes)
            return pd.concat(dfs, keys=keys, names=['class', data.index.name])
        else:
            raise DataProcessingError("The return values from SHAP don't match")

    def explain_local_json(self, ctx: RunContext) -> dict[str, Any]:
        raise NotImplementedError