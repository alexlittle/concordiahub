import shap
import pandas as pd
from typing import Any

from xai_compare.exceptions import MissingDataError, DataProcessingError
from xai_compare.context import RunContext
from xai_compare.registry.xai_registry import register_xai
from xai_compare.adapters.explainers.xaimethod_base import XAIMethodAdapter
from xai_compare.run import Run
from xai_compare import constants, strings, config

@register_xai("tree_shap")
class TreeSHAPAdapter(XAIMethodAdapter):
    """Tree SHAP Explainer"""

    display_name = strings.TREE_SHAP_DISPLAY_NAME
    default_visualisation = config.TREE_SHAP_DEFAULT_VIZ
    compatible_visualisations = config.TREE_SHAP_COMPATIBLE_VIZ

    @classmethod
    def load_explanation(cls, run: Run, sample_id: str) -> pd.Series:
        df = run.store.load_dataframe(f"explanations_{cls.key}")
        row = df[df[run.get(constants.DATA_SCHEMA_ID_COLUMN)] == sample_id]
        if row.empty:
            raise ValueError(f"Sample '{sample_id}' not found in explanations for {cls.key}")
        return row.iloc[0]


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
        if ctx.data.data_schema.sample_id_column in data.columns:
            data = data.set_index(ctx.data.data_schema.sample_id_column)
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