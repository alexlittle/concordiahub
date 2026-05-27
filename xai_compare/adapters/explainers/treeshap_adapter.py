import shap
import pandas as pd

from xai_compare.exceptions import MissingDataError
from xai_compare.context import RunData
from xai_compare.registry.xai_registry import register_xai
from xai_compare.adapters.explainers.xaimethod_base import XAIMethodAdapter



@register_xai("shap_tree")
class TreeSHAPAdapter(XAIMethodAdapter):
    """Tree Shap Explainer"""

    def explain_local(self, data: RunData, model=None) -> pd.DataFrame:
        """Computes the local explanation from the coefficients of the model
        Args:
            model: Trained model object
            data: RunData object

        Returns:
            pd.DataFrame: dataframe containing local explanations

        Raises:
            MissingDataError: if data.x_test is not provided
        """
        if not hasattr(data, "x_test") or data.x_test is None:
            raise MissingDataError("CoefficientsAdapter requires x_test in RunData")
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(data.x_test)

        # (n_samples, 1, n_features)
        if shap_values.ndim == 2:
            dfs = [pd.DataFrame(shap_values, index=data.x_test.index, columns=data.x_test.columns)]
            dfs[0]['class'] = 0
            return dfs[0].set_index('class', append=True)

        # SHAP returns (n_samples, n_features, n_classes) for multiclass tree models
        # slice last axis per class to get (n_samples, n_features) per DataFrame
        if shap_values.ndim == 3:
            n_classes = shap_values.shape[2]
            dfs = []

            for i in range(n_classes):
                # Slice the last axis [:, :, i] to get (n_samples, n_features)
                df = pd.DataFrame(
                    shap_values[:, :, i],
                    index=data.x_test.index,
                    columns=data.x_test.columns
                )
                dfs.append(df)

            keys = data.data_schema.class_names if data.data_schema.class_names is not None else range(n_classes)
            return pd.concat(dfs, keys=keys, names=['class', data.x_test.index.name])