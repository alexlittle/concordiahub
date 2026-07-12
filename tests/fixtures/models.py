"""
pytest fixtures for xailens.entities.model.Model.

"""
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from xailens.entities.model import Model

# --------------------
#  helpers
# -------------------

def _build_metadata(id_column, target_column, prediction_column,
                    feature_columns=None, class_names=None,
                    model_display_name=None, xai_methods=None, model_uuid=None):
    return {
        "model_display_name": model_display_name,
        "model_uuid": model_uuid,
        "xai_methods": xai_methods,
        "data_schema": {
            "instance_id_column": id_column,
            "target_column": target_column,
            "prediction_column": prediction_column,
            "feature_columns": feature_columns,
            "class_names": class_names,
        }
    }


def _build_mock_store(y_test, y_pred, metrics, raw_instances, extra_dataframes=None, extra_json=None):
    """
    MagicMock for FilesystemStore.
    """
    store = MagicMock(name="MockFilesystemStore")

    dataframes = {
        "y_test": y_test,
        "y_pred": y_pred,
        "raw_instances": raw_instances,
    }

    if extra_dataframes:
        dataframes.update(extra_dataframes)

    json_data = dict(extra_json or {})

    def load_dataframe(name):
        try:
            return dataframes[name].copy()
        except KeyError:
            raise FileNotFoundError(
                f"No fake data registered for store key '{name}'. "
                f"Available keys: {list(dataframes)}"
            )

    def load_json(name):
        try:
            return json_data[name]
        except KeyError:
            raise FileNotFoundError(
                f"No fake JSON registered for store key '{name}'. "
                f"Available keys: {list(json_data)}"
            )

    store.load_dataframe.side_effect = load_dataframe
    store.load_json.side_effect = load_json
    store.load_series.return_value = metrics.copy()
    return store


def _make_model(path, metadata, y_test, y_pred, metrics, raw_instances, extra_dataframes=None, extra_json=None):
    with patch("xailens.entities.model.FilesystemStore"):
        model = Model(path=path, metadata=metadata)
    model.store = _build_mock_store(y_test, y_pred, metrics, raw_instances, extra_dataframes, extra_json)
    return model


# ----------------------
# Logistic Regression
# -----------------------

@pytest.fixture
def lr_model_factory():
    """
    Factory for a Logistic Regression Model
    """

    def _factory(path="fake/lr_model", n_instances=5,
                 feature_columns=None, class_names=None,
                 model_display_name="Logistic Regression", xai_methods=None,
                 y_test=None, y_pred=None, metrics=None, raw_instances=None, global_explanations=None, local_explanations=None, model_uuid=None):
        id_col, target_col, pred_col = "instance_id", "target", "prediction"
        feature_columns = feature_columns or [
            "age", "sex", "cp", "trestbps", "chol", "fbs",
            "restecg", "thalach", "exang", "oldpeak", "slope", "ca", "thal",
        ]
        class_names = class_names or ["no_disease", "disease"]
        ids = [f"inst_{i}" for i in range(n_instances)]

        if y_test is None:
            y_test = pd.DataFrame({
                id_col: ids,
                target_col: [i % 2 for i in range(n_instances)],
            })
        if y_pred is None:
            y_pred = pd.DataFrame({
                id_col: ids,
                pred_col: [(i + 1) % 2 for i in range(n_instances)],
            })
        if metrics is None:
            metrics = pd.Series({
                "accuracy": 0.83,
                "f1_weighted": 0.81,
                "precision_weighted": 0.82,
                "recall_weighted": 0.83,
            })
        if raw_instances is None:
            raw_instances = pd.DataFrame({
                id_col: ids,
                **{col: [float(i + j) for j in range(n_instances)]
                   for i, col in enumerate(feature_columns)},
            })
        xai_methods = xai_methods or ["coefficients"]

        if global_explanations is None:
            global_explanations = pd.DataFrame({
                "feature": feature_columns,
                "importance": [1.0 / (i + 1) for i in
                               range(len(feature_columns))],
            })

        if local_explanations is None:
            local_explanations = pd.DataFrame({
                id_col: ids,
                **{col: [float((i + j) % 3) - 1 for j in range(n_instances)]
                   for i, col in enumerate(feature_columns)},
            })

        extra_dataframes = {
            f"exp_global_{xai_methods[0]}": global_explanations,
            f"exp_local_{xai_methods[0]}": local_explanations,
        }
        model_uuid = model_uuid or "lr-model-uuid"
        metadata = _build_metadata(id_col, target_col, pred_col,
                                   feature_columns, class_names,
                                   model_display_name, xai_methods, model_uuid)
        return _make_model(path, metadata, y_test, y_pred, metrics,
                           raw_instances, extra_dataframes)

    return _factory


@pytest.fixture
def lr_model(lr_model_factory):
    return lr_model_factory()


# -------------
# XGBoost
# -------------

@pytest.fixture
def xgboost_model_factory():
    """
    Factory for an XGBoost Model
    """

    def _factory(path="fake/xgboost_model", n_instances=5,
                 feature_columns=None, class_names=None,
                 model_display_name="XGBoost", xai_methods=None,
                 y_test=None, y_pred=None, metrics=None, raw_instances=None, global_explanations=None, local_explanations=None, model_uuid=None):
        id_col, target_col, pred_col = "instance_id", "target", "prediction"
        feature_columns = feature_columns or [
            "age", "sex", "cp", "trestbps", "chol", "fbs",
            "restecg", "thalach", "exang", "oldpeak", "slope", "ca", "thal",
        ]
        class_names = class_names or ["no_disease", "disease"]
        ids = [f"inst_{i}" for i in range(n_instances)]

        if y_test is None:
            y_test = pd.DataFrame({
                id_col: ids,
                target_col: [i % 2 for i in range(n_instances)],
            })
        if y_pred is None:
            y_pred = pd.DataFrame({
                id_col: ids,
                pred_col: [i % 2 for i in range(n_instances)],
            })
        if metrics is None:
            metrics = pd.Series({
                "accuracy": 0.89,
                "f1_weighted": 0.88,
                "precision_weighted": 0.87,
                "recall_weighted": 0.89,
            })
        if raw_instances is None:
            raw_instances = pd.DataFrame({
                id_col: ids,
                **{col: [float(i + j) for j in range(n_instances)]
                   for i, col in enumerate(feature_columns)},
            })
        xai_methods = xai_methods or ["tree_shap"]

        if global_explanations is None:
            global_explanations = pd.DataFrame({
                "feature": feature_columns,
                "importance": [1.0 / (i + 1) for i in
                               range(len(feature_columns))],
            })

        if local_explanations is None:
            local_explanations = pd.DataFrame({
                id_col: ids,
                **{col: [float((i + j) % 3) - 1 for j in range(n_instances)]
                   for i, col in enumerate(feature_columns)},
            })

        extra_dataframes = {
            f"exp_global_{xai_methods[0]}": global_explanations,
            f"exp_local_{xai_methods[0]}": local_explanations,
        }

        model_uuid = model_uuid or "xgb-model-uuid"
        metadata = _build_metadata(id_col, target_col, pred_col,
                                   feature_columns, class_names,
                                   model_display_name, xai_methods, model_uuid)
        return _make_model(path, metadata, y_test, y_pred, metrics,
                           raw_instances, extra_dataframes)

    return _factory


@pytest.fixture
def xgboost_model(xgboost_model_factory):
    return xgboost_model_factory()


# --------------
# LLM
# ---------------

@pytest.fixture
def llm_model_factory():
    """
    Factory for an LLM
    """

    def _factory(path="fake/llm_model", n_instances=5,
                 feature_columns=None, class_names=None,
                 model_display_name="LLM (CoT)", xai_methods=None,
                 y_test=None, y_pred=None, metrics=None, raw_instances=None, global_explanations=None, local_explanations_json=None, model_uuid=None):
        id_col, target_col, pred_col = "instance_id", "target", "prediction"
        feature_columns = feature_columns or [
            "age", "sex", "cp", "trestbps", "chol", "fbs",
            "restecg", "thalach", "exang", "oldpeak", "slope", "ca", "thal",
        ]
        class_names = class_names or ["no_disease", "disease"]
        ids = [f"inst_{i}" for i in range(n_instances)]

        if y_test is None:
            y_test = pd.DataFrame({
                id_col: ids,
                target_col: [i % 2 for i in range(n_instances)],
            })
        if y_pred is None:
            y_pred = pd.DataFrame({
                id_col: ids,
                pred_col: [i % 2 for i in range(n_instances)],
                "confidence": [0.6 + 0.05 * i for i in range(n_instances)],
            })
        if metrics is None:
            metrics = pd.Series({
                "accuracy": 0.79,
                "f1_weighted": 0.77,
                "precision_weighted": 0.78,
                "recall_weighted": 0.79,
            })
        if raw_instances is None:
            raw_instances = pd.DataFrame({
                id_col: ids,
                **{col: [float(i + j) for j in range(n_instances)]
                   for i, col in enumerate(feature_columns)},
            })

        xai_methods = xai_methods or ["llm_reasoning"]
        if global_explanations is None:
            global_explanations = pd.DataFrame({
                "feature": feature_columns,
                "importance": [1.0 / (i + 1) for i in
                               range(len(feature_columns))],
            })


        if local_explanations_json is None:
            local_explanations_json = [
                {
                    id_col: id_,
                    "reasoning": f"Fake reasoning for {id_}",
                    "confidence": 0.75,
                    "contributions": {
                        feat: {"score": float((i + 1) * 0.1),
                               "reasoning": f"{feat} contributed"}
                        for i, feat in enumerate(feature_columns)
                    },
                }
                for id_ in ids
            ]

        extra_json = {f"exp_local_{xai_methods[0]}": local_explanations_json}
        extra_dataframes = {f"exp_global_{xai_methods[0]}": global_explanations}

        model_uuid = model_uuid or "llm-model-uuid"
        metadata = _build_metadata(id_col, target_col, pred_col,
                                   feature_columns, class_names,
                                   model_display_name, xai_methods, model_uuid)
        return _make_model(path, metadata, y_test, y_pred, metrics,
                           raw_instances, extra_dataframes, extra_json)

    return _factory


@pytest.fixture
def llm_model(llm_model_factory):
    return llm_model_factory()