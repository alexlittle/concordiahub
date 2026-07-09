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
                     feature_columns=None, class_names=None):
    return {
        "data_schema": {
            "instance_id_column": id_column,
            "target_column": target_column,
            "prediction_column": prediction_column,
            "feature_columns": feature_columns,
            "class_names": class_names,
        }
    }


def _build_mock_store(y_test, y_pred, metrics, raw_instances):
    """
    Build a MagicMock standing in for FilesystemStore.
    """
    store = MagicMock(name="MockFilesystemStore")

    dataframes = {
        "y_test": y_test,
        "y_pred": y_pred,
        "raw_instances": raw_instances,
    }

    def load_dataframe(name):
        try:
            return dataframes[name].copy()
        except KeyError:
            raise FileNotFoundError(
                f"No fake data registered for store key '{name}'. "
                f"Available keys: {list(dataframes)}"
            )

    store.load_dataframe.side_effect = load_dataframe
    store.load_series.return_value = metrics.copy()
    return store


def _make_model(path, metadata, y_test, y_pred, metrics, raw_instances):
    with patch("xailens.entities.model.FilesystemStore"):
        model = Model(path=path, metadata=metadata)
    model.store = _build_mock_store(y_test, y_pred, metrics, raw_instances)
    return model


# ---------------------
# Logistic Regression
# ----------------------

@pytest.fixture
def lr_model_factory():
    """
    Factory for a Logistic Regression Model on Cleveland-style heart
    disease data. Call with overrides as needed, e.g.:

        lr_model_factory(n_instances=10, metrics={"f1_weighted": 0.5})
    """
    def _factory(path="fake/lr_model", n_instances=5,
                 feature_columns=None, class_names=None,
                 y_test=None, y_pred=None, metrics=None, raw_instances=None):
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

        metadata = _build_metadata(id_col, target_col, pred_col,
                                    feature_columns, class_names)
        return _make_model(path, metadata, y_test, y_pred, metrics, raw_instances)

    return _factory


@pytest.fixture
def lr_model(lr_model_factory):
    """Ready-to-use Logistic Regression Model with default fake data."""
    return lr_model_factory()


# --------------------
# XGBoost
# --------------------

@pytest.fixture
def xgboost_model_factory():
    """
    Factory for an XGBoost Model over Cleveland-style heart disease data.
    """
    def _factory(path="fake/xgboost_model", n_instances=5,
                 feature_columns=None, class_names=None,
                 y_test=None, y_pred=None, metrics=None, raw_instances=None):
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
            # XGBoost gets one more instance right than the LR fixture,
            # useful for testing agreement/disagreement logic
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

        metadata = _build_metadata(id_col, target_col, pred_col,
                                    feature_columns, class_names)
        return _make_model(path, metadata, y_test, y_pred, metrics, raw_instances)

    return _factory


@pytest.fixture
def xgboost_model(xgboost_model_factory):
    """Ready-to-use XGBoost Model with default fake data."""
    return xgboost_model_factory()


# ---------------------------------------------------------------------------
# LLM
# ---------------------------------------------------------------------------

@pytest.fixture
def llm_model_factory():

    def _factory(path="fake/llm_model", n_instances=5,
                 feature_columns=None, class_names=None,
                 y_test=None, y_pred=None, metrics=None, raw_instances=None):
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

        metadata = _build_metadata(id_col, target_col, pred_col,
                                    feature_columns, class_names)
        return _make_model(path, metadata, y_test, y_pred, metrics, raw_instances)

    return _factory


@pytest.fixture
def llm_model(llm_model_factory):
    """Ready-to-use LLM Model with default fake data."""
    return llm_model_factory()