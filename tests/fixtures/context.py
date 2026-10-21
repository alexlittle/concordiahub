"""
Reusable pytest fixtures for concordiahub.runner.run(), built around
ModelContext / ModelData / DataSchema.
"""
from unittest.mock import MagicMock

import pandas as pd
import pytest

from concordiahub.entities.context import ModelContext, ModelData, DataSchema


@pytest.fixture
def data_schema_factory():
    """Fixture for building a DataSchema object"""
    def _factory(instance_id_column="instance_id", target_column="target",
                 prediction_column="prediction", feature_columns=None,
                 class_names=None):
        feature_columns = feature_columns or ["age", "sex", "chol"]
        class_names = class_names or ["no_disease", "disease"]
        return DataSchema(
            instance_id_column=instance_id_column,
            target_column=target_column,
            prediction_column=prediction_column,
            feature_columns=feature_columns,
            class_names=class_names,
        )
    return _factory


@pytest.fixture
def model_data_factory(data_schema_factory):
    """Fixture for building a ModelData object"""
    def _factory(n_instances=5, data_schema=None,
                 raw_instances=None, x_test=None, y_test=None,
                 y_pred=None, y_exp=None):
        data_schema = data_schema or data_schema_factory()
        id_col = data_schema.instance_id_column
        target_col = data_schema.target_column
        pred_col = data_schema.prediction_column
        feature_columns = data_schema.feature_columns
        ids = [f"inst_{i}" for i in range(n_instances)]

        if raw_instances is None:
            raw_instances = pd.DataFrame({
                id_col: ids,
                **{col: [float(i + j) for j in range(n_instances)]
                   for i, col in enumerate(feature_columns)},
            })
        if x_test is None:
            x_test = pd.DataFrame({
                id_col: ids,
                **{col: [float(i + j) for j in range(n_instances)]
                   for i, col in enumerate(feature_columns)},
            })
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

        return ModelData(
            data_schema=data_schema,
            raw_instances=raw_instances,
            x_test=x_test,
            y_test=y_test,
            y_pred=y_pred,
            y_exp=y_exp,
        )
    return _factory

@pytest.fixture
def model_context_factory(model_data_factory):
    """Fixture for building a ModelContext object"""
    def _factory(model_type="logistic_regression", xai_methods=None,
                 dataset_id="cleveland_heart_disease", data=None,
                 model_display_name="Logistic Regression", model=None, response_schema=None):
        xai_methods = xai_methods or ["coefficients"]
        data = data or model_data_factory()
        return ModelContext(
            model_type=model_type,
            xai_methods=xai_methods,
            dataset_id=dataset_id,
            data=data,
            model_display_name=model_display_name,
            model=model or MagicMock(name="fake_sklearn_model"),
            response_schema=response_schema,
        )
    return _factory