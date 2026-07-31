"""
Tests for TreeSHAPAdapter

Tests cover:

- missing data
- binary classification
- multiclass classification
- returns a dataframe object
"""

import pytest
import numpy as np
import pandas as pd
from unittest.mock import MagicMock, patch
from xailens.adapters.explainers.treeshap_adapter import TreeSHAPAdapter
from xailens.entities.context import ModelData, DataSchema, ModelContext
from xailens.exceptions import MissingDataError

MODULE_UNDER_TEST = "xailens.adapters.explainers.treeshap_adapter"

@pytest.fixture
def adapter():
    return TreeSHAPAdapter()

@pytest.fixture
def instance_uuids():
    return ['uuid-001', 'uuid-002', 'uuid-003']

@pytest.fixture
def data_schema():
    return DataSchema("instance_uuid", "target", "prediction")

@pytest.fixture
def x_test(instance_uuids):
    df = pd.DataFrame(
        np.random.rand(3, 5),
        columns=['f1', 'f2', 'f3', 'f4', 'f5'],
    )
    df.insert(0, 'instance_uuid', instance_uuids)
    return df

@pytest.fixture
def mock_model():
    return MagicMock()

def test_raises_missing_data_error_when_x_test_is_none(adapter, mock_model, data_schema):
    data = ModelData(data_schema, x_test=None, y_test=None, raw_instances=NameError, y_pred=None)
    ctx = ModelContext(model_type="xgboost",
                       xai_methods=["shap_tree"],
                       dataset_id="dataset",
                       data=data,
                       model_display_name="model_display_name")
    with pytest.raises(MissingDataError):
        adapter.explain_local_df(ctx, mock_model)

def test_binary_returns_dataframe(adapter, mock_model, x_test, data_schema):
    shap_values = np.random.rand(3, 5)  # (n_instances, n_features)
    with patch(f"{MODULE_UNDER_TEST}.shap.TreeExplainer") as mock_explainer:
        mock_explainer.return_value.shap_values.return_value = shap_values
        data = ModelData(data_schema, x_test=x_test, y_test=None, raw_instances=NameError, y_pred=None)
        ctx = ModelContext(model_type="xgboost",
                           xai_methods=["shap_tree"],
                           dataset_id="dataset",
                           data=data,
                           model_display_name="model_display_name")
        result = adapter.explain_local_df(ctx, mock_model)
        assert isinstance(result, pd.DataFrame)

def test_binary_has_class_column(adapter, mock_model, x_test, data_schema):
    shap_values = np.random.rand(3, 5)
    with patch(f"{MODULE_UNDER_TEST}.shap.TreeExplainer") as mock_explainer:
        mock_explainer.return_value.shap_values.return_value = shap_values
        data = ModelData(data_schema, x_test=x_test, y_test=None, raw_instances=NameError, y_pred=None)
        ctx = ModelContext(model_type="xgboost",
                           xai_methods=["shap_tree"],
                           dataset_id="dataset",
                           data=data,
                           model_display_name="model_display_name")
        result = adapter.explain_local_df(ctx, mock_model)
        assert 'class' in result.columns

def test_multiclass_returns_dataframe(adapter, mock_model, x_test, data_schema):
    shap_values = np.random.rand(3, 5, 3)  # (n_instances, n_features, n_classes)
    with patch(f"{MODULE_UNDER_TEST}.shap.TreeExplainer") as mock_explainer:
        mock_explainer.return_value.shap_values.return_value = shap_values
        data = ModelData(data_schema, x_test=x_test, y_test=None, raw_instances=NameError, y_pred=None)
        ctx = ModelContext(model_type="xgboost",
                           xai_methods=["shap_tree"],
                           dataset_id="dataset",
                           data=data,
                           model_display_name="model_display_name")
        result = adapter.explain_local_df(ctx, mock_model)
        assert isinstance(result, pd.DataFrame)

def test_multiclass_has_correct_shape(adapter, mock_model, x_test, data_schema):
    shap_values = np.random.rand(3, 5, 3)
    with patch(f"{MODULE_UNDER_TEST}.shap.TreeExplainer") as mock_explainer:
        mock_explainer.return_value.shap_values.return_value = shap_values
        data = ModelData(data_schema, x_test=x_test, y_test=None, raw_instances=NameError, y_pred=None)
        ctx = ModelContext(model_type="xgboost",
                           xai_methods=["shap_tree"],
                           dataset_id="dataset",
                           data=data,
                           model_display_name="model_display_name")
        result = adapter.explain_local_df(ctx, mock_model)
        assert result.shape == (9, 7)  # 3 instances * 3 classes, 7 features including class and instance_uuid

def test_multiclass_uses_class_names(adapter, mock_model, x_test, data_schema):
    shap_values = np.random.rand(3, 5, 3)
    class_names = ['cat_a', 'cat_b', 'cat_c']
    with patch(f"{MODULE_UNDER_TEST}.shap.TreeExplainer") as mock_explainer:
        mock_explainer.return_value.shap_values.return_value = shap_values
        data_schema.class_names = class_names
        data = ModelData(data_schema, x_test=x_test, y_test=None, raw_instances=NameError, y_pred=None)
        ctx = ModelContext(model_type="xgboost",
                           xai_methods=["shap_tree"],
                           dataset_id="dataset",
                           data=data,
                           model_display_name="model_display_name")
        result = adapter.explain_local_df(ctx, mock_model)
        assert sorted(result['class'].unique().tolist()) == sorted(class_names)

def test_multiclass_falls_back_to_integer_class_index(adapter, mock_model, x_test, data_schema):
    shap_values = np.random.rand(3, 5, 3)
    with patch(f"{MODULE_UNDER_TEST}.shap.TreeExplainer") as mock_explainer:
        mock_explainer.return_value.shap_values.return_value = shap_values
        data_schema.class_names = None
        data = ModelData(data_schema, x_test=x_test, y_test=None, raw_instances=NameError, y_pred=None)
        ctx = ModelContext(model_type="xgboost",
                           xai_methods=["shap_tree"],
                           dataset_id="dataset",
                           data=data,
                           model_display_name="model_display_name")
        result = adapter.explain_local_df(ctx, mock_model)
        assert sorted(result['class'].unique().tolist()) == [0, 1, 2]


def test_global_binary(adapter, mock_model, x_test, data_schema):
    shap_values = np.random.rand(3, 5)
    with patch(f"{MODULE_UNDER_TEST}.shap.TreeExplainer") as mock_explainer:
        mock_explainer.return_value.shap_values.return_value = shap_values
        data = ModelData(data_schema, x_test=x_test, y_test=None, raw_instances=NameError, y_pred=None)
        ctx = ModelContext(model_type="xgboost",
                           xai_methods=["shap_tree"],
                           dataset_id="dataset",
                           data=data,
                           model_display_name="model_display_name")
        result = adapter.explain_global_df(ctx, mock_model)
        assert 'feature' in result.columns
        assert 'importance' in result.columns


def test_global_multiclass(adapter, mock_model, x_test, data_schema):
    shap_values = np.random.rand(3, 5, 3)
    with patch(f"{MODULE_UNDER_TEST}.shap.TreeExplainer") as mock_explainer:
        mock_explainer.return_value.shap_values.return_value = shap_values
        data = ModelData(data_schema, x_test=x_test, y_test=None, raw_instances=NameError, y_pred=None)
        ctx = ModelContext(model_type="xgboost",
                           xai_methods=["shap_tree"],
                           dataset_id="dataset",
                           data=data,
                           model_display_name="model_display_name")
        result = adapter.explain_global_df(ctx, mock_model)
        assert result.shape == (15, 2)
