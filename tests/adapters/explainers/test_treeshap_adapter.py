import pytest
import numpy as np
import pandas as pd
from unittest.mock import MagicMock, patch
from xai_compare.adapters.explainers.treeshap_adapter import TreeSHAPAdapter
from xai_compare.context import RunData, DataSchema
from xai_compare.exceptions import MissingDataError

MODULE_UNDER_TEST = "xai_compare.adapters.explainers.treeshap_adapter"

@pytest.fixture
def adapter():
    return TreeSHAPAdapter()

@pytest.fixture
def sample_uuids():
    return ['uuid-001', 'uuid-002', 'uuid-003']

@pytest.fixture
def data_schema():
    return DataSchema("sample_uuid", "target")

@pytest.fixture
def x_test(sample_uuids):
    return pd.DataFrame(
        np.random.rand(3, 5),
        columns=['f1', 'f2', 'f3', 'f4', 'f5'],
        index=pd.Index(sample_uuids, name='sample_uuid')
    )

@pytest.fixture
def mock_model():
    return MagicMock()

def test_raises_missing_data_error_when_x_test_is_none(adapter, mock_model, data_schema):
    data = RunData(data_schema, x_test=None)
    with pytest.raises(MissingDataError):
        adapter.explain_local_df(data, mock_model)

def test_binary_returns_dataframe(adapter, mock_model, x_test, data_schema):
    shap_values = np.random.rand(3, 5)  # (n_samples, n_features)
    with patch(f"{MODULE_UNDER_TEST}.shap.TreeExplainer") as mock_explainer:
        mock_explainer.return_value.shap_values.return_value = shap_values
        data = RunData(data_schema, x_test=x_test)
        result = adapter.explain_local_df(data, mock_model)
        assert isinstance(result, pd.DataFrame)

def test_binary_has_class_index(adapter, mock_model, x_test, data_schema):
    shap_values = np.random.rand(3, 5)
    with patch(f"{MODULE_UNDER_TEST}.shap.TreeExplainer") as mock_explainer:
        mock_explainer.return_value.shap_values.return_value = shap_values
        data = RunData(data_schema, x_test=x_test)
        result = adapter.explain_local_df(data, mock_model)
        assert 'class' in result.index.names

def test_multiclass_returns_dataframe(adapter, mock_model, x_test, data_schema):
    shap_values = np.random.rand(3, 5, 3)  # (n_samples, n_features, n_classes)
    with patch(f"{MODULE_UNDER_TEST}.shap.TreeExplainer") as mock_explainer:
        mock_explainer.return_value.shap_values.return_value = shap_values
        data = RunData(data_schema, x_test=x_test)
        result = adapter.explain_local_df(data, mock_model)
        assert isinstance(result, pd.DataFrame)

def test_multiclass_has_correct_shape(adapter, mock_model, x_test, data_schema):
    shap_values = np.random.rand(3, 5, 3)
    with patch(f"{MODULE_UNDER_TEST}.shap.TreeExplainer") as mock_explainer:
        mock_explainer.return_value.shap_values.return_value = shap_values
        data = RunData(data_schema, x_test=x_test)
        result = adapter.explain_local_df(data, mock_model)
        assert result.shape == (9, 5)  # 3 samples * 3 classes, 5 features

def test_multiclass_uses_class_names(adapter, mock_model, x_test, data_schema):
    shap_values = np.random.rand(3, 5, 3)
    class_names = ['cat_a', 'cat_b', 'cat_c']
    with patch(f"{MODULE_UNDER_TEST}.shap.TreeExplainer") as mock_explainer:
        mock_explainer.return_value.shap_values.return_value = shap_values
        data_schema.class_names = class_names
        data = RunData(data_schema, x_test=x_test)
        result = adapter.explain_local_df(data, mock_model)
        assert result.index.get_level_values('class').unique().tolist() == class_names

def test_multiclass_falls_back_to_integer_class_index(adapter, mock_model, x_test, data_schema):
    shap_values = np.random.rand(3, 5, 3)
    with patch(f"{MODULE_UNDER_TEST}.shap.TreeExplainer") as mock_explainer:
        mock_explainer.return_value.shap_values.return_value = shap_values
        data_schema.class_names = None
        data = RunData(data_schema, x_test=x_test)
        result = adapter.explain_local_df(data, mock_model)
        assert result.index.get_level_values('class').unique().tolist() == [0, 1, 2]

def test_uuid_preserved_in_index(adapter, mock_model, x_test, sample_uuids, data_schema):
    shap_values = np.random.rand(3, 5, 3)
    with patch(f"{MODULE_UNDER_TEST}.shap.TreeExplainer") as mock_explainer:
        mock_explainer.return_value.shap_values.return_value = shap_values
        data = RunData(data_schema, x_test=x_test)
        result = adapter.explain_local_df(data, mock_model)
        assert all(uuid in result.index.get_level_values('sample_uuid') for uuid in sample_uuids)