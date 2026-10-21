"""
Tests for ArtifactStore class

Covers tests for:

- Initial instantiation of ArtifactStore
- write_metadata
- write_model
- write_data
- write_explanation
- load_dataframe
- write_metrics

"""


import pytest
import numpy as np
import pandas as pd
import uuid

from unittest.mock import MagicMock
from concordiahub.artifacts import ArtifactStore
from concordiahub.entities.context import ModelContext, ModelData, DataSchema

MODULE_UNDER_TEST = "concordiahub.artifacts"

@pytest.fixture
def mock_store():
    return MagicMock()


@pytest.fixture
def ctx_with_model_dir_name():
    data_schema = DataSchema("sample_uuid", "target", "prediction")
    model_data = ModelData(data_schema,
                        raw_instances=pd.DataFrame({"a": [3]}),
                        x_test = pd.DataFrame({"a": [3]}),
                        y_test = pd.Series([0]),
                        y_pred = np.array([0]))

    ctx = ModelContext("xgboost", [], "test_dataset", model_data, "model disp name")
    ctx.model_dir_name = "my_model_name"
    ctx.model = MagicMock()
    ctx.model_display_name = "model_1"
    return ctx


@pytest.fixture
def ctx_without_model_dir_name():
    data_schema = DataSchema("sample_uuid", "target", "prediction")
    model_data = ModelData(data_schema,
                        raw_instances=pd.DataFrame({"a": [3]}),
                        x_test = pd.DataFrame({"a": [3]}),
                        y_test = pd.Series([0]),
                        y_pred = np.array([0]))

    ctx = ModelContext("xgboost", [], "test_dataset", model_data, "model disp name")
    ctx.model_dir_name = None
    ctx.model = None
    return ctx

@pytest.fixture
def metrics_series():
    """Classification metrics"""
    return pd.Series({
        "accuracy": 0.83,
        "f1_weighted": 0.81,
        "precision_weighted": 0.82,
        "recall_weighted": 0.83,
    })

def make_artifact_store(ctx, mock_store):
    store = ArtifactStore(ctx, store=mock_store)
    return store

#--------------------
# __init__ / _resolve_dirname
#----------------
def test_resolve_dirname_with_model_dir_name(ctx_with_model_dir_name, mock_store):
    """Test uses the dirname"""
    artifact_store = make_artifact_store(ctx_with_model_dir_name, mock_store)
    assert artifact_store.store is mock_store


def test_resolve_dirname_without_model_dir_name(ctx_without_model_dir_name, mock_store):
    """When model_dir_name is None, dirname should be a timestamped path."""
    artifact_store = make_artifact_store(ctx_without_model_dir_name, mock_store)
    assert artifact_store.store is mock_store

#--------------------
# write_metadata
#----------------
def test_write_metadata(ctx_with_model_dir_name, mock_store):
    """Test writing metadata"""
    artifact_store = make_artifact_store(ctx_with_model_dir_name, mock_store)
    artifact_store.write_metadata()
    mock_store.save_json.assert_called_once()
    saved = mock_store.save_json.call_args[0][1]
    assert saved["model_type"] == ctx_with_model_dir_name.model_type
    assert saved["model_dir_name"] == ctx_with_model_dir_name.model_dir_name
    assert "model_uuid" in saved
    uuid.UUID(saved["model_uuid"])

#--------------------
# write_model
#----------------
def test_write_model_with_model(ctx_with_model_dir_name, mock_store):
    """Test writing a model"""
    artifact_store = make_artifact_store(ctx_with_model_dir_name, mock_store)
    artifact_store.write_model()
    mock_store.save_model.assert_called_once()


def test_write_model_without_model(ctx_without_model_dir_name, mock_store):
    """Test doesn't write any model"""
    artifact_store = make_artifact_store(ctx_without_model_dir_name, mock_store)
    artifact_store.write_model()
    mock_store.save_model.assert_not_called()

#--------------------
# write_data
#----------------
def test_write_data_dataframe(ctx_with_model_dir_name, mock_store):
    """Test writing a dataframe"""
    artifact_store = make_artifact_store(ctx_with_model_dir_name, mock_store)
    artifact_store.write_data("x_test", ctx_with_model_dir_name.data.x_test)
    mock_store.save_dataframe.assert_called_once_with("x_test", ctx_with_model_dir_name.data.x_test)


def test_write_data_missing_attribute(ctx_with_model_dir_name, mock_store):
    """Should return without calling any save method."""
    artifact_store = make_artifact_store(ctx_with_model_dir_name, mock_store)
    artifact_store.write_data("nonexistent_field", None)
    mock_store.save_dataframe.assert_not_called()

def test_write_data_series(ctx_with_model_dir_name, mock_store):
    """Test writing a dataframe from series"""
    artifact_store = make_artifact_store(ctx_with_model_dir_name, mock_store)
    series = pd.Series([1, 0, 1, 0], name="prediction")

    assert artifact_store.write_data("y_pred", series) is True

    saved = mock_store.save_dataframe.call_args.args[1]
    assert isinstance(saved, pd.DataFrame)
    assert list(saved.columns) == ["prediction"]
    assert saved["prediction"].tolist() == [1, 0, 1, 0]

def test_write_data_ndarray(ctx_with_model_dir_name, mock_store):
    """Tests 2d ndarray to a DataFrame"""
    artifact_store = make_artifact_store(ctx_with_model_dir_name, mock_store)
    array = np.array([[1.0, 2.0], [3.0, 4.0]])

    assert artifact_store.write_data("scores", array) is True

    saved = mock_store.save_dataframe.call_args.args[1]
    assert saved.shape == (2, 2)
    assert saved.iloc[1, 0] == 3.0

def test_write_data_list(ctx_with_model_dir_name, mock_store):
    """Tests a flat list to a DataFrame."""
    artifact_store = make_artifact_store(ctx_with_model_dir_name, mock_store)

    artifact_store.write_data("y_pred", [1, 0, 1, 0])

    saved = mock_store.save_dataframe.call_args.args[1]
    assert saved.shape == (4, 1)

#--------------------
# write_metrics
#----------------
def test_write_metrics(ctx_with_model_dir_name, mock_store, metrics_series):
    """Test writes the metrics series to the store under the fixed 'metrics' key."""
    artifact_store = make_artifact_store(ctx_with_model_dir_name, mock_store)
    artifact_store.write_metrics(metrics_series)
    mock_store.save_series.assert_called_once_with("metrics", metrics_series)


# --------------------
# write_explanation
# ----------------
def test_write_explanation_dataframe(ctx_with_model_dir_name, mock_store):
    """Test writing an explanation dataframe"""
    artifact_store = make_artifact_store(ctx_with_model_dir_name, mock_store)
    df = pd.DataFrame({"feature": ["age"], "importance": [0.5]})
    artifact_store.write_explanation("exp_global_coefficients", df)
    mock_store.save_explanation.assert_called_once_with("exp_global_coefficients", df, "df")


def test_write_explanation_json(ctx_with_model_dir_name, mock_store):
    """Test writing a JSON explanation"""
    artifact_store = make_artifact_store(ctx_with_model_dir_name, mock_store)
    records = [{"instance_id": "inst_0", "reasoning": "age matters"}]
    artifact_store.write_explanation("exp_local_llm_reasoning", records, "json")
    mock_store.save_explanation.assert_called_once_with("exp_local_llm_reasoning", records, "json")


# --------------------
# load_dataframe
# ----------------
def test_load_dataframe(ctx_with_model_dir_name, mock_store):
    """Test loads a DataFrame from the store unchanged."""
    artifact_store = make_artifact_store(ctx_with_model_dir_name, mock_store)
    expected = pd.DataFrame({"instance_id": ["inst_0"], "target": [1]})
    mock_store.load_dataframe.side_effect = None
    mock_store.load_dataframe.return_value = expected

    result = artifact_store.load_dataframe("y_test")

    mock_store.load_dataframe.assert_called_once_with("y_test")
    assert result is expected
