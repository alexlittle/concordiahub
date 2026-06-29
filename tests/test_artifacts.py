import pytest
import numpy as np
import pandas as pd
import uuid

from unittest.mock import MagicMock
from xailens.artifacts import ArtifactStore
from xailens.entities.context import ModelContext, ModelData, DataSchema

MODULE_UNDER_TEST = "xailens.artifacts"

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


def make_artifact_store(ctx, mock_store):
    store = ArtifactStore(ctx, store=mock_store)
    return store

#--------------------
# __init__ / _resolve_dirname

def test_resolve_dirname_with_model_dir_name(ctx_with_model_dir_name, mock_store):
    artifact_store = make_artifact_store(ctx_with_model_dir_name, mock_store)
    assert artifact_store.store is mock_store  # store was accepted


def test_resolve_dirname_without_model_dir_name(ctx_without_model_dir_name, mock_store):
    """When model_dir_name is None, dirname should be a timestamped path."""
    artifact_store = make_artifact_store(ctx_without_model_dir_name, mock_store)
    assert artifact_store.store is mock_store

#--------------------
# write_metadata

def test_write_metadata(ctx_with_model_dir_name, mock_store):
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

def test_write_model_with_model(ctx_with_model_dir_name, mock_store):
    artifact_store = make_artifact_store(ctx_with_model_dir_name, mock_store)
    artifact_store.write_model()
    mock_store.save_model.assert_called_once()


def test_write_model_without_model(ctx_without_model_dir_name, mock_store):
    artifact_store = make_artifact_store(ctx_without_model_dir_name, mock_store)
    artifact_store.write_model()
    mock_store.save_model.assert_not_called()

#--------------------
# write_data
def test_write_data_dataframe(ctx_with_model_dir_name, mock_store):
    artifact_store = make_artifact_store(ctx_with_model_dir_name, mock_store)
    artifact_store.write_data("x_test", ctx_with_model_dir_name.data.x_test)
    mock_store.save_dataframe.assert_called_once_with("x_test", ctx_with_model_dir_name.data.x_test)


def test_write_data_missing_attribute(ctx_with_model_dir_name, mock_store):
    """Should return without calling any save method."""
    artifact_store = make_artifact_store(ctx_with_model_dir_name, mock_store)
    artifact_store.write_data("nonexistent_field", None)
    mock_store.save_dataframe.assert_not_called()