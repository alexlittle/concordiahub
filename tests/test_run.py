import pytest
import pandas as pd
from unittest.mock import MagicMock

from xailens.entities.modeleval import ModelEval

MODULE_UNDER_TEST = "xailens.run"

# -----------------
# Fixtures

@pytest.fixture
def mock_metadata():
    return {
        "run_name": "test_run",
        "model_type": "xgboost",
        "dataset_id": "heart",
        "data_schema": {
            "instance_id_column": "sample_uuid",
            "target_column": "target",
            "prediction_column": "y_pred"
        }
    }

@pytest.fixture
def store_with_fallback(mock_store, mock_metadata, y_test, y_pred):
    def load_side_effect(name):
        if name == "y_test":
            raise FileNotFoundError
        if name == "y":
            return y_test
        if name == "y_pred":
            return y_pred
    mock_store.load_dataframe.side_effect = load_side_effect
    return mock_store, mock_metadata

@pytest.fixture
def mock_store():
    return MagicMock()

@pytest.fixture
def explorer(mock_store):
    return ModelEval(store=mock_store, metadata=mock_metadata)

@pytest.fixture
def y_test():
    return pd.DataFrame({
        'sample_uuid': ['a', 'b', 'c', 'd'],
        'target': [0, 1, 0, 1]})

@pytest.fixture
def y_pred():
    return pd.DataFrame({
        'sample_uuid': ['a', 'b', 'c', 'd'],
        'y_pred': [0, 1, 0, 0]})

#--------------------
# Tests



