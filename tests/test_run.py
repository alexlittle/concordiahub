import pytest
import pandas as pd
from unittest.mock import MagicMock

from xailens.run import Run

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
            "sample_id_column": "sample_uuid",
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
    return Run(store=mock_store, metadata=mock_metadata)

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

# Accuracy
def test_get_accuracy_score_falls_back_to_y(y_test, y_pred, store_with_fallback):
    store, metadata = store_with_fallback
    run = Run(store=store, metadata=metadata)
    assert run.accuracy() == 0.75

# Confusion matrix
def test_get_confusion_matrix_falls_back_to_y(y_test, y_pred, store_with_fallback):
    store, metadata = store_with_fallback
    run = Run(store=store, metadata=metadata)
    cm = run.confusion_matrix()
    assert cm.shape == (2, 2)

# Classification report
def test_get_classification_report_falls_back_to_y(y_test, y_pred, store_with_fallback):
    store, metadata = store_with_fallback
    run = Run(store=store, metadata=metadata)
    report = run.classification_report()
    assert "precision" in report
    assert "recall" in report

