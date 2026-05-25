import pytest
import pandas as pd
from unittest.mock import MagicMock

from xai_compare.run import Run

MODULE_UNDER_TEST = "xai_compare.run"

# -----------------
# Fixtures
@pytest.fixture
def store_with_fallback(mock_store, y_test, y_pred):
    def load_side_effect(name):
        if name == "y_test":
            raise FileNotFoundError
        if name == "y":
            return y_test
        if name == "y_pred":
            return y_pred
    mock_store.load_dataframe.side_effect = load_side_effect
    return mock_store

@pytest.fixture
def mock_store():
    return MagicMock()

@pytest.fixture
def explorer(mock_store):
    return Run(store=mock_store)

@pytest.fixture
def y_test():
    return pd.DataFrame({'target': [0, 1, 0, 1]})

@pytest.fixture
def y_pred():
    return pd.DataFrame({'y_pred': [0, 1, 0, 0]})

#--------------------
# Tests

# Accuracy
def test_get_accuracy_score_falls_back_to_y(y_test, y_pred, store_with_fallback):
    run = Run(store=store_with_fallback)
    assert run.accuracy() == 0.75

# Confusion matrix
def test_get_confusion_matrix_falls_back_to_y(y_test, y_pred, store_with_fallback):
    run = Run(store=store_with_fallback)
    cm = run.confusion_matrix()
    assert cm.shape == (2, 2)

# Classification report
def test_get_classification_report_falls_back_to_y(y_test, y_pred, store_with_fallback):
    run = Run(store=store_with_fallback)
    report = run.classification_report()
    assert "precision" in report
    assert "recall" in report

