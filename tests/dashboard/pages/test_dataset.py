from pathlib import Path
from streamlit.testing.v1 import AppTest

from xailens import strings

PAGE = Path(__file__).parents[3] / "xailens" / "dashboard" / "pages" / "dataset.py"

def test_dataset_page_loads_overview(monkeypatch, mock_explorer):
    monkeypatch.setattr(
        "xailens.dashboard.explorer.get_explorer",
        lambda: mock_explorer
    )

    at = AppTest.from_file(str(PAGE))
    at.query_params["dataset"] = "cleveland_heart_disease"
    at.query_params["models"] = "Logistic Regression,XGBoost,LLM (CoT)"
    at.run()

    assert not at.exception
    assert not at.error

def test_dataset_page_loads_metrics(monkeypatch, mock_explorer):
    monkeypatch.setattr(
        "xailens.dashboard.explorer.get_explorer",
        lambda: mock_explorer
    )

    at = AppTest.from_file(str(PAGE))
    at.query_params["dataset"] = "cleveland_heart_disease"
    at.query_params["models"] = "Logistic Regression,XGBoost,LLM (CoT)"
    at.query_params["ds_tab"] = strings.DATASET_TAB_MORE_METRICS
    at.run()

    assert not at.exception
    assert not at.error


def test_dataset_page_loads_consensus(monkeypatch, mock_explorer):
    monkeypatch.setattr(
        "xailens.dashboard.explorer.get_explorer",
        lambda: mock_explorer
    )

    at = AppTest.from_file(str(PAGE))
    at.query_params["dataset"] = "cleveland_heart_disease"
    at.query_params["models"] = "Logistic Regression,XGBoost,LLM (CoT)"
    at.query_params["ds_tab"] = strings.DATASET_TAB_INSTANCE_CONSENSUS
    at.run()

    assert not at.exception
    assert not at.error

def test_dataset_page_invalid_dataset(monkeypatch, mock_explorer):
    monkeypatch.setattr(
        "xailens.dashboard.pages.dataset.get_explorer",
        lambda: mock_explorer
    )

    at = AppTest.from_file(str(PAGE))
    at.query_params["dataset"] = "some_dataset"
    at.run()

    assert not at.exception
