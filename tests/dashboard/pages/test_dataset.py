from pathlib import Path
from streamlit.testing.v1 import AppTest

PAGE = Path(__file__).parents[3] / "xailens" / "dashboard" / "pages" / "dataset.py"

def test_dataset_page_loads(monkeypatch, mock_explorer):
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


def test_dataset_page_invalid_dataset(monkeypatch, mock_explorer):
    monkeypatch.setattr(
        "xailens.dashboard.pages.dataset.get_explorer",
        lambda: mock_explorer
    )

    at = AppTest.from_file(str(PAGE))
    at.query_params["dataset"] = "some_dataset"
    at.run()

    assert not at.exception
