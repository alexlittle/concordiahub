from pathlib import Path
from streamlit.testing.v1 import AppTest

PAGE = Path(__file__).parents[3] / "xailens" / "dashboard" / "pages" / "instancedetail.py"

def test_instancedetail_page_loads(monkeypatch, mock_explorer):

    monkeypatch.setattr(
        "xailens.dashboard.explorer.get_explorer",
        lambda: mock_explorer
    )

    at = AppTest.from_file(str(PAGE))
    at.query_params["dataset"] = "cleveland_heart_disease"
    at.query_params["instances"] = "allcorrect"
    at.query_params["instance_uuid"] = "inst_0"
    at.run()

    assert not at.exception
    assert not at.error