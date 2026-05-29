from pathlib import Path
from streamlit.testing.v1 import AppTest
from unittest.mock import patch, MagicMock

MODULE_UNDER_TEST = "xai_compare.dashboard.pages.detail"

PAGE_PATH = Path(__file__).parents[3] / "xai_compare/dashboard/pages/detail.py"

def test_detail_no_runs():
    at = AppTest.from_file(str(PAGE_PATH))
    at.run()
    assert any("No runs selected" in w.value for w in at.warning)


def test_detail_reads_query_params():
    at = AppTest.from_file(str(PAGE_PATH))
    at.query_params["runs"] = "a,b"
    at.run()
    assert len(at.warning) == 0


@patch("xai_compare.dashboard.services.get_explorer")
def test_explorer_called_with_correct_ids(mock_get_explorer):
    mock_explorer = MagicMock()
    mock_get_explorer.return_value = mock_explorer
    at = AppTest.from_file(str(PAGE_PATH))
    at.query_params["runs"] = "uuid1,uuid2"
    at.run()
    mock_explorer.get_runs_by_uuids.assert_called_once_with(["uuid1", "uuid2"])


