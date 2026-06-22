from pathlib import Path
from streamlit.testing.v1 import AppTest
from unittest.mock import patch, MagicMock

MODULE_UNDER_TEST = "xailens.dashboard.pages.pairwise_overview"

PAGE_PATH = Path(__file__).parents[3] / "xailens/dashboard/pages/pairwise_overview.py"

def test_overview_no_runs():
    with patch("xailens.dashboard.services.get_explorer") as mock_explorer:
        mock_explorer.return_value.get_runs_by_uuids.return_value = []
        at = AppTest.from_file(str(PAGE_PATH))
        at.run()
        assert any("No runs selected" in e.value for e in at.error)


def test_overview_reads_query_params():
    at = AppTest.from_file(str(PAGE_PATH))
    at.query_params["runs"] = "a,b"
    at.run()
    assert len(at.warning) == 0


@patch("xailens.dashboard.utils.params.get_explorer")
def test_explorer_called_with_correct_ids(mock_get_explorer):
    mock_explorer = MagicMock()
    mock_get_explorer.return_value = mock_explorer
    at = AppTest.from_file(str(PAGE_PATH))
    at.query_params["runs"] = "uuid1,uuid2"
    at.run()
    mock_explorer.get_runs_by_uuids.assert_called_once_with(["uuid1", "uuid2"])


@patch("xailens.dashboard.utils.params.get_explorer")
@patch("xailens.services.comparison.build_comparison_df")
def test_build_comparison_df_called_with_runs(mock_build_comparison_df, mock_get_explorer):
    mock_run_a = MagicMock()
    mock_run_b = MagicMock()
    mock_explorer = MagicMock()
    mock_explorer.get_runs_by_uuids.return_value = [mock_run_a, mock_run_b]
    mock_get_explorer.return_value = mock_explorer
    mock_build_comparison_df.return_value = MagicMock()

    at = AppTest.from_file(str(PAGE_PATH))
    at.query_params["runs"] = "uuid1,uuid2"
    at.run()

    mock_build_comparison_df.assert_called_once_with(run_a=mock_run_a, run_b=mock_run_b)
