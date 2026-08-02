"""
Tests for dashboard.utils.matrix_set.py

Covers:

- display of chart
- chart click throughs
"""
import pytest
from unittest.mock import MagicMock, patch

from xailens.dashboard.utils.matrix_set import render_matrixset_figure
MODULE = "xailens.dashboard.utils.matrix_set"

@pytest.fixture
def mock_st():
    """Patch streamlit"""
    with patch(f"{MODULE}.st") as st:
        st.query_params = {}
        st.plotly_chart.return_value = None
        yield st


@pytest.fixture
def clicked(mock_st):
    """Mock click function"""
    def _click(set_key):
        mock_st.plotly_chart.return_value = {
            "selection": {"points": [{"customdata": set_key}]}
        }
    return _click

@pytest.fixture
def dataset():
    """Mock dataset class"""
    ds = MagicMock()
    ds.get_matrixset_data.return_value = (MagicMock(), MagicMock())
    return ds


@pytest.fixture(autouse=True)
def stub_builders():
    with patch(f"{MODULE}.build_matrixset_figure"), \
         patch(f"{MODULE}.apply_matrixset_theme"):
        yield

def test_navigates_on_click(mock_st, clicked, dataset):
    """Tests switching page when a new set is selected."""
    mock_st.query_params["dataset"] = "cleveland"
    clicked("set_a")
    render_matrixset_figure(dataset)
    mock_st.switch_page.assert_called_once_with("pages/instances.py",
        query_params={"dataset": "cleveland", "matrix_set": "set_a"})


def test_updates_params_without_navigating(mock_st, clicked, dataset):
    """Tests updating the query param"""
    clicked("set_a")
    render_matrixset_figure(dataset, navigate_on_click=False)
    mock_st.switch_page.assert_not_called()
    assert mock_st.query_params["matrix_set"] == "set_a"


def test_no_action_when_set_already_selected(mock_st, clicked, dataset):
    """Tests does nothing when already in the current set"""
    mock_st.query_params["matrix_set"] = "set_a"
    clicked("set_a")
    render_matrixset_figure(dataset)
    mock_st.switch_page.assert_not_called()


def test_no_action_without_selection(mock_st, dataset):
    """Tests does nothing when the chart has no selection"""
    mock_st.plotly_chart.return_value = {"selection": {"points": []}}
    render_matrixset_figure(dataset)
    mock_st.switch_page.assert_not_called()


def test_no_action_when_customdata_missing(mock_st, dataset):
    """Tests points that have no set key are ignored"""
    mock_st.plotly_chart.return_value = {"selection": {"points": [{}]}}
    render_matrixset_figure(dataset)
    mock_st.switch_page.assert_not_called()