"""
Tests for Streamlit dataset page

Covers:
- overview tab
- metrics tab
- consensus tab
- tab session/url state
"""

import pytest
from unittest.mock import patch
from pathlib import Path
from streamlit.testing.v1 import AppTest

from concordiahub import strings
from concordiahub.dashboard.pages.dataset import _sync_tab

PAGE = Path(__file__).parents[3] / "concordiahub" / "dashboard" / "pages" / "dataset.py"

@pytest.fixture
def mock_st():
    with patch("concordiahub.dashboard.pages.dataset.st") as st:
        st.session_state = {}
        st.query_params = {}
        yield st


def test_dataset_page_loads_overview(monkeypatch, mock_explorer):
    """Test loads page with overview tab"""
    monkeypatch.setattr(
        "concordiahub.dashboard.explorer.get_explorer",
        lambda: mock_explorer
    )

    at = AppTest.from_file(str(PAGE))
    at.query_params["dataset"] = "cleveland_heart_disease"
    at.query_params["models"] = "Logistic Regression,XGBoost,LLM"
    at.run()

    assert not at.exception
    assert not at.error

def test_dataset_page_loads_metrics(monkeypatch, mock_explorer):
    """Test loads page with metrics tab"""
    monkeypatch.setattr(
        "concordiahub.dashboard.explorer.get_explorer",
        lambda: mock_explorer
    )

    at = AppTest.from_file(str(PAGE))
    at.query_params["dataset"] = "cleveland_heart_disease"
    at.query_params["models"] = "Logistic Regression,XGBoost,LLM"
    at.query_params["ds_tab"] = strings.DATASET_TAB_MORE_METRICS
    at.run()

    assert not at.exception
    assert not at.error


def test_dataset_page_loads_consensus(monkeypatch, mock_explorer):
    """Test loads page with consensus tab"""
    monkeypatch.setattr(
        "concordiahub.dashboard.explorer.get_explorer",
        lambda: mock_explorer
    )

    at = AppTest.from_file(str(PAGE))
    at.query_params["dataset"] = "cleveland_heart_disease"
    at.query_params["models"] = "Logistic Regression,XGBoost,LLM"
    at.query_params["ds_tab"] = strings.DATASET_TAB_INSTANCE_CONSENSUS
    at.run()

    assert not at.exception
    assert not at.error

def test_dataset_page_invalid_dataset(monkeypatch, mock_explorer):
    """Test doesn't crash with invalid dataset"""
    monkeypatch.setattr(
        "concordiahub.dashboard.pages.dataset.get_explorer",
        lambda: mock_explorer
    )

    at = AppTest.from_file(str(PAGE))
    at.query_params["dataset"] = "some_dataset"
    at.run()

    assert not at.exception


# ----------------
# test for tabs
#------------------
def test_sync_tab_copies_selection_to_url_and_query_params(mock_st):
    """Mirrors the selected tab into the URL state and query params."""
    mock_st.session_state["ds_tab"] = strings.DATASET_TAB_OVERVIEW

    _sync_tab()

    assert mock_st.session_state["ds_tab"] == strings.DATASET_TAB_OVERVIEW
    assert mock_st.session_state["_ds_tab_url"] == strings.DATASET_TAB_OVERVIEW
    assert mock_st.query_params["ds_tab"] == strings.DATASET_TAB_OVERVIEW


def test_sync_tab_falls_back_to_overview_when_empty(mock_st):
    """Defaults to the overview tab"""
    mock_st.session_state["ds_tab"] = None

    _sync_tab()

    assert mock_st.session_state["ds_tab"] == strings.DATASET_TAB_OVERVIEW
    assert mock_st.query_params["ds_tab"] == strings.DATASET_TAB_OVERVIEW
