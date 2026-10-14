"""
Tests for Streamlit instance detail page

Covers:
- overview tab
- explanations tab
- tab session/url state
"""
import pytest

from unittest.mock import patch
from pathlib import Path
from streamlit.testing.v1 import AppTest

from concordiahub import strings
from concordiahub.dashboard.pages.instancedetail import _sync_tab

PAGE = Path(__file__).parents[3] / "concordiahub" / "dashboard" / "pages" / "instancedetail.py"



@pytest.fixture
def mock_st():
    with patch("concordiahub.dashboard.pages.instancedetail.st") as st:
        st.session_state = {}
        st.query_params = {}
        yield st


def test_instancedetail_page_loads_default_tab(monkeypatch, mock_explorer):
    """Test loading default tab"""
    monkeypatch.setattr(
        "concordiahub.dashboard.explorer.get_explorer",
        lambda: mock_explorer
    )

    at = AppTest.from_file(str(PAGE))
    at.query_params["dataset"] = "cleveland_heart_disease"
    at.query_params["instances"] = "allcorrect"
    at.query_params["instance_uuid"] = "inst_0"
    at.run()

    assert not at.exception
    assert not at.error

def test_instancedetail_page_with_models(monkeypatch, mock_explorer):
    """Test loading default tab but with specific models"""
    monkeypatch.setattr(
        "concordiahub.dashboard.explorer.get_explorer",
        lambda: mock_explorer
    )

    at = AppTest.from_file(str(PAGE))
    at.query_params["models"] = "XGBoost"
    at.query_params["dataset"] = "cleveland_heart_disease"
    at.query_params["instances"] = "allcorrect"
    at.query_params["instance_uuid"] = "inst_0"
    at.run()

    assert not at.exception
    assert not at.error


def test_instancedetail_page_loads_overview_tab(monkeypatch, mock_explorer):
    """Test loading overview tab"""
    monkeypatch.setattr(
        "concordiahub.dashboard.explorer.get_explorer",
        lambda: mock_explorer
    )

    at = AppTest.from_file(str(PAGE))
    at.query_params["dataset"] = "cleveland_heart_disease"
    at.query_params["instances"] = "allcorrect"
    at.query_params["instance_uuid"] = "inst_0"
    at.query_params["detail_tab"] = strings.INSTANCE_DETAIL_TAB_OVERVIEW
    at.run()

    assert not at.exception
    assert not at.error

def test_instancedetail_page_loads_explanations_tab(monkeypatch, mock_explorer):
    """Test loading explanations tab"""
    monkeypatch.setattr(
        "concordiahub.dashboard.explorer.get_explorer",
        lambda: mock_explorer
    )

    at = AppTest.from_file(str(PAGE))
    at.query_params["dataset"] = "cleveland_heart_disease"
    at.query_params["instances"] = "allcorrect"
    at.query_params["instance_uuid"] = "inst_0"
    at.query_params["detail_tab"] = strings.INSTANCE_DETAIL_MODEL_EXPLANATIONS
    at.run()

    assert not at.exception
    assert not at.error

# ----------------
# test for tabs
#------------------
def test_sync_tab_copies_selection_to_url_and_query_params(mock_st):
    """Mirrors the selected tab into the URL state and query params."""
    mock_st.session_state["detail_tab"] = strings.INSTANCE_DETAIL_TAB_OVERVIEW

    _sync_tab()

    assert mock_st.session_state["detail_tab"] == strings.INSTANCE_DETAIL_TAB_OVERVIEW
    assert mock_st.session_state["_detail_tab_url"] == strings.INSTANCE_DETAIL_TAB_OVERVIEW
    assert mock_st.query_params["detail_tab"] == strings.INSTANCE_DETAIL_TAB_OVERVIEW


def test_sync_tab_falls_back_to_overview_when_empty(mock_st):
    """Defaults to the overview tab"""
    mock_st.session_state["detail_tab"] = None

    _sync_tab()

    assert mock_st.session_state["detail_tab"] == strings.INSTANCE_DETAIL_TAB_OVERVIEW
    assert mock_st.query_params["detail_tab"] == strings.INSTANCE_DETAIL_TAB_OVERVIEW