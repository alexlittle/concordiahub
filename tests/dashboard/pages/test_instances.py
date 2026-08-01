"""
Tests for the instances page

Covers:


"""
from pathlib import Path
from unittest.mock import MagicMock

import pandas as pd
import pytest
from streamlit.testing.v1 import AppTest

from xailens import strings

PAGE = Path(__file__).parents[3] / "xailens" / "dashboard" / "pages" / "instances.py"


# ---------------------------------
# Mocks for render_matrixset_figure / interactive_table
# ---------------------------------

@pytest.fixture
def mock_render_matrixset_figure(monkeypatch):
    mock = MagicMock()
    monkeypatch.setattr(
        "xailens.dashboard.utils.matrix_set.render_matrixset_figure", mock)
    return mock


@pytest.fixture
def mock_interactive_table(monkeypatch):
    mock = MagicMock()
    monkeypatch.setattr("itables.streamlit.interactive_table", mock)
    return mock


# -----------------
# Dataset fixtures for specific agreement combinations
# ----------------
# Default lr/xgboost/llm_model_factory predictions already produce
# agreement == "disagree" for every instance so heart_disease_dataset_factory
# / mock_explorer covers the "disagree" and "all" cases
#
# This function creates instances dataframe for "allcorrect" / "allincorrect"

def _build_all_correct_incorrect(n_instances, correct):
    """
    Build dataframes so every model's prediction equals (or differs from) the
    target for every instance, i.e. agreement is "allcorrect" or "allincorrect"
    across the whole dataset.
    """
    ids = [f"inst_{i}" for i in range(n_instances)]
    target = [i % 2 for i in range(n_instances)]
    pred = target if correct else [1 - t for t in target]

    y_test = pd.DataFrame({"instance_id": ids, "target": target})
    y_pred = pd.DataFrame({"instance_id": ids, "prediction": pred})
    return y_test, y_pred


@pytest.fixture
def all_correct_dataset(
        mock_explorer_factory,
        lr_model_factory,
        xgboost_model_factory,
        llm_model_factory):
    y_test, y_pred = _build_all_correct_incorrect(n_instances=4, correct=True)
    llm_y_pred = y_pred.copy()
    llm_y_pred["confidence"] = 0.9
    models = [
        lr_model_factory(y_test=y_test, y_pred=y_pred),
        xgboost_model_factory(y_test=y_test, y_pred=y_pred),
        llm_model_factory(y_test=y_test, y_pred=llm_y_pred),
    ]
    return mock_explorer_factory(models=models)


@pytest.fixture
def all_incorrect_dataset(
        mock_explorer_factory,
        lr_model_factory,
        xgboost_model_factory,
        llm_model_factory):
    y_test, y_pred = _build_all_correct_incorrect(n_instances=4, correct=False)
    llm_y_pred = y_pred.copy()
    llm_y_pred["confidence"] = 0.9
    models = [
        lr_model_factory(y_test=y_test, y_pred=y_pred),
        xgboost_model_factory(y_test=y_test, y_pred=y_pred),
        llm_model_factory(y_test=y_test, y_pred=llm_y_pred),
    ]
    return mock_explorer_factory(models=models)


# ------------------
# matrix_set
# ------------------

def test_matrix_set_shows_plot_and_table(
        monkeypatch,
        mock_explorer,
        mock_render_matrixset_figure,
        mock_interactive_table):
    monkeypatch.setattr(
        "xailens.dashboard.explorer.get_explorer", lambda: mock_explorer
    )
    at = AppTest.from_file(str(PAGE))
    at.query_params["dataset"] = "cleveland_heart_disease"
    at.query_params["matrix_set"] = "Logistic Regression"
    at.run()

    assert not at.exception
    assert not at.error

    mock_render_matrixset_figure.assert_called_once()
    assert mock_render_matrixset_figure.call_args.kwargs.get("navigate_on_click") is False

    mock_interactive_table.assert_called_once()


# ---------------------------
# instance_group (non-"all")
# ---------------------------

def test_instance_group_disagree_shows_table_only(
        monkeypatch,
        mock_explorer,
        mock_render_matrixset_figure,
        mock_interactive_table):
    monkeypatch.setattr(
        "xailens.dashboard.explorer.get_explorer", lambda: mock_explorer
    )
    at = AppTest.from_file(str(PAGE))
    at.query_params["dataset"] = "cleveland_heart_disease"
    at.query_params["instances"] = "disagree"
    at.run()

    assert not at.exception
    assert not at.error

    mock_render_matrixset_figure.assert_not_called()
    mock_interactive_table.assert_called_once()

    displayed_df = mock_interactive_table.call_args.args[0]
    # default lr/xgboost/llm_model_factory fixtures use n_instances=5 and
    # agreement == "disagree" for every instance
    assert len(displayed_df) == 5


def test_instance_group_allcorrect_filters_table(
        monkeypatch,
        all_correct_dataset,
        mock_render_matrixset_figure,
        mock_interactive_table):
    monkeypatch.setattr(
        "xailens.dashboard.explorer.get_explorer",
        lambda: all_correct_dataset
    )
    at = AppTest.from_file(str(PAGE))
    at.query_params["dataset"] = "cleveland_heart_disease"
    at.query_params["instances"] = "allcorrect"
    at.run()

    assert not at.exception
    assert not at.error

    mock_render_matrixset_figure.assert_not_called()
    mock_interactive_table.assert_called_once()

    displayed_df = mock_interactive_table.call_args.args[0]
    # all 4 instances are allcorrect
    assert len(displayed_df) == 4


def test_instance_group_allincorrect_filters_table(
        monkeypatch,
        all_incorrect_dataset,
        mock_render_matrixset_figure,
        mock_interactive_table):
    monkeypatch.setattr(
        "xailens.dashboard.explorer.get_explorer",
        lambda: all_incorrect_dataset
    )
    at = AppTest.from_file(str(PAGE))
    at.query_params["dataset"] = "cleveland_heart_disease"
    at.query_params["instances"] = "allincorrect"
    at.run()

    assert not at.exception
    assert not at.error

    mock_render_matrixset_figure.assert_not_called()
    mock_interactive_table.assert_called_once()

    displayed_df = mock_interactive_table.call_args.args[0]
    assert len(displayed_df) == 4


# --------------------------
# instance_group == "all"
# ------------------------

def test_instance_group_all_shows_unfiltered_table(
        monkeypatch,
        mock_explorer,
        mock_render_matrixset_figure,
        mock_interactive_table):
    monkeypatch.setattr(
        "xailens.dashboard.explorer.get_explorer",
        lambda: mock_explorer
    )
    at = AppTest.from_file(str(PAGE))
    at.query_params["dataset"] = "cleveland_heart_disease"
    at.query_params["instances"] = "all"
    at.run()

    assert not at.exception
    assert not at.error

    mock_render_matrixset_figure.assert_not_called()
    mock_interactive_table.assert_called_once()


# -----------------------------------------------
# invalid combinations: neither / both params present
# -------------------------------------------

def test_no_params_shows_warning(
        monkeypatch,
        mock_explorer,
        mock_render_matrixset_figure,
        mock_interactive_table):
    monkeypatch.setattr(
        "xailens.dashboard.explorer.get_explorer",
        lambda: mock_explorer
    )
    at = AppTest.from_file(str(PAGE))
    at.query_params["dataset"] = "cleveland_heart_disease"
    at.run()

    assert not at.exception
    warnings = [w.value for w in at.warning]
    assert strings.WARNING_INVALID_URL_PARAMS in warnings
    mock_render_matrixset_figure.assert_called_once()
    mock_interactive_table.assert_not_called()


def test_both_params_shows_warning_no_plot(
        monkeypatch,
        mock_explorer,
        mock_render_matrixset_figure,
        mock_interactive_table):
    monkeypatch.setattr(
        "xailens.dashboard.explorer.get_explorer",
        lambda: mock_explorer
    )
    at = AppTest.from_file(str(PAGE))
    at.query_params["dataset"] = "cleveland_heart_disease"
    at.query_params["instances"] = "disagree"
    at.query_params["matrix_set"] = "Logistic Regression"
    at.run()

    assert not at.exception
    warnings = [w.value for w in at.warning]
    assert strings.WARNING_INVALID_URL_PARAMS in warnings

    mock_render_matrixset_figure.assert_not_called()
    mock_interactive_table.assert_not_called()


def test_matrix_set_shows_plot_and_table(
        monkeypatch,
        mock_explorer,
        mock_render_matrixset_figure,
        mock_interactive_table):
    """Test loading with specific models"""
    monkeypatch.setattr(
        "xailens.dashboard.explorer.get_explorer", lambda: mock_explorer
    )
    at = AppTest.from_file(str(PAGE))
    at.query_params["models"] = "XGBoost"
    at.query_params["dataset"] = "cleveland_heart_disease"
    at.query_params["matrix_set"] = "xgboost"
    at.run()

    assert not at.exception
    assert not at.error

    mock_render_matrixset_figure.assert_called_once()
    assert mock_render_matrixset_figure.call_args.kwargs.get("navigate_on_click") is False

    mock_interactive_table.assert_called_once()