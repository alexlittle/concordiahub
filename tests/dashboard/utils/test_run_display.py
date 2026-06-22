from unittest.mock import patch, MagicMock

from xailens.dashboard.utils.run_display import render_modeleval_cell
from xailens.entities.modeleval import ModelEval

MODULE_UNDER_TEST = "xailens.dashboard.utils.run_display"


def test_render_run_html_basic():
    run = ModelEval(metadata={
        "model_type": "RandomForest",
        "model_subtype": "Classifier",
        "xai_methods": ["SHAP"],
        "run_name": "test_run",
        "datetime": "20240525-143000"
    })

    html = render_modeleval_cell(run)

    assert "RandomForest" in html
    assert "14:30 25 May 2024" in html


def test_render_run_html_no_subtype():
    run = ModelEval(metadata={
        "model_type": "XGBoost",
        "model_subtype": None,
        "xai_methods": ["SHAP"],
        "run_name": "run1",
        "datetime": "20240525-143000"
    })

    html = render_modeleval_cell(run)

    assert "XGBoost" in html
    assert "XGBoost (" not in html


def test_render_run_html_no_xai_methods():
    run = ModelEval(metadata={
        "model_type": "SVM",
        "model_subtype": None,
        "xai_methods": None,
        "run_name": "run2",
        "datetime": "20240525-143000"
    })

    html = render_modeleval_cell(run)

    assert "run-cell" in html



def test_display_run_cell_calls_markdown():

    run = ModelEval(metadata={
        "model_type": "RF",
        "model_subtype": None,
        "xai_methods": ["SHAP"],
        "run_name": "run1",
        "datetime": "20240525-143000"
    })

    html = render_modeleval_cell(run)

    assert "run-cell" in html


@patch("xailens.dashboard.utils.run_display.st")
def test_display_comparable_runs_renders(mock_st):
    from xailens.dashboard.utils.run_display import display_comparable_runs

    mock_col = MagicMock()
    mock_col.button.return_value = False

    mock_st.columns.return_value = [mock_col] * 5

    run_a = ModelEval(metadata={
        "dataset_id": "ds1",
        "model_type": "RF",
        "model_subtype": None,
        "xai_methods": ["SHAP"],
        "run_name": "run1",
        "datetime": "20240525-143000",
        "model_uuid": "run1-uuid"
    })

    run_b = ModelEval(metadata={
        "dataset_id": "ds1",
        "model_type": "XGB",
        "model_subtype": None,
        "xai_methods": ["LIME"],
        "run_name": "run2",
        "datetime": "20240525-150000",
        "runmodel_uuid_uuid": "run2-uuid"
    })

    html = display_comparable_runs([[run_a, run_b]])

    assert "run-card-header" in html
