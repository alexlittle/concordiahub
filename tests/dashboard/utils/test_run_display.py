from unittest.mock import patch, MagicMock

from xailens.dashboard.utils.run_display import render_modeleval_cell
from xailens.entities.modeleval import ModelEval

MODULE_UNDER_TEST = "xailens.dashboard.utils.run_display"


def test_render_modeleval_html_basic():
    modeleval = ModelEval(metadata={
        "model_type": "RandomForest",
        "model_subtype": "Classifier",
        "xai_methods": ["SHAP"],
        "model_dir_name": "test_run",
        "model_display_name": "test_run",
        "datetime": "20240525-143000"
    })

    html = render_modeleval_cell(modeleval)

    assert "test_run" in html


def test_render_run_html_no_subtype():
    run = ModelEval(metadata={
        "model_type": "XGBoost",
        "model_subtype": None,
        "xai_methods": ["SHAP"],
        "model_dir_name": "run1",
        "model_display_name": "XGBoost",
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
        "model_dir_name": "run2",
        "model_display_name": "SVM",
        "datetime": "20240525-143000"
    })

    html = render_modeleval_cell(run)

    assert "run-cell" in html



def test_display_run_cell_calls_markdown():

    run = ModelEval(metadata={
        "model_type": "RF",
        "model_subtype": None,
        "xai_methods": ["SHAP"],
        "model_dir_name": "run1",
        "model_display_name": "RF",
        "datetime": "20240525-143000"
    })

    html = render_modeleval_cell(run)

    assert "run-cell" in html


@patch("xailens.dashboard.utils.run_display.st")
def test_display_comparable_models_renders(mock_st):
    from xailens.dashboard.utils.run_display import display_comparable_runs

    mock_col = MagicMock()
    mock_col.button.return_value = False

    mock_st.columns.return_value = [mock_col] * 5

    modeleval_a = ModelEval(metadata={
        "dataset_id": "ds1",
        "model_type": "RF",
        "model_subtype": None,
        "xai_methods": ["SHAP"],
        "model_dir_name": "run1",
        "model_display_name": "RF",
        "datetime": "20240525-143000",
        "model_uuid": "run1-uuid"
    })

    modeleval_b = ModelEval(metadata={
        "dataset_id": "ds1",
        "model_type": "XGB",
        "model_subtype": None,
        "xai_methods": ["LIME"],
        "model_dir_name": "run2",
        "model_display_name": "XGB",
        "datetime": "20240525-150000",
        "runmodel_uuid_uuid": "run2-uuid"
    })

    html = display_comparable_runs([[modeleval_a, modeleval_b]])

    assert "run-card-header" in html
