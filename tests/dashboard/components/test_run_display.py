from unittest.mock import patch, MagicMock

from xai_compare.dashboard.components.run_display import format_datetime, render_run_html, display_run_cell

MODULE_UNDER_TEST = "xai_compare.dashboard.components.run_display"

def test_format_datetime():
    result = format_datetime("20240525-143000")
    assert result == "14:30 25 May 2024"


def test_render_run_html_basic():
    run = {
        "model_type": "RandomForest",
        "model_subtype": "Classifier",
        "xai_methods": ["SHAP", "LIME"],
        "run_name": "test_run",
        "datetime": "20240525-143000"
    }

    html = render_run_html(run)

    assert "RandomForest (Classifier)" in html
    assert "SHAP, LIME" in html
    assert "test_run" in html
    assert "14:30 25 May 2024" in html


def test_render_run_html_no_subtype():
    run = {
        "model_type": "XGBoost",
        "model_subtype": None,
        "xai_methods": ["SHAP"],
        "run_name": "run1",
        "datetime": "20240525-143000"
    }

    html = render_run_html(run)

    assert "XGBoost" in html
    assert "XGBoost (" not in html


def test_render_run_html_no_xai_methods():
    run = {
        "model_type": "SVM",
        "model_subtype": None,
        "xai_methods": None,
        "run_name": "run2",
        "datetime": "20240525-143000"
    }

    html = render_run_html(run)

    assert "<code></code>" in html


def test_display_run_cell_calls_markdown():
    col = MagicMock()

    run = {
        "model_type": "RF",
        "model_subtype": None,
        "xai_methods": ["SHAP"],
        "run_name": "run1",
        "datetime": "20240525-143000"
    }

    display_run_cell(col, run)

    col.markdown.assert_called_once()

    args, kwargs = col.markdown.call_args
    assert "RF" in args[0]
    assert kwargs["unsafe_allow_html"] is True


@patch("xai_compare.dashboard.components.run_display.st")
def test_display_comparable_runs_renders(mock_st):
    from xai_compare.dashboard.components.run_display import display_comparable_runs

    mock_col = MagicMock()
    mock_col.button.return_value = False

    mock_st.columns.return_value = [mock_col] * 5

    run_a = {
        "dataset_id": "ds1",
        "model_type": "RF",
        "model_subtype": None,
        "xai_methods": ["SHAP"],
        "run_name": "run1",
        "datetime": "20240525-143000",
        "run_uuid": "run1-uuid"
    }

    run_b = {
        "dataset_id": "ds1",
        "model_type": "XGB",
        "model_subtype": None,
        "xai_methods": ["LIME"],
        "run_name": "run2",
        "datetime": "20240525-150000",
        "run_uuid": "run2-uuid"
    }

    display_comparable_runs([[run_a, run_b]])

    assert mock_st.columns.called
    assert mock_st.divider.called
