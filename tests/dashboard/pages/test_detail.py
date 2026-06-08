import pytest
from pathlib import Path
from streamlit.testing.v1 import AppTest
from unittest.mock import patch, MagicMock

from xai_compare.context import VizOption

MODULE_UNDER_TEST = "xai_compare.dashboard.pages.detail"

PAGE_PATH = Path(__file__).parents[3] / "xai_compare/dashboard/pages/detail.py"


@patch("xai_compare.dashboard.utils.params.get_explorer")
def test_explorer_called_with_correct_ids(mock_get_explorer):
    mock_explorer = MagicMock()
    mock_get_explorer.return_value = mock_explorer
    at = AppTest.from_file(str(PAGE_PATH))
    at.query_params["runs"] = "uuid1,uuid2"
    at.run()
    mock_explorer.get_runs_by_uuids.assert_called_once_with(["uuid1", "uuid2"])




def make_run(model_id="model-a",
             model_type="xgboost",
             xai_methods=None,
             sample_id_column="id",
             run_id="run_a"):
    run = MagicMock()
    run.get = lambda key, default=None: {
        "model_id": model_id,
        "model_type": model_type,
        "xai_methods": xai_methods or ["shap_tree"],
        "sample_id_column": sample_id_column,
    }.get(key, default)
    return run

@pytest.fixture
def mock_runs():
    return make_run(model_id="model-a", run_id="run_a"), make_run(model_id="model-b", run_id="run_b")


@pytest.fixture
def patched_app(mock_runs):
    """Patch all external dependencies for complete page display"""
    run_a, run_b = mock_runs
    mock_fig = MagicMock()

    with patch("xai_compare.dashboard.utils.params.get_explorer") as mock_get_explorer, \
         patch("xai_compare.registry.xai_registry.get_xai_adapter") as mock_xai, \
         patch("xai_compare.registry.viz_registry.get_viz_adapter") as mock_viz, \
         patch("xai_compare.registry.comparison_registry.get_comparison_adapter") as mock_comp:

        mock_get_explorer.return_value.get_runs_by_uuids.return_value = [run_a, run_b]

        mock_xai_instance = MagicMock()
        mock_xai_instance.load_explanation.return_value = MagicMock()
        mock_xai_instance.get_abs_max.return_value = 0.5
        mock_xai_instance.default_visualisation = "bar"
        mock_xai.return_value = mock_xai_instance

        mock_viz_instance = MagicMock()
        mock_viz_instance.return_value.render_local.return_value = mock_fig
        mock_viz.return_value = mock_viz_instance

        mock_comp_class = MagicMock()
        mock_comp_instance = MagicMock()
        mock_comp_class.return_value = mock_comp_instance

        mock_comp_instance.get_visualisations_available.return_value = {
            "diff_bar": VizOption(display_name="Difference (bar)",
                                  method="viz_diff_bar")
        }
        mock_comp_instance.get_default_visualisation.return_value = "diff_bar"
        mock_comp_instance.get_abs_max.return_value = 0.5
        mock_comp_instance.render_comparison.return_value = mock_fig

        mock_comp.return_value = mock_comp_class

        yield {
            "mock_get_explorer": mock_get_explorer,
            "mock_xai": mock_xai,
            "mock_viz": mock_viz,
            "mock_comp": mock_comp,
        }

def test_no_runs_shows_error():
    with patch("xai_compare.dashboard.utils.params.get_explorer") as mock_get_explorer:
        mock_get_explorer.return_value.get_runs_by_uuids.return_value = []
        at = AppTest.from_file(str(PAGE_PATH))
        at.run()
        assert any("No runs selected" in e.value for e in at.error)

def test_no_runs_param_shows_error():
    with patch("xai_compare.dashboard.utils.params.get_explorer") as mock_get_explorer:
        mock_get_explorer.return_value.get_runs_by_uuids.return_value = []
        at = AppTest.from_file(str(PAGE_PATH))
        at.run()
        assert any(e.value for e in at.error)

def test_no_comparison_available_shows_warning(patched_app):
    from xai_compare.exceptions import InvalidConfigurationError
    mock_comp = patched_app["mock_comp"]
    mock_comp.return_value.return_value.render_comparison.side_effect = InvalidConfigurationError
    at = AppTest.from_file(str(PAGE_PATH))
    at.query_params["runs"] = "uuid1,uuid2"
    at.run()
    assert any("No comparison available" in w.value for w in at.warning)