import pytest
from pathlib import Path
from streamlit.testing.v1 import AppTest
from unittest.mock import patch, MagicMock

from xailens import constants
from xailens.entities.context import VizOption
from xailens.exceptions import InvalidConfigurationError

PAGE_PATH = Path(__file__).parents[3] / "xailens/dashboard/pages/pairwise_instance.py"


def make_model(model_display_name="model-a", model_type="xgboost", xai_methods=None):
    model = MagicMock()
    model.model_display_name = model_display_name
    model.model_type = model_type
    model.xai_methods = xai_methods or ["shap_tree"]
    model.get.side_effect = lambda key, default=None: {
        "model_display_name": model_display_name,
        "model_type": model_type,
        "xai_methods": model.xai_methods,
        constants.DATA_SCHEMA_ID_COLUMN: "id",
    }.get(key, default)
    return model


@pytest.fixture
def mock_models():
    return make_model(model_display_name="model-a"), make_model(model_display_name="model-b")



# ---------
# Tests
# ---------

def test_no_runs_shows_error():
    with patch("xailens.dashboard.utils.params.get_explorer") as mock_get_explorer:
        mock_get_explorer.return_value.get_modelevals_by_uuids.return_value = []
        at = AppTest.from_file(str(PAGE_PATH))
        at.run()
        assert any("No model_data selected" in e.value for e in at.error)

def test_no_runs_param_shows_error():
    with patch("xailens.dashboard.utils.params.get_explorer") as mock_get_explorer:
        mock_get_explorer.return_value.get_modelevals_by_uuids.return_value = []
        at = AppTest.from_file(str(PAGE_PATH))
        at.run()
        assert any(e.value for e in at.error)


def test_no_comparison_available_shows_warning(mock_models):
    modeleval_a, modeleval_b = mock_models

    mock_panel_a, mock_panel_b = MagicMock(), MagicMock()
    mock_panel_a.explanation = "mock_explanation_a"
    mock_panel_b.explanation = "mock_explanation_b"

    mock_comp_instance = MagicMock()
    mock_comp_instance.get_visualisations_available.return_value = {
        "diff_bar": VizOption(display_name="Difference (bar)", method="viz_diff_bar")
    }
    mock_comp_instance.get_default_visualisation.return_value = "diff_bar"
    mock_comp_instance.render_comparison.side_effect = InvalidConfigurationError("no adapter")

    mock_comp_class = MagicMock()
    mock_comp_class.get_abs_max.return_value = 1.0
    mock_comp_class.return_value = mock_comp_instance

    mock_explorer = MagicMock()
    mock_explorer.get_runs_by_uuids.return_value = [modeleval_a, modeleval_b]

    with patch("xailens.dashboard.utils.params.get_explorer", return_value=mock_explorer), \
         patch("xailens.dashboard.utils.detail_helpers.build_model_panel",
               side_effect=[mock_panel_a, mock_panel_b]), \
         patch("xailens.dashboard.utils.detail_helpers.render_model_panel"), \
         patch("xailens.dashboard.utils.detail_helpers.select_comparison_viz",
               return_value="<p>viz</p>"), \
         patch("xailens.dashboard.utils.render.breadcrumb.display_breadcrumbs",
               return_value="<p>breadcrumb</p>"), \
         patch("xailens.dashboard.utils.layout.apply_layout"), \
         patch("xailens.registry.comparison_registry.get_comparison_adapter",
               return_value=mock_comp_class), \
         patch("xailens.entities.context.ComparisonContext"):

        at = AppTest.from_file(str(PAGE_PATH))
        at.query_params["model_data"] = "uuid1,uuid2"
        at.query_params["instance_id"] = "some_instance_id"
        at.run()

    assert any("No comparison available" in w.value for w in at.warning)