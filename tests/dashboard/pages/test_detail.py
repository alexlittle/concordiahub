import pytest
from pathlib import Path
from streamlit.testing.v1 import AppTest
from unittest.mock import patch, MagicMock

from xailens import constants
from xailens.entities.context import VizOption
from xailens.exceptions import InvalidConfigurationError
from xailens.dashboard import settings

PAGE_PATH = Path(__file__).parents[3] / f"xailens/dashboard/pages/{settings.PW_INSTANCE_FILENAME}.py"


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