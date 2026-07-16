import pytest
from unittest.mock import MagicMock


@pytest.fixture
def mock_explorer_factory(lr_model_factory,
                          xgboost_model_factory,
                          llm_model_factory):
    def _factory(dataset_id="cleveland_heart_disease",
                 models=None,
                 valid=True):
        if models is None:
            models = [
                lr_model_factory(),
                xgboost_model_factory(),
                llm_model_factory(),
            ]
        model_names = [m.metadata["model_display_name"] for m in models]

        explorer = MagicMock()
        explorer.is_valid_dataset.return_value = valid
        explorer.get_models_for_dataset.return_value = models
        explorer.get_selected_models_for_dataset.return_value = models
        explorer.get_models_for_dataset.return_value = [
            {"model_display_name": name} for name in model_names
        ]
        return explorer
    return _factory


@pytest.fixture
def mock_explorer(mock_explorer_factory):
    return mock_explorer_factory()