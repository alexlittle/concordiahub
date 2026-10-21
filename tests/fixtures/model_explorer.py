"""
Reusable pytest fixtures for concordiahub.entities.model_explorer.ModelExplorer

"""

import pytest
from unittest.mock import MagicMock


@pytest.fixture
def mock_explorer_factory(lr_model_factory, xgboost_model_factory, llm_model_factory):
    """Fixture for building a ModelExplorer factory"""
    def _factory(dataset_id="cleveland_heart_disease",
                 datasets=None,
                 models=None,
                 valid=True):
        if models is None:
            models = [
                lr_model_factory(),
                xgboost_model_factory(),
                llm_model_factory(),
            ]
        if datasets is None:
            datasets = [dataset_id]

        explorer = MagicMock(name="ModelExplorer")
        explorer.is_valid_dataset.return_value = valid
        explorer.get_datasets.return_value = datasets
        explorer.get_models_for_dataset.return_value = models
        explorer.get_selected_models_for_dataset.return_value = models
        return explorer

    return _factory


@pytest.fixture
def mock_explorer(mock_explorer_factory):
    return mock_explorer_factory()