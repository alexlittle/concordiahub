"""
Reusable pytest fixtures for concordiahub.entities.dataset.Dataset.

"""
import pytest

from concordiahub.entities.dataset import Dataset


@pytest.fixture
def heart_disease_dataset_factory(lr_model_factory, xgboost_model_factory,
                                   llm_model_factory):
    """
    Factory for a Dataset built from the three Cleveland-heart-disease model
    fixtures.
    """
    def _factory(dataset_id="cleveland_heart_disease",
                 lr_kwargs=None, xgboost_kwargs=None, llm_kwargs=None,
                 models=None):
        if models is None:
            models = [
                lr_model_factory(**(lr_kwargs or {})),
                xgboost_model_factory(**(xgboost_kwargs or {})),
                llm_model_factory(**(llm_kwargs or {})),
            ]
        return Dataset(dataset_id=dataset_id, models=models)

    return _factory


@pytest.fixture
def heart_disease_dataset(heart_disease_dataset_factory):
    return heart_disease_dataset_factory()