import pytest

from xailens.dashboard.utils.render.breadcrumb import display_breadcrumbs
from xailens.entities.modeleval import ModelEval

@pytest.fixture
def modeleval_a():
    return ModelEval(metadata={
        "model_type": "XGBoost",
        "model_subtype": None,
        "xai_methods": ["SHAP"],
        "model_name": "run1",
        "datetime": "20240525-143000",
        "model_display_name": None
    })

@pytest.fixture
def modeleval_b():
    return ModelEval(metadata={
        "model_type": "LinReg",
        "model_subtype": None,
        "xai_methods": ["SHAP"],
        "model_name": "run1",
        "datetime": "20240525-143000",
        "model_display_name": None
    })

def test_no_params():
    breadcrumbs = display_breadcrumbs()
    assert "<a href='/' target='_self'>Home</a>" == breadcrumbs

def test_one_run(modeleval_a):
    breadcrumbs = display_breadcrumbs(modeleval_a)
    assert "<a href='/' target='_self'>Home</a>" == breadcrumbs

def test_two_runs(modeleval_a, modeleval_b):
    breadcrumbs = display_breadcrumbs(modeleval_a, modeleval_b)
    assert "LinReg" in breadcrumbs
    assert "XGBoost" in breadcrumbs
    assert "/overview?" not in breadcrumbs

def test_sample_id(modeleval_a, modeleval_b):
    breadcrumbs = display_breadcrumbs(modeleval_a, modeleval_b, instance_id="12345")
    assert "LinReg" in breadcrumbs
    assert "XGBoost" in breadcrumbs
    assert "/overview?" in breadcrumbs
    assert "12345" in breadcrumbs