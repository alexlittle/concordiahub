import pytest

from xai_compare.dashboard.components.breadcrumb import display_breadcrumbs
from xai_compare.run import Run

@pytest.fixture
def run_a():
    return Run(metadata={
        "model_type": "XGBoost",
        "model_subtype": None,
        "xai_methods": ["SHAP"],
        "run_name": "run1",
        "datetime": "20240525-143000",
        "model_id": None
    })

@pytest.fixture
def run_b():
    return Run(metadata={
        "model_type": "LinReg",
        "model_subtype": None,
        "xai_methods": ["SHAP"],
        "run_name": "run1",
        "datetime": "20240525-143000",
        "model_id": None
    })

def test_no_params():
    breadcrumbs = display_breadcrumbs()
    assert "<a href='/' target='_self'>Home</a>" == breadcrumbs

def test_one_run(run_a):
    breadcrumbs = display_breadcrumbs(run_a)
    assert "<a href='/' target='_self'>Home</a>" == breadcrumbs

def test_two_runs(run_a, run_b):
    breadcrumbs = display_breadcrumbs(run_a, run_b)
    assert "LinReg" in breadcrumbs
    assert "XGBoost" in breadcrumbs
    assert "/overview?runs" not in breadcrumbs

def test_sample_id(run_a, run_b):
    breadcrumbs = display_breadcrumbs(run_a, run_b, sample_id="12345")
    assert "LinReg" in breadcrumbs
    assert "XGBoost" in breadcrumbs
    assert "/overview?runs" in breadcrumbs
    assert "12345" in breadcrumbs