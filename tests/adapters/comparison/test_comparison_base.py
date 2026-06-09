import pytest

from xailens.adapters.comparison.comparison_base import ComparisonAdapter
from tests.helpers import make_explanation

MODULE_UNDER_TEST = "xailens.adapters.comparison.comparison_base"



def test_returns_correct_abs_max():
    explanation_a = make_explanation({"feat1": 0.5, "feat2": 0.2})
    explanation_b = make_explanation({"feat1": 0.2, "feat2": 0.5})
    result = ComparisonAdapter.get_abs_max(explanation_a, explanation_b)
    assert result == pytest.approx(3/7)

def test_returns_float():
    explanation_a = make_explanation({"feat1": 0.5, "feat2": 0.2})
    explanation_b = make_explanation({"feat1": 0.2, "feat2": 0.5})
    result = ComparisonAdapter.get_abs_max(explanation_a, explanation_b)
    assert isinstance(result, float)

def test_sample_id_column_excluded():
    """sample_id and class columns are not present in the new DataFrame structure"""
    explanation_a = make_explanation({"feat1": 0.5, "feat2": 0.2})
    explanation_b = make_explanation({"feat1": 0.5, "feat2": 0.2})
    result = ComparisonAdapter.get_abs_max(explanation_a, explanation_b)
    assert result == pytest.approx(0.0)

def test_identical_explanations_returns_zero():
    explanation_a = make_explanation({"feat1": 0.5, "feat2": 0.3})
    explanation_b = make_explanation({"feat1": 0.5, "feat2": 0.3})
    result = ComparisonAdapter.get_abs_max(explanation_a, explanation_b)
    assert result == pytest.approx(0.0)

def test_missing_feature_in_one_explanation():
    """fill_value=0 means a missing feature should be treated as zero"""
    explanation_a = make_explanation({"feat1": 0.5, "feat2": 0.5})
    explanation_b = make_explanation({"feat1": 0.5})
    result = ComparisonAdapter.get_abs_max(explanation_a, explanation_b)
    assert result > 0