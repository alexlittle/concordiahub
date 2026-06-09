import pytest
import pandas as pd

from xailens.adapters.comparison.comparison_base import ComparisonAdapter

MODULE_UNDER_TEST = "xailens.adapters.comparison.comparison_base"

def test_returns_correct_abs_max():
    explanation_a = pd.Series({"feat1": 0.5, "feat2": 0.2})
    explanation_b = pd.Series({"feat1": 0.2, "feat2": 0.5})
    result = ComparisonAdapter.get_abs_max(explanation_a, explanation_b, sample_id_column="id")
    assert result == pytest.approx(3/7)

def test_returns_float():
    explanation_a = pd.Series({"feat1": 0.5, "feat2": 0.2})
    explanation_b = pd.Series({"feat1": 0.2, "feat2": 0.5})
    result = ComparisonAdapter.get_abs_max(explanation_a, explanation_b, sample_id_column="id")
    assert isinstance(result, float)

def test_sample_id_column_excluded():
    """sample_id and class columns should not affect the result"""
    explanation_a = pd.Series({"feat1": 0.5, "feat2": 0.2, "id": 99, "class": 1})
    explanation_b = pd.Series({"feat1": 0.5, "feat2": 0.2, "id": 99, "class": 1})
    result = ComparisonAdapter.get_abs_max(explanation_a, explanation_b, sample_id_column="id")
    assert result == pytest.approx(0.0)

def test_identical_explanations_returns_zero():
    explanation_a = pd.Series({"feat1": 0.5, "feat2": 0.3})
    explanation_b = pd.Series({"feat1": 0.5, "feat2": 0.3})
    result = ComparisonAdapter.get_abs_max(explanation_a, explanation_b, sample_id_column="id")
    assert result == pytest.approx(0.0)

def test_missing_feature_in_one_explanation():
    """fill_value=0 means a missing feature should be treated as zero"""
    explanation_a = pd.Series({"feat1": 0.5, "feat2": 0.5})
    explanation_b = pd.Series({"feat1": 0.5})
    result = ComparisonAdapter.get_abs_max(explanation_a, explanation_b, sample_id_column="id")
    assert result > 0