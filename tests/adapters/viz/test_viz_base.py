"""
Tests for VizAdapter base


Covers:
- _top_n_features
"""

import pytest
import pandas as pd

from concordiahub.adapters.viz.viz_base import VizAdapter


class _FakeVizAdapter(VizAdapter):
    """Minimal concrete subclass so the base class helpers can be tested."""

    key = "fake"
    display_name = "Fake"

    def render(self, explanation, axis_range=None):
        raise NotImplementedError


@pytest.fixture
def adapter():
    return _FakeVizAdapter()


@pytest.fixture
def explanation():
    """Explanation dataframe with mixed signs, unsorted magnitudes"""
    return pd.DataFrame(
        {
            "feature_name": ["age", "sex", "chol", "cp", "thalach", "oldpeak"],
            "score": [0.2, -0.9, 0.5, -0.1, 0.7, -0.3],
            "score_normalized_max": [0.2, -0.9, 0.5, -0.1, 0.7, -0.3],
            "reasoning": [f"reason {i}" for i in range(6)],
        }
    )


def test_top_3_features(adapter, explanation):
    """Tests only top 3 features returned and in correct order"""
    results = adapter._top_features(explanation, 3)

    assert len(results) == 3
    assert results["feature_name"].tolist() == ["sex", "thalach", "chol"]


def test_top_5_features(adapter, explanation):
    """Tests only top 5 features returned and in correct order"""
    results = adapter._top_features(explanation, 5)

    assert len(results) == 5
    assert results["feature_name"].tolist() == ["sex", "thalach", "chol", "oldpeak", "age"]

def test_all_features(adapter, explanation):
    """Tests only top 5 features returned and in correct order"""
    results = adapter._top_features(explanation)

    assert len(results) == 6
    assert results["feature_name"].tolist() == ["sex", "thalach", "chol", "oldpeak", "age", "cp"]