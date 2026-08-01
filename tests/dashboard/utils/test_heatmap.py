"""
Tests for heatmap.py

Covers:
- render_features_heatmap - with and without missing feature alignment
"""
import pytest
import pandas as pd
import plotly.graph_objects as go

from xailens.dashboard.utils.heatmap import render_features_heatmap
from xailens import strings

@pytest.fixture
def df_exp_factory():
    """Dataframe fixture for testing heatmap"""
    def _factory(with_missing=False):
        df = pd.DataFrame({
            "feature_name": ["age", "chol", "thalach"],
            "logistic_regression": [0.9, 0.5, 0.2],
            "xgboost": [0.8, 0.4, 0.3],
            "llama_3_1": [0.7, 0.6, 0.1],
        })
        if with_missing:
            # llama has no importance scores for any feature
            df["llama_3_1"] = [float("nan")] * 3
        return df
    return _factory


def test_heatmap_without_missing_features(df_exp_factory):
    """Heatmap with all features aligned and present in all models"""
    fig = render_features_heatmap(df_exp_factory())

    assert isinstance(fig, go.Figure)
    assert len(fig.data) == 1
    assert list(fig.data[0].x) == ["llama_3_1", "logistic_regression", "xgboost"]
    assert list(fig.data[0].y) == ["age", "chol", "thalach"]
    assert fig.layout.shapes == ()
    assert fig.layout.annotations == ()


def test_heatmap_with_missing_features(df_exp_factory):
    """Heatmap with all features missing for one model"""
    fig = render_features_heatmap(df_exp_factory(with_missing=True))

    assert len(fig.data) == 2

    # fully-missing model is marked with an asterisk
    assert "llama_3_1 *" in list(fig.data[0].x)

    # grey overlay
    grey = fig.data[1]
    assert grey.showscale is False
    assert grey.z[0] == [1, None, None]

    assert len(fig.layout.shapes) == 1
    assert fig.layout.shapes[0].fillcolor == "lightgrey"
    assert len(fig.layout.annotations) == 1
    assert fig.layout.annotations[0].text == strings.HEATMAP_NO_DATA

def test_heatmap_with_some_missing_features(df_exp_factory):
    """Heatmap with some features missing for one model"""
    df = df_exp_factory()
    df.loc[1, "llama_3_1"] = float("nan")

    fig = render_features_heatmap(df)

    assert len(fig.data) == 2
    assert "llama_3_1" in list(fig.data[0].x)
    assert fig.data[1].z[1] == [1, None, None]