"""
Tests for BarChartVizAdapter, specifically the render function.

Tests cover:

- 2 parts returned for LLM reasoning (overall and feature level text)
- single table returned is no overall reasoning
- different combinations of features and number of rows
- verifies columns are renamed and dropped as expected.
"""
import pandas as pd
import pytest
import plotly.graph_objs as go

from xailens.adapters.viz.barchart_viz_adapter import BarChartVizAdapter

from xailens import strings, config



@pytest.fixture
def adapter():
    return BarChartVizAdapter()


# test returns single part that's a plotly figure
def test_returns_reasoning_text_and_renamed_table(adapter, monkeypatch):
    monkeypatch.setattr(config, "PLOT_MAX_FEATURES", 10)

    explanation = pd.DataFrame({
        "feature_name": ["age", "sex", "chol"],
        "score": [0.5, -0.2, 0.1],
        "score_normalised_max": [0.5, -0.2, 0.1],
        "overall_reasoning": ["age matters"] * 3,
        "overall_confidence": ["high"] * 3,
        "reasoning": ["age reason", "sex reason", "chol reason"],
    })

    parts = adapter.render(explanation)

    fig = parts[0]
    assert isinstance(fig, go.Figure)
    assert len(fig.data) == 1
    trace = fig.data[0]
    assert list(trace.y) == ["sex", "chol", "age"] # should be in correct order
    assert list(trace.x) == [-0.2, 0.1, 0.5]
