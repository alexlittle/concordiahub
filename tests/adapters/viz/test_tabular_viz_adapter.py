"""
Tests for TabularAdapter, specifically the render function.

Tests cover:

- 2 parts returned for LLM reasoning (overall and feature level text)
- single table returned is no overall reasoning
- different combinations of features and number of rows
- verifies columns are renamed and dropped as expected.
"""
import pandas as pd
import pytest

from xailens.adapters.viz.tabular_viz_adapter import TabularAdapter

from xailens import strings, config



@pytest.fixture
def adapter():
    return TabularAdapter()


# test returns overall reasoning and a correctly formatted table
# specifically for LLM reasoning where there is overall reasoning text and the
# tabular text
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

    assert len(parts) == 2
    assert "age matters" in parts[0]
    assert "high" in parts[0]

    table = parts[1]
    assert list(table.columns) == ["Feature",
                                   strings.TABULAR_SCORE_TITLE, "Reasoning"]
    # dropped columns shouldn't be returned
    for dropped in ("score", "overall_reasoning", "overall_confidence"):
        assert dropped not in table.columns
    # sorted descending by score title
    assert table[strings.TABULAR_SCORE_TITLE].tolist() == sorted(
        table[strings.TABULAR_SCORE_TITLE].tolist(), reverse=True
    )

# Test that the max number of features is returned as expected
def test_top_n_feature_limit(adapter, monkeypatch):
    monkeypatch.setattr(config, "PLOT_MAX_FEATURES", 2)

    explanation = pd.DataFrame({
        "feature_name": ["age", "sex", "chol"],
        "score": [0.1, 0.2, 0.9],
        "score_normalised_max": [0.1, -0.9, 0.5],
        "overall_reasoning": ["x"] * 3,
        "overall_confidence": ["y"] * 3,
        "reasoning": ["a", "b", "c"],
    })

    parts = adapter.render(explanation)
    table = parts[1]
    # only the top 2, sex and col, should remain, as these are the highest scoring
    assert set(table["Feature"]) == {"sex", "chol"}
    assert len(table) == 2


# test returns a single table without the text reasoning column
def test_returns_single_table_with_reasoning_dropped(adapter, monkeypatch):
    monkeypatch.setattr(config, "PLOT_MAX_FEATURES", 10)

    explanation = pd.DataFrame({
        "feature_name": ["age", "sex", "chol"],
        "score": [0.5, -0.2, 0.1],
        "score_normalised_max": [0.5, -0.2, 0.1],
        "reasoning": ["age reason", "sex reason", "chol reason"],
    })

    parts = adapter.render(explanation)

    assert len(parts) == 1
    table = parts[0]
    assert list(table.columns) == ["Feature", strings.TABULAR_SCORE_TITLE]
    assert "Reasoning" not in table.columns
    assert "reasoning" not in table.columns