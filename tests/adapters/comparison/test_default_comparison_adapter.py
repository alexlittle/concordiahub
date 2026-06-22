import pytest
import plotly.graph_objects as go

from unittest.mock import MagicMock


from xailens.entities.context import ComparisonContext
from xailens.adapters.comparison.default_comparison_adapter import DefaultComparisonAdapter
from tests.helpers import make_explanation

def make_modeleval(model_display_name="model-a", model_type="xgboost", model_subtype=None):
    modeleval = MagicMock()
    modeleval.get = lambda key, default=None: {
        "model_display_name": model_display_name,
        "model_type": model_type,
        "model_subtype": model_subtype,
    }.get(key, default)
    return modeleval

def make_ctx(explanation_a=None, explanation_b=None, axis_range=None):
    """Helper for creating a ComparisonContext"""
    ctx = MagicMock(spec=ComparisonContext)
    ctx.explanation_a = explanation_a if explanation_a is not None else make_explanation({"feat1": 0.5, "feat2": -0.3})
    ctx.explanation_b = explanation_b if explanation_b is not None else make_explanation({"feat1": 0.2, "feat2": -0.1})
    ctx.axis_range = axis_range
    ctx.modeleval_a = make_modeleval("model-a")
    ctx.modeleval_b = make_modeleval("model-b")
    return ctx


@pytest.fixture
def comparison():
    return DefaultComparisonAdapter()


# -------------------------
# General
# -------------------------
def test_render_invalid_viz_type_raises(comparison):
    ctx = make_ctx()
    with pytest.raises(KeyError):
        comparison.render_comparison(ctx, "Nonexistent")


# -------------------------
# Diff bar viz
# -------------------------
def test_diff_bar_returns_figure(comparison):
    ctx = make_ctx()
    fig = comparison.viz_diff_bar(ctx)
    assert fig is not None

def test_diff_bar_figure_has_bar_trace(comparison):
    ctx = make_ctx()
    fig = comparison.viz_diff_bar(ctx)
    assert len(fig.data) == 1
    assert fig.data[0].type == "bar"

def test_diff_bar_values_are_diff(comparison):
    explanation_a = make_explanation({"feat1": 0.5, "feat2": 0.2})
    explanation_b = make_explanation({"feat1": 0.2, "feat2": 0.5})
    ctx = make_ctx(explanation_a=explanation_a, explanation_b=explanation_b)
    fig = comparison.viz_diff_bar(ctx)
    a = explanation_a.set_index("feature_name")["score_normalised_max"]
    b = explanation_b.set_index("feature_name")["score_normalised_max"]
    diff = a.sub(b).sort_values()
    assert list(fig.data[0].x) == list(diff.values)

def test_diff_bar_axis_range_is_dynamic(comparison):
    """Diff bar uses dynamic range based on actual diff values"""
    ctx = make_ctx(axis_range=[-1, 1])
    fig = comparison.viz_diff_bar(ctx)
    assert fig.layout.xaxis.range is not None

def test_diff_bar_axis_range_fits_data(comparison):
    """Axis range should be wider than the actual diff values to include margin"""
    explanation_a = make_explanation({"feat1": 0.5, "feat2": 0.2})
    explanation_b = make_explanation({"feat1": 0.2, "feat2": 0.5})
    ctx = make_ctx(explanation_a=explanation_a, explanation_b=explanation_b)
    fig = comparison.viz_diff_bar(ctx)
    x_min, x_max = fig.layout.xaxis.range
    assert x_min < min(fig.data[0].x)
    assert x_max > max(fig.data[0].x)

def test_diff_bar_render_alignment_scale(comparison):
    ctx = make_ctx()
    fig = comparison.render_comparison(ctx, "diff_bar")
    assert fig is not None

# -------------------------
# Dumbbell viz
# -------------------------
def test_dumbbell_returns_figure(comparison):
    ctx = make_ctx()
    fig = comparison.viz_dumbbell(ctx)
    assert fig is not None

def test_dumbbell_figure_is_scatter(comparison):
    ctx = make_ctx()
    fig = comparison.viz_dumbbell(ctx)
    assert len(fig.data) == 2
    assert fig.data[0].type == "scatter"

def test_dumbbell_has_two_scatter_traces(comparison):
    # Check one trace per model.
    ctx = make_ctx(
        explanation_a=make_explanation({"feat1": 0.5, "feat2": 0.2}),
        explanation_b=make_explanation({"feat1": 0.2, "feat2": 0.5})
    )
    fig = comparison.viz_dumbbell(ctx)
    scatter_traces = [t for t in fig.data if isinstance(t, go.Scatter)]
    assert len(scatter_traces) == 2


def test_dumbbell_trace_values_match_inputs(comparison):
    # Trace x values should be the normalised A and B values
    explanation_a = make_explanation({"feat1": 0.5, "feat2": 0.2})
    explanation_b = make_explanation({"feat1": 0.2, "feat2": 0.5})
    ctx = make_ctx(explanation_a=explanation_a, explanation_b=explanation_b)
    fig = comparison.viz_dumbbell(ctx)
    all_x = set(v for t in fig.data for v in t.x)
    expected_a = set(explanation_a["score_normalised_max"].tolist())
    expected_b = set(explanation_b["score_normalised_max"].tolist())
    assert all_x == expected_a | expected_b

def test_dumbbell_connector_count_matches_features(comparison):
    # Check 1 connector line per feature
    ctx = make_ctx(
        explanation_a=make_explanation({"feat1": 0.5, "feat2": 0.2, "feat3": 0.8}),
        explanation_b=make_explanation({"feat1": 0.2, "feat2": 0.5, "feat3": 0.6})
    )
    fig = comparison.viz_dumbbell(ctx)
    assert len(fig.layout.shapes) == 3

def test_dumbbell_handles_missing_features(comparison):
    ctx = make_ctx(
        explanation_a=make_explanation({"feat1": 0.5, "feat2": 0.2}),
        explanation_b=make_explanation({"feat1": 0.3})
    )
    fig = comparison.viz_dumbbell(ctx)
    assert 0.0 in list(fig.data[0].x) or 0.0 in list(fig.data[1].x)

def test_dumbbell_axis_range_applied_when_set(comparison):
    ctx = make_ctx(axis_range=[-1, 1])
    fig = comparison.viz_dumbbell(ctx)
    assert fig.layout.xaxis.range == (-1, 1)

def test_dumbbell_axis_range_not_applied_when_none(comparison):
    ctx = make_ctx(axis_range=None)
    fig = comparison.viz_dumbbell(ctx)
    assert fig.layout.xaxis.range is None

def test_dumbbell_render_alignment_scale(comparison):
    ctx = make_ctx()
    fig = comparison.render_comparison(ctx, "dumbbell")
    assert fig is not None

