import pytest
import pandas as pd
import plotly.graph_objects as go

from unittest.mock import MagicMock


from xailens.context import ComparisonContext
from xailens.adapters.comparison.default_comparison_adapter import DefaultComparisonAdapter


def make_run(model_id="model-a", model_type="xgboost", model_subtype=None):
    run = MagicMock()
    run.get = lambda key, default=None: {
        "model_id": model_id,
        "model_type": model_type,
        "model_subtype": model_subtype,
    }.get(key, default)
    return run



def make_ctx(features_a=None, features_b=None, axis_range=None):
    """Helper for creating a ComparisonContext
    """
    ctx = MagicMock(spec=ComparisonContext)
    ctx.features_a = features_a if features_a is not None else pd.Series({"feat1": 0.5, "feat2": -0.3})
    ctx.features_b = features_b if features_b is not None else pd.Series({"feat1": 0.2, "feat2": -0.1})
    ctx.axis_range = axis_range
    ctx.run_a = make_run("model-a")
    ctx.run_b = make_run("model-b")
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
    features_a = pd.Series({"feat1": 0.5, "feat2": 0.2})
    features_b = pd.Series({"feat1": 0.2, "feat2": 0.5})
    ctx = make_ctx(features_a=features_a, features_b=features_b)
    fig = comparison.viz_diff_bar(ctx)
    diff = features_a.sub(features_b).sort_values()
    assert list(fig.data[0].x) == list(diff.values)

def test_diff_bar_axis_range_applied_when_set(comparison):
    ctx = make_ctx(axis_range=[-1, 1])
    fig = comparison.viz_diff_bar(ctx)
    assert fig.layout.xaxis.range == (-1, 1)

def test_diff_bar_axis_range_not_applied_when_none(comparison):
    ctx = make_ctx(axis_range=None)
    fig = comparison.viz_diff_bar(ctx)
    assert fig.layout.xaxis.range is None

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
    ctx = make_ctx(features_a=pd.Series({"feat1": 0.5, "feat2": 0.2}),
                   features_b=pd.Series({"feat1": 0.2, "feat2": 0.5}))
    fig = comparison.viz_dumbbell(ctx)
    scatter_traces = [t for t in fig.data if isinstance(t, go.Scatter)]
    assert len(scatter_traces) == 2


def test_dumbbell_trace_values_match_inputs(comparison):
    # Trace x values should be the raw A and B values
    features_a = pd.Series({"feat1": 0.5, "feat2": 0.2})
    features_b = pd.Series({"feat1": 0.2, "feat2": 0.5})
    ctx = make_ctx(features_a=features_a, features_b=features_b)
    fig = comparison.viz_dumbbell(ctx)
    all_x = set(v for t in fig.data for v in t.x)
    assert all_x == {0.5, 0.2}


def test_dumbbell_connector_count_matches_features(comparison):
    # Check 1 connector line pre feature
    features_a = pd.Series({"feat1": 0.5, "feat2": 0.2, "feat3": 0.8})
    features_b = pd.Series({"feat1": 0.2, "feat2": 0.5, "feat3": 0.6})
    ctx = make_ctx(features_a=features_a, features_b=features_b)
    fig = comparison.viz_dumbbell(ctx)
    assert len(fig.layout.shapes) == 3


def test_dumbbell_handles_missing_features(comparison):
    # missing features should default to 0
    features_a = pd.Series({"feat1": 0.5, "feat2": 0.2})
    features_b = pd.Series({"feat1": 0.3})
    ctx = make_ctx(features_a=features_a, features_b=features_b)
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

