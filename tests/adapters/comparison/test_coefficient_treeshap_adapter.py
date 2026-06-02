import pytest
import pandas as pd
from unittest.mock import MagicMock, patch
from xai_compare.context import ComparisonContext
from xai_compare.adapters.comparison.coefficient_treeshap_adapter import CoefficientTreeShapComparison


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
    return CoefficientTreeShapComparison()


def test_returns_figure(comparison):
    ctx = make_ctx()
    fig = comparison._viz_alignment_scale(ctx)
    assert fig is not None

def test_figure_has_bar_trace(comparison):
    ctx = make_ctx()
    fig = comparison._viz_alignment_scale(ctx)
    assert len(fig.data) == 1
    assert fig.data[0].type == "bar"

def test_bar_values_are_diff(comparison):
    features_a = pd.Series({"feat1": 0.5, "feat2": 0.2})
    features_b = pd.Series({"feat1": 0.2, "feat2": 0.5})
    ctx = make_ctx(features_a=features_a, features_b=features_b)
    fig = comparison._viz_alignment_scale(ctx)
    diff = features_a.sub(features_b).sort_values()
    assert list(fig.data[0].x) == list(diff.values)

def test_axis_range_applied_when_set(comparison):
    ctx = make_ctx(axis_range=[-1, 1])
    fig = comparison._viz_alignment_scale(ctx)
    assert fig.layout.xaxis.range == (-1, 1)

def test_axis_range_not_applied_when_none(comparison):
    ctx = make_ctx(axis_range=None)
    fig = comparison._viz_alignment_scale(ctx)
    assert fig.layout.xaxis.range is None

def test_render_alignment_scale(comparison):
    ctx = make_ctx()
    fig = comparison.render_comparison(ctx, "Alignment Scale")
    assert fig is not None

def test_render_invalid_viz_type_raises(comparison):
    ctx = make_ctx()
    with pytest.raises(KeyError):
        comparison.render_comparison(ctx, "Nonexistent")